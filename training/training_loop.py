# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop with optional Consistency Distillation (CD) support."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc

from validation import maybe_validate

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),

    # CD-specific parameters (all None/default = normal diffusion training).
    teacher_pkl         = None,     # Path to teacher checkpoint. None = normal training.
    cd_kwargs           = None,     # Dict of CD hyperparams (S, T_start, T_end, etc.)
    ema_halflife_kimg   = 500.0,    # Halflife of exponential validation EMA (kimg) [CD only]
    ema_rampup_ratio    = 0.05,     # EMA rampup ratio [CD only]

    # Validation (FID).
    validation_kwargs   = None,     # Dict of validation params. None = no in-training validation.

    # Weights & Biases.
    wandb_kwargs        = None,     # Dict of W&B params. None = no W&B logging.
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Determine if we're in CD mode.
    is_cd_mode = (teacher_pkl is not None)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # CD setup: load teacher, construct CD loss, seed student.
    teacher_net = None
    if is_cd_mode:
        dist.print0(f'Loading teacher from {teacher_pkl}...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(teacher_pkl, verbose=(dist.get_rank() == 0)) as f:
            teacher_data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        teacher_net = teacher_data['ema'].eval().requires_grad_(False).to(device)

        # Encoder consistency check (OD-3).
        teacher_encoder = teacher_data.get('encoder', None)
        if teacher_encoder is not None:
            teacher_encoder_cls = type(teacher_encoder).__name__
            student_encoder_cls = encoder_kwargs['class_name'].split('.')[-1]
            if teacher_encoder_cls != student_encoder_cls:
                raise RuntimeError(
                    f'Encoder mismatch: teacher uses {teacher_encoder_cls!r} but '
                    f'student config specifies {student_encoder_cls!r}. '
                    f'Teacher and student must share the same latent space.'
                )

        # Seed student from teacher if shapes match.
        try:
            misc.copy_params_and_buffers(src_module=teacher_net, dst_module=net, require_all=False)
            dist.print0('[CD INIT] Seeded student from teacher.')
        except Exception as e:
            dist.print0(f'[CD INIT] Could not seed from teacher: {e}')

        del teacher_data

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)

    # DDP: tuned kwargs for CD mode, default for base training.
    if is_cd_mode:
        ddp = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[device],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
        )
    else:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])

    # Construct loss function.
    if is_cd_mode:
        from training.loss_cd import EDMConsistencyDistillLoss
        loss_fn = EDMConsistencyDistillLoss(
            teacher_net=teacher_net,
            teacher_pkl_path=teacher_pkl,
            **(cd_kwargs or {}),
        )
    else:
        loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)

    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)

    # EMA setup.
    # For CD mode: run two EMA copies in parallel (OD-2).
    #   1) phema (PowerFunctionEMA) — unchanged from base EDM2, for post-hoc reconstruction.
    #   2) ema_val — standard exponential EMA for FID validation and snapshots.
    # For base training: only phema (existing behavior).
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None
    ema_val = None
    if is_cd_mode:
        ema_val = copy.deepcopy(net).eval().requires_grad_(False)
        # If student was seeded from teacher, ema_val starts from teacher weights too.
        dist.print0(f'[CD EMA] Validation EMA: halflife={ema_halflife_kimg} kimg, rampup={ema_rampup_ratio}')

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)

    # Re-attach teacher after checkpoint load (OD-8).
    if is_cd_mode and hasattr(loss_fn, 'teacher_net') and loss_fn.teacher_net is None:
        loss_fn.reload_teacher(device)
        dist.print0('[CD RESUME] Re-attached teacher after checkpoint load.')

    # W&B initialization.
    wandb_run = None
    if wandb_kwargs is not None and wandb_kwargs.get('enabled', False) and dist.get_rank() == 0:
        try:
            import wandb as _wandb
            try:
                import sys
                for stream in [sys.stdout, sys.stderr]:
                    if not hasattr(stream, 'isatty'):
                        stream.isatty = lambda: False
            except Exception:
                pass
            init_kwargs = dict(
                project=wandb_kwargs.get('project', 'edm2-cd'),
                entity=wandb_kwargs.get('entity', None),
                name=wandb_kwargs.get('name', None),
                tags=wandb_kwargs.get('tags', None),
            )
            mode = wandb_kwargs.get('mode', 'online')
            if mode in ('offline', 'disabled'):
                init_kwargs['mode'] = mode
            wandb_run = _wandb.init(**init_kwargs)
            try:
                _wandb.save(os.path.join(run_dir, 'log.txt'), policy='live')
            except Exception:
                pass
        except Exception as _e:
            dist.print0(f'[W&B] init failed: {_e}')
            wandb_run = None

    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # W&B tick-level logging.
            if wandb_run is not None and dist.get_rank() == 0:
                try:
                    import wandb as _wandb
                    log_dict = {name: value.mean for name, value in training_stats.default_collector.as_dict().items() if np.isfinite(value.mean)}
                    log_dict['progress_kimg'] = state.cur_nimg / 1e3
                    _wandb.log(log_dict, commit=True)
                except Exception as _e:
                    dist.print0(f'[W&B] log failed: {_e}')

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            if is_cd_mode:
                # CD mode: save ema_val as the primary EMA snapshot.
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
                data.ema = copy.deepcopy(ema_val).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data

                # Also save phema snapshots if enabled.
                if ema is not None:
                    ema_list = ema.get() if isinstance(ema.get(), list) else [(ema.get(), '')]
                    for ema_net, ema_suffix in ema_list:
                        data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
                        data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                        fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                        dist.print0(f'Saving {fname} ... ', end='', flush=True)
                        with open(os.path.join(run_dir, fname), 'wb') as f:
                            pickle.dump(data, f)
                        dist.print0('done')
                        del data
            else:
                # Base training: save phema snapshots (original behavior).
                ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                    dist.print0(f'Saving {fname} ... ', end='', flush=True)
                    with open(os.path.join(run_dir, fname), 'wb') as f:
                        pickle.dump(data, f)
                    dist.print0('done')
                    del data

        # In-training FID validation (runs at snapshot boundaries).
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0):
            try:
                val_net = ema_val if ema_val is not None else net
                maybe_validate(
                    cur_nimg=state.cur_nimg,
                    snapshot_nimg=snapshot_nimg,
                    net_ema=val_net,
                    encoder=encoder,
                    run_dir=run_dir,
                    dataset_kwargs=dataset_kwargs,
                    validation_kwargs=validation_kwargs,
                    wandb_run=wandb_run,
                )
            except Exception as _e:
                dist.print0(f'[VAL] validation failed: {_e}')

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Teacher annealing (CD only).
        if is_cd_mode and hasattr(loss_fn, 'set_global_kimg'):
            loss_fn.set_global_kimg(state.cur_nimg / 1e3)

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)

        # Update validation EMA (CD mode only, OD-2).
        if ema_val is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None and ema_rampup_ratio > 0:
                ema_halflife_nimg = min(ema_halflife_nimg, state.cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema_val.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cumulative_training_time += time.time() - batch_start_time

    # Close W&B run.
    try:
        if wandb_run is not None and dist.get_rank() == 0:
            import wandb as _wandb
            _wandb.finish()
    except Exception:
        pass

#----------------------------------------------------------------------------
