# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models",
with optional Multi-Step Consistency Distillation (MSCD) support."""

import os
import re
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-img512-xxs':  dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=64,  lr=0.0170, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-xs':    dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=128, lr=0.0120, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-img512-s', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.get('cond', True))
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Detect CD mode.
    is_cd = bool(opts.get('teacher'))

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)

    # Network: override dropout for CD mode if the --dropout flag was explicitly set.
    net_dropout = opts.dropout
    if is_cd and opts.get('cd_dropout') is not None:
        net_dropout = opts.cd_dropout
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels, dropout=net_dropout)
    dout_res = opts.get('dout_resolutions')
    if dout_res is not None:
        c.network_kwargs.dout_resolutions = dout_res
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # DataLoader workers.
    workers = opts.get('workers', 2)
    c.data_loader_kwargs = dnnlib.EasyDict(
        class_name='torch.utils.data.DataLoader',
        pin_memory=True,
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None,
    )

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.phema_snapshot_nimg = opts.get('phema_snap', 0) or None
    c.checkpoint_keep_recent = int(opts.get('checkpoint_keep_recent', 3))
    c.checkpoint_cleanup_snapshots = not bool(opts.get('no_checkpoint_snapshot_prune', False))
    c.seed = opts.get('seed', 0)

    # Resume from explicit checkpoint.
    # Unlike EDM1, EDM2's CheckpointIO saves ALL state (net, optimizer, ema, ema_val, cur_nimg)
    # in a single .pt file, so only resume_state_dump is needed — no separate pkl is required.
    import re as _re
    resume_pt = opts.get('resume')
    if resume_pt is not None:
        if not _re.fullmatch(r'training-state-(\d+)\.pt', os.path.basename(resume_pt)):
            raise click.ClickException('--resume must point to a training-state-*.pt file from a previous run')
        if not os.path.isfile(resume_pt):
            raise click.ClickException(f'--resume: file not found: {resume_pt}')
        c.resume_state_dump = resume_pt

    # CD-specific configuration.
    if is_cd:
        c.teacher_pkl = opts['teacher']
        # NOTE: Click 8.x lowercases all parameter names derived from option strings,
        # so --S → 's', --T_start → 't_start', --T_end → 't_end', --T_anneal_kimg → 't_anneal_kimg'.
        cd_S = opts.get('s', 8)
        c.cd_kwargs = dict(
            S=cd_S,
            T_start=opts.get('t_start', 256),
            T_end=opts.get('t_end', 1024),
            T_anneal_kimg=opts.get('t_anneal_kimg', 750),
            rho=7.0,
            sigma_min=opts.get('sigma_min', 0.002),
            sigma_max=opts.get('sigma_max', 80.0),
            loss_type=opts.get('cd_loss', 'pseudo_huber'),
            weight_mode=opts.get('cd_weight_mode', 'sqrt_karras'),
            sigma_data=0.5,
            sampling_mode=opts.get('sampling_mode', 'vp'),
            terminal_anchor=opts.get('terminal_anchor', True),
            terminal_teacher_hop=opts.get('terminal_teacher_hop', False),
            sync_dropout=opts.get('sync_dropout', True),
        )

        # EMA for validation (OD-2).
        c.ema_halflife_kimg = opts.get('ema_halflife_kimg', 500.0)
        c.ema_rampup_ratio = opts.get('ema_rampup', 0.05) or None

        # LR overrides for CD (OD-7).
        if opts.get('cd_lr') is not None:
            c.lr_kwargs['ref_lr'] = opts['cd_lr']
        if opts.get('cd_decay') is not None:
            c.lr_kwargs['ref_batches'] = opts['cd_decay']

    # Validation configuration (in-training FID).
    val_ref = opts.get('val_ref')
    if val_ref is not None:
        default_val_steps = opts.get('s', 8) if is_cd else 32
        # CD: default to Euler (num_steps == num_NFEs); base training: default to Heun.
        default_use_heun = not is_cd
        c.validation_kwargs = dnnlib.EasyDict(
            enabled=True,
            ref=val_ref,
            every=opts.get('val_every', 1),
            num_images=opts.get('val_num', 50000),
            steps=opts.get('val_steps') or default_val_steps,
            seed=opts.get('val_seed', 0),
            batch=opts.get('val_batch', 32),
            sigma_min=opts.get('sigma_min', 0.002),
            sigma_max=opts.get('sigma_max', 80.0),
            rho=7.0,
            at_start=opts.get('val_at_start', False),
            use_heun=opts.get('val_heun', default_use_heun),
        )

    # Weights & Biases configuration.
    if opts.get('wandb', False):
        tags = opts.get('wandb_tags')
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        c.wandb_kwargs = dnnlib.EasyDict(
            enabled=True,
            project=opts.get('wandb_project', 'edm2-cd'),
            entity=opts.get('wandb_entity', None),
            name=opts.get('wandb_run', None),
            tags=tags,
            mode=opts.get('wandb_mode', 'online'),
        )

    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    if c.get('teacher_pkl'):
        dist.print0(f'CD mode:                 True')
        dist.print0(f'Teacher:                 {c.teacher_pkl}')
        dist.print0(f'CD params:               S={c.cd_kwargs["S"]}, T={c.cd_kwargs["T_start"]}->{c.cd_kwargs["T_end"]}')
        dist.print0(f'CD loss:                 {c.cd_kwargs["loss_type"]} / {c.cd_kwargs["weight_mode"]}')
    snap_kimg  = (c.snapshot_nimg or 0) // 1000
    phema_kimg = (c.get('phema_snapshot_nimg') or c.snapshot_nimg or 0) // 1000
    dist.print0(f'Snapshot interval:       {snap_kimg} kimg  |  phEMA interval: {phema_kimg} kimg')
    if c.get('validation_kwargs') and c.validation_kwargs.get('enabled'):
        vk = c.validation_kwargs
        steps = vk.get('steps', 8)
        use_heun = vk.get('use_heun', False)
        nfe = (2 * steps - 1) if use_heun else steps
        dist.print0(f'Validation:              FID every {vk.get("every",1)} snapshot(s), {vk.get("num_images",50000)} images, {steps} steps ({nfe} NFEs, {"heun" if use_heun else "euler"})')
    if c.get('wandb_kwargs') and c.wandb_kwargs.get('enabled'):
        dist.print0(f'W&B:                     project={c.wandb_kwargs.get("project")}, entity={c.wandb_kwargs.get("entity")}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

def parse_int_list(s):
    """Parse a comma-separated list of ints, e.g. '16,8' -> [16, 8]."""
    if s is None:
        return None
    if isinstance(s, list):
        return s
    return [int(x.strip()) for x in s.split(',') if x.strip()]

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='edm2-img512-s', show_default=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',           type=click.IntRange(min=0), default=2, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',                type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots (ema_val / base)', metavar='NIMG', type=parse_nimg, default='8Mi', show_default=True)
@click.option('--phema_snap',       help='Interval of phEMA snapshots (default: same as --snapshot)', metavar='NIMG', type=parse_nimg, default=None, show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',       type=parse_nimg, default='128Mi', show_default=True)
@click.option('--checkpoint_keep_recent', help='Retain N newest training-state .pt plus best-FID .pt', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--no_checkpoint_snapshot_prune', help='Keep all primary network-snapshot-{kimg}.pkl (phEMA *-* files are never pruned)', is_flag=True, default=False)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--resume',           help='Resume from training-state-*.pt', metavar='PT',       type=str, default=None)
@click.option('--nosubdir',         help='Do not create a numbered subdirectory inside --outdir', is_flag=True)
@click.option('--desc',             help='String to include in the output directory name',      metavar='STR', type=str, default=None)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

# ── Teacher / CD core ──
@click.option('--teacher',          help='Teacher EDM2 pickle (enables CD mode)', metavar='PKL|URL', type=str, default=None)
@click.option('--S',                help='Student step count', type=click.IntRange(min=2), default=8, show_default=True)
@click.option('--T_start',          help='Initial teacher edges', type=click.IntRange(min=2), default=256, show_default=True)
@click.option('--T_end',            help='Final teacher edges', type=click.IntRange(min=2), default=1024, show_default=True)
@click.option('--T_anneal_kimg',    help='Teacher edge annealing horizon (kimg)', type=click.IntRange(min=0), default=750, show_default=True)
@click.option('--cd_loss',          help='CD loss type', type=click.Choice(['huber','l2','l2_root','pseudo_huber']), default='pseudo_huber', show_default=True)
@click.option('--cd_weight_mode',   help='CD loss weight mode', type=click.Choice(['edm','sqrt_karras','flat','snr','karras','uniform']), default='sqrt_karras', show_default=True)
@click.option('--sampling_mode',    help='Edge sampling distribution', type=click.Choice(['uniform','vp','edm']), default='vp', show_default=True)
@click.option('--terminal_anchor/--no_terminal_anchor', help='Anchor terminal edge to 1/T probability', default=True, show_default=True)
@click.option('--terminal_teacher_hop/--no_terminal_teacher_hop', help='Use teacher hop for terminal edge instead of clean image', default=False, show_default=True)

# ── Sigma grid bounds (OD-5) ──
@click.option('--sigma_min',        help='Min sigma for CD Karras grids', type=float, default=0.002, show_default=True)
@click.option('--sigma_max',        help='Max sigma for CD Karras grids', type=float, default=80.0, show_default=True)

# ── Dropout (OD-6) ──
@click.option('--cd_dropout',       help='Student dropout for CD (overrides preset)', type=click.FloatRange(min=0, max=1), default=None)
@click.option('--sync_dropout/--no_sync_dropout', help='Sync CUDA RNG for dropout', default=True, show_default=True)
@click.option('--dout_resolutions', help='Apply dropout only at these resolutions (e.g. 16,8). None = all.', type=parse_int_list, default=None)

# ── LR overrides for CD (OD-7) ──
@click.option('--cd_lr',            help='CD-mode ref_lr override. None = use preset LR.', type=float, default=None)
@click.option('--cd_decay',         help='CD-mode ref_batches override. 0 = constant LR after rampup.', type=float, default=None)

# ── EMA for validation (OD-2) ──
@click.option('--ema_halflife_kimg', help='Halflife of exponential validation EMA (kimg)', type=float, default=500.0, show_default=True)
@click.option('--ema_rampup',       help='EMA rampup ratio (0 = no rampup)', type=float, default=0.05, show_default=True)

# ── FID validation (OD-4) ──
@click.option('--val_ref',          help='FID reference stats (.npz or URL)', metavar='NPZ|URL', type=str, default=None)
@click.option('--val_every',        help='Validate every N snapshots', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--val_num',          help='Images for FID evaluation', type=int, default=50000, show_default=True)
@click.option('--val_steps',        help='Sampler steps for validation (None = S)', type=int, default=None)
@click.option('--val_seed',         help='Validation base seed', type=int, default=0, show_default=True)
@click.option('--val_batch',        help='Validation batch size per GPU', type=int, default=32, show_default=True)
@click.option('--val_at_start',     help='Run validation at first snapshot', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--val_heun',         help='Use Heun (2nd-order) sampler for val; default False for CD, True for base', metavar='BOOL', type=bool, default=None)

# ── Weights & Biases ──
@click.option('--wandb',            help='Enable W&B logging', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--wandb_project',    help='W&B project name', type=str, default='edm2-cd', show_default=True)
@click.option('--wandb_entity',     help='W&B entity (user/team)', type=str, default=None)
@click.option('--wandb_run',        help='W&B run name', type=str, default=None)
@click.option('--wandb_tags',       help='W&B tags (comma-separated)', type=str, default=None)
@click.option('--wandb_mode',       help='W&B mode', type=click.Choice(['online','offline','disabled']), default='online', show_default=True)

def cmdline(outdir, dry_run, nosubdir, desc, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models",
    with optional Multi-Step Consistency Distillation (MSCD) support.

    Examples:

    \b
    # Train XS-sized model for ImageNet-512 using 8 GPUs (creates training-runs/00000-edm2-img512-s-...)
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-s \\
        --batch-gpu=32

    \b
    # Consistency distillation from a pre-trained teacher
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-s \\
        --teacher=path/to/teacher.pkl \\
        --S=8 --cd_loss=pseudo_huber \\
        --batch-gpu=32

    \b
    # Use --nosubdir to write directly into --outdir (e.g. for explicit resume paths).
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)

    # Determine run directory (rank-0 only; broadcast to others via the training loop).
    # Mirrors EDM1: by default creates a numbered subdirectory inside --outdir so that
    # re-submitting the same script never clobbers a previous run.
    if nosubdir:
        run_dir = outdir
    else:
        # Build a short description string from key config fields.
        data_name = os.path.splitext(os.path.basename(opts.get('data', 'data')))[0]
        cond_str  = 'cond' if c.dataset_kwargs.get('use_labels', False) else 'uncond'
        dtype_str = 'fp16' if c.network_kwargs.get('use_fp16', False) else 'fp32'
        preset    = opts.get('preset', 'custom')
        gpus      = dist.get_world_size()
        batch     = c.batch_size
        auto_desc = f'{data_name}-{cond_str}-{preset}-gpus{gpus}-batch{batch}-{dtype_str}'
        if c.get('teacher_pkl'):
            cd_S     = c.cd_kwargs.get('S', 8)
            cd_Ts    = c.cd_kwargs.get('T_start', 64)
            cd_Te    = c.cd_kwargs.get('T_end', 1280)
            auto_desc += f'-cdS{cd_S}-T{cd_Ts}-{cd_Te}'
        if desc is not None:
            auto_desc += f'-{desc}'

        # Find the next available run ID (same logic as EDM1).
        if dist.get_rank() == 0:
            prev_run_dirs = []
            if os.path.isdir(outdir):
                prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
            prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            cur_run_id   = max(prev_run_ids, default=-1) + 1
            run_dir      = os.path.join(outdir, f'{cur_run_id:05d}-{auto_desc}')
            assert not os.path.exists(run_dir), f'Run directory already exists: {run_dir}'
        else:
            run_dir = None

        # Broadcast run_dir from rank 0 to all other ranks.
        run_dir_list = [run_dir]
        torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]

    print_training_config(run_dir=run_dir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=run_dir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
