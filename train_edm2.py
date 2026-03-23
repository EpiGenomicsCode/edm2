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

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)

    # CD-specific configuration.
    if is_cd:
        c.teacher_pkl = opts['teacher']
        cd_S = opts.get('S', 8)
        c.cd_kwargs = dict(
            teacher_pkl_path=opts['teacher'],
            S=cd_S,
            T_start=opts.get('T_start', 256),
            T_end=opts.get('T_end', 1024),
            T_anneal_kimg=opts.get('T_anneal_kimg', 750),
            rho=7.0,
            sigma_min=opts.get('sigma_min', 0.002),
            sigma_max=opts.get('sigma_max', 80.0),
            loss_type=opts.get('cd_loss', 'pseudo_huber'),
            weight_mode=opts.get('cd_weight_mode', 'sqrt_karras'),
            sigma_data=0.5,
            sampling_mode=opts.get('sampling_mode', 'vp'),
            terminal_anchor=opts.get('terminal_anchor', True),
            terminal_teacher_hop=False,
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

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
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

def cmdline(outdir, dry_run, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models",
    with optional Multi-Step Consistency Distillation (MSCD) support.

    Examples:

    \b
    # Train XS-sized model for ImageNet-512 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs/00000-edm2-img512-xs \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-xs \\
        --batch-gpu=32

    \b
    # Consistency distillation from a pre-trained teacher
    torchrun --standalone --nproc_per_node=8 train_edm2.py \\
        --outdir=training-runs/00001-cd-img512-s \\
        --data=datasets/img512-sd.zip \\
        --preset=edm2-img512-s \\
        --teacher=path/to/teacher.pkl \\
        --S=8 --cd_loss=pseudo_huber \\
        --batch-gpu=32

    \b
    # To resume training, run the same command again.
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
