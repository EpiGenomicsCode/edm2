# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Distributed FID validation hook for in-training evaluation.
Adapted from the EDM validation system for EDM2's latent-space pipeline."""

import os
import math
import json
import pickle
import numpy as np
import torch
import scipy.linalg
import dnnlib
from typing import Optional, Dict, Any

from torch_utils import distributed as dist
from torch_utils import misc
import tqdm

from generate_images import edm_sampler, StackedRandomGenerator

#----------------------------------------------------------------------------
# Pure first-order Euler sampler (S_churn=0, no 2nd-order correction).
# Used for CD validation so that num_steps == num_NFEs, matching EDM1 behavior.

def _euler_sampler(net, noise, labels, *, num_steps, sigma_min, sigma_max, rho, randn_like):
    dtype = torch.float32
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x = noise.to(dtype) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        d = (x - net(x, t_cur, labels).to(dtype)) / t_cur
        x = x + (t_next - t_cur) * d
    return x

#----------------------------------------------------------------------------

def _load_inception_detector(device: torch.device):
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    local_path = os.environ.get('EDM_INCEPTION_PATH', None)
    if local_path is None:
        repo_local = os.path.join(os.path.dirname(__file__), 'metrics', 'inception-2015-12-05.pkl')
        if os.path.isfile(repo_local):
            local_path = repo_local

    if local_path is not None and os.path.isfile(local_path):
        dist.print0(f'Loading Inception-v3 model from local file "{local_path}"...')
        with open(local_path, 'rb') as f:
            detector_net = pickle.load(f).to(device)
    else:
        dist.print0('Loading Inception-v3 model...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
            detector_net = pickle.load(f).to(device)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    return detector_net, detector_kwargs, feature_dim

#----------------------------------------------------------------------------

def _prepare_reference_stats(ref: Optional[str], *, device: torch.device):
    """Load reference Inception mu/sigma from .npz (EDM format) or .pkl (EDM2 format)."""
    if ref is None:
        raise RuntimeError('Validation requires --val_ref (.npz or .pkl path/URL).')
    mu_ref = None
    sigma_ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref) as f:
            if ref.lower().endswith('.npz'):
                data = dict(np.load(f))
                mu_ref = data['mu']
                sigma_ref = data['sigma']
            else:
                data = pickle.load(f)
                if 'fid' in data and isinstance(data['fid'], dict):
                    mu_ref = data['fid']['mu']
                    sigma_ref = data['fid']['sigma']
                else:
                    mu_ref = data['mu']
                    sigma_ref = data['sigma']
    return mu_ref, sigma_ref

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

def run_fid_validation(
    net: torch.nn.Module,
    encoder,
    *,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    num_images: int = 50000,
    batch: int = 32,
    seed: int = 0,
    num_steps: int = 8,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    ref: Optional[str] = None,
    step_kimg: Optional[int] = None,
    use_heun: bool = False,
    wandb_run=None,
) -> Dict[str, Any]:
    """Compute FID by generating images with the given network (ema_val).

    use_heun=False (default): pure Euler ODE, num_steps == num_NFEs. Matches EDM1
      CD validation and is correct for evaluating a consistency model student.
    use_heun=True: 2nd-order Heun correction, NFEs = 2*num_steps-1. Appropriate
      for evaluating a full diffusion model (e.g. base EDM2 training).
    """
    device = torch.device('cuda')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    net = net.eval().requires_grad_(False).to(device)

    nfe = num_steps if not use_heun else (2 * num_steps - 1)
    sampler_label = f'heun({nfe} NFEs)' if use_heun else f'euler({nfe} NFEs)'

    mu_ref, sigma_ref = _prepare_reference_stats(ref, device=device)
    detector, detector_kwargs, feature_dim = _load_inception_detector(device)

    all_indices = torch.arange(num_images, device=torch.device('cpu'))
    num_batches = math.ceil(num_images / (batch * world_size)) * world_size
    all_batches = all_indices.tensor_split(num_batches)
    rank_batches = all_batches[rank :: world_size]

    dist.print0(f'[VAL] Starting: num_images={num_images}, batches={num_batches}, batch_per_gpu={batch}, steps={num_steps}, sampler={sampler_label}')

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    label_dim = getattr(net, 'label_dim', 0)
    use_labels = bool(label_dim and dataset_kwargs.get('use_labels', False))

    rank_batches_list = list(rank_batches)
    non_empty = sum(1 for b in rank_batches_list if len(b) > 0)
    progress = tqdm.tqdm(rank_batches_list, unit='batch', disable=(rank != 0), ascii=True, mininterval=5.0)
    local_idx = 0

    for b_idxs in progress:
        bsize = len(b_idxs)
        if bsize == 0:
            continue
        local_idx += 1
        if rank == 0 and (local_idx == 1 or local_idx % 10 == 0 or local_idx == non_empty):
            pct = 100.0 * local_idx / max(non_empty, 1)
            dist.print0(f'[VAL] Progress (rank0): {local_idx}/{non_empty} ({pct:.1f}%)')

        seeds = (seed + b_idxs).tolist()
        rnd = StackedRandomGenerator(device, seeds)
        noise = rnd.randn([bsize, net.img_channels, net.img_resolution, net.img_resolution], device=device)

        class_labels = None
        if use_labels:
            class_labels = torch.eye(label_dim, device=device)[rnd.randint(label_dim, size=[bsize], device=device)]

        if use_heun:
            latents = edm_sampler(
                net, noise, labels=class_labels,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                randn_like=rnd.randn_like,
            )
        else:
            latents = _euler_sampler(
                net, noise, labels=class_labels,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                randn_like=rnd.randn_like,
            )

        images_u8 = encoder.decode(latents)

        if images_u8.shape[1] == 1:
            images_u8 = images_u8.repeat([1, 3, 1, 1])
        feats = detector(images_u8, **detector_kwargs).to(torch.float64)
        mu += feats.sum(0)
        sigma += feats.T @ feats

    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= (num_images - 1)

    fid_value = None
    if dist.get_rank() == 0:
        fid_value = calculate_fid_from_inception_stats(mu.cpu().numpy(), sigma.cpu().numpy(), mu_ref, sigma_ref)
        result = dict(
            kimg=int(step_kimg) if step_kimg is not None else None,
            fid=float(fid_value),
            num_images=int(num_images),
        )
        try:
            import time
            with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
                f.write(json.dumps(dict(result, timestamp=time.time())) + '\n')
        except Exception:
            with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
                f.write(json.dumps(result) + '\n')
        if wandb_run is not None:
            try:
                import wandb as _wandb
                _wandb.log({'val/fid': float(fid_value), 'val/num_images': int(num_images), 'progress_kimg': int(step_kimg) if step_kimg is not None else None}, commit=True)
            except Exception:
                pass
        dist.print0(f'[VAL] kimg={step_kimg} FID={fid_value:g}')
    torch.distributed.barrier()
    return {'fid': float(fid_value) if fid_value is not None else None}

#----------------------------------------------------------------------------

def maybe_validate(
    *,
    cur_nimg: int,
    snapshot_nimg: int,
    net_ema: torch.nn.Module,
    encoder,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    validation_kwargs: Optional[Dict[str, Any]],
    wandb_run=None,
):
    """Lightweight scheduler called at each snapshot boundary."""
    if validation_kwargs is None:
        return
    if not validation_kwargs.get('enabled', False):
        return
    every = int(validation_kwargs.get('every', 1))
    if snapshot_nimg is None or snapshot_nimg == 0:
        return

    snapshot_idx = cur_nimg // snapshot_nimg
    at_start = bool(validation_kwargs.get('at_start', False))

    should_run = False
    if snapshot_idx == 0:
        should_run = at_start
    elif every > 0 and (snapshot_idx % every == 0):
        should_run = True

    if not should_run:
        return

    flag = torch.tensor([1 if should_run else 0], dtype=torch.int64, device=torch.device('cuda'))
    torch.distributed.broadcast(flag, src=0)
    if int(flag.item()) == 0:
        return

    return run_fid_validation(
        net_ema,
        encoder,
        run_dir=run_dir,
        dataset_kwargs=dataset_kwargs,
        num_images=int(validation_kwargs.get('num_images', 50000)),
        batch=int(validation_kwargs.get('batch', 32)),
        seed=int(validation_kwargs.get('seed', 0)),
        num_steps=int(validation_kwargs.get('steps', 8)),
        sigma_min=float(validation_kwargs.get('sigma_min', 0.002)),
        sigma_max=float(validation_kwargs.get('sigma_max', 80.0)),
        rho=float(validation_kwargs.get('rho', 7.0)),
        ref=validation_kwargs.get('ref', None),
        step_kimg=int(cur_nimg // 1000),
        use_heun=bool(validation_kwargs.get('use_heun', False)),
        wandb_run=wandb_run,
    )

#----------------------------------------------------------------------------
