"""Consistency distillation loss for EDM2.

Ported from edm/training/loss_cd.py with EDM2 adaptations:
- Removed augment_pipe / augment_labels (EDM2 has no augmentation)
- Removed round_fn from grid builders (EDM2 is fully continuous sigma)
- Added __getstate__ / reload_teacher for checkpoint serialization (OD-8)
- Keeps both EDM2-style and pseudo-Huber loss options
"""

import math
import copy
from typing import Optional, Tuple, Dict, List, Any

import torch
import dnnlib
from torch_utils import persistence
from torch_utils import training_stats
from .networks_edm2 import inplace_norm_flag

from .consistency_ops import (
    make_karras_sigmas,
    partition_edges_by_sigma,
    filter_teacher_edges_by_sigma,
    sample_segment_and_teacher_pair,
    heun_hop_edm,
    inv_ddim_edm,
    ddim_step_edm,
)


def _huber_loss(x: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.as_tensor(delta, device=x.device, dtype=x.dtype))
    return 0.5 * (quad * quad) + (abs_x - quad) * delta


def _pseudo_huber_vector_norm(diff: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Pseudo-Huber applied to the Euclidean vector norm (per sample).

    Computes  sqrt(||diff||^2 + eps^2) - eps  where ||·|| is the L2 norm
    across all spatial/channel dims.

    Args:
        diff: [N, C, H, W] tensor of per-pixel differences.
        eps:  Huber smoothing parameter (paper uses 1e-4 for ImageNet).

    Returns:
        [N] tensor of per-sample pseudo-Huber norms.
    """
    norm_sq = (diff * diff).sum(dim=[1, 2, 3])  # [N]
    return torch.sqrt(norm_sq + eps * eps) - eps


@persistence.persistent_class
class EDMConsistencyDistillLoss:
    def __init__(
        self,
        teacher_net,                 # Frozen EDM2 Precond instance
        teacher_pkl_path=None,       # Path to teacher pickle for reload on resume (OD-8)
        S: int = 8,                  # Student steps
        T_start: int = 256,          # Initial teacher edges
        T_end: int = 1024,           # Final teacher edges
        T_anneal_kimg: int = 750,    # Linear anneal horizon (kimg)
        rho: float = 7.0,            # Karras exponent
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        loss_type: str = "pseudo_huber",  # "huber" | "l2" | "l2_root" | "pseudo_huber"
        weight_mode: str = "sqrt_karras",  # "edm" | "vlike" | "flat" | "snr" | "snr+1" | "karras" | "sqrt_karras" | "truncated-snr" | "uniform"
        sigma_data: float = 0.5,
        enable_stats: bool = True,
        debug_invariants: bool = False,
        sampling_mode: str = "vp",   # "uniform" | "vp" | "edm"
        sync_dropout: bool = True,
        terminal_anchor: bool = True,
        terminal_teacher_hop: bool = False,
    ):
        assert S >= 2, "Student steps S must be >= 2"
        assert T_start >= 2 and T_end >= T_start
        assert loss_type in ("huber", "l2", "l2_root", "pseudo_huber")
        assert weight_mode in (
            "edm", "vlike", "flat",
            "snr", "snr+1", "karras", "sqrt_karras", "truncated-snr", "uniform",
        )
        self.teacher_net = teacher_net.eval().requires_grad_(False) if teacher_net is not None else None
        self._teacher_pkl_path = teacher_pkl_path
        self.S = int(S)
        self.T_start = int(T_start)
        self.T_end = int(T_end)
        self.T_anneal_kimg = float(T_anneal_kimg)
        self.rho = float(rho)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.loss_type = loss_type
        self.weight_mode = weight_mode
        self.sigma_data = float(sigma_data)
        self.enable_stats = enable_stats
        self.debug_invariants = debug_invariants
        assert sampling_mode in ("uniform", "vp", "edm"), f"Invalid sampling_mode: {sampling_mode}"
        self.sampling_mode = sampling_mode
        self.terminal_anchor = bool(terminal_anchor)
        self.terminal_teacher_hop = bool(terminal_teacher_hop)
        self.sync_dropout = bool(sync_dropout)

        self._global_kimg = 0.0
        self._count_terminal_edges = 0
        self._count_boundary_match = 0
        self._count_general_edges = 0
        self._count_total_calls = 0
        self._count_total_edges = 0
        self._filter_cache = {}

    def __getstate__(self):
        """Exclude frozen teacher from checkpoint serialization."""
        state = self.__dict__.copy()
        state['teacher_net'] = None
        state['_filter_cache'] = {}
        return state

    def reload_teacher(self, device):
        """Re-attach teacher from original pkl path after checkpoint load."""
        import pickle as _pickle
        with dnnlib.util.open_url(self._teacher_pkl_path) as f:
            data = _pickle.load(f)
        self.teacher_net = data['ema'].eval().requires_grad_(False).to(device)

    def set_run_dir(self, run_dir: str) -> None:
        self._run_dir = run_dir

    def set_global_kimg(self, kimg: float) -> None:
        self._global_kimg = float(kimg)

    def get_edge_stats(self, reset: bool = True) -> dict:
        total_edges = max(self._count_total_edges, 1)
        stats = {
            'total_calls': self._count_total_calls,
            'total_edges': self._count_total_edges,
            'terminal_edges': self._count_terminal_edges,
            'boundary_match': self._count_boundary_match,
            'general_edges': self._count_general_edges,
            'terminal_pct': 100.0 * self._count_terminal_edges / total_edges,
            'boundary_match_pct': 100.0 * self._count_boundary_match / total_edges,
        }
        if reset:
            self._count_terminal_edges = 0
            self._count_boundary_match = 0
            self._count_general_edges = 0
            self._count_total_calls = 0
            self._count_total_edges = 0
        return stats

    def _current_T_edges(self) -> int:
        if self.T_anneal_kimg <= 0:
            return self.T_end
        ratio = min(max(self._global_kimg / self.T_anneal_kimg, 0.0), 1.0)
        log_T_start = math.log(self.T_start)
        log_T_end = math.log(self.T_end)
        log_T_now = log_T_start + ratio * (log_T_end - log_T_start)
        T_now = int(round(math.exp(log_T_now)))
        T_now = max(self.T_start, min(self.T_end, T_now))
        return T_now

    def _build_student_grid(self, net, device: torch.device) -> torch.Tensor:
        sigmas_prepad = make_karras_sigmas(
            num_nodes=self.S,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        ).to(device)
        zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
        sigmas = torch.cat([sigmas_prepad, zero], dim=0)
        return sigmas

    def _build_teacher_grid(self, student_sigmas: torch.Tensor, device: torch.device):
        target_T = self._current_T_edges()

        if target_T in self._filter_cache:
            cached_sigmas, cached_terminal_k = self._filter_cache[target_T]
            return cached_sigmas.to(device), cached_terminal_k

        raw_T = target_T
        while True:
            sigmas_prepad = make_karras_sigmas(
                num_nodes=raw_T,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
            ).to(device)
            zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
            teacher_full = torch.cat([sigmas_prepad, zero], dim=0)

            teacher_filtered, terminal_k = filter_teacher_edges_by_sigma(
                student_sigmas=student_sigmas,
                teacher_sigmas=teacher_full,
            )
            T_eff = teacher_filtered.shape[0] - 1
            if T_eff >= target_T:
                break
            raw_T += 1

        self._filter_cache[target_T] = (teacher_filtered.clone(), terminal_k)
        return teacher_filtered.to(device), terminal_k

    def _weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Per-sample weighting as a function of sigma_t."""
        if self.weight_mode == "edm":
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.weight_mode == "vlike":
            return (1.0 / (sigma ** 2)) + 1.0
        if self.weight_mode == "flat":
            return torch.ones_like(sigma)

        snr = 1.0 / (sigma ** 2 + 1e-20)
        if self.weight_mode == "snr":
            return snr
        if self.weight_mode == "snr+1":
            return snr + 1.0
        if self.weight_mode == "karras":
            return snr + (1.0 / (self.sigma_data ** 2))
        if self.weight_mode == "sqrt_karras":
            return torch.sqrt(sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)
        if self.weight_mode == "truncated-snr":
            return torch.clamp(snr, min=1.0)
        assert self.weight_mode == "uniform"
        return torch.ones_like(sigma)

    def __call__(self, net, images, labels=None):
        """Return per-sample loss tensor. EDM2 adaptation: no augment_pipe/augment_labels.

        `images` are already latents (encoded by the training loop before calling).
        """
        device = images.device
        batch_size = images.shape[0]

        y = images

        student_sigmas = self._build_student_grid(net=net, device=device)

        teacher_sigmas, terminal_k = self._build_teacher_grid(
            student_sigmas=student_sigmas, device=device,
        )
        T_edges = teacher_sigmas.shape[0] - 1

        sigma_bounds = partition_edges_by_sigma(
            student_sigmas=student_sigmas,
            teacher_sigmas=teacher_sigmas,
        )

        sample_dict = sample_segment_and_teacher_pair(
            sigma_bounds=sigma_bounds,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=batch_size,
            device=device,
            terminal_k=terminal_k,
            sampling_mode=self.sampling_mode,
            rho=self.rho,
            terminal_anchor=self.terminal_anchor,
        )

        j = sample_dict["step_j"].long()
        n_rel = sample_dict["n_rel"].long()
        sigma_t_vec = sample_dict["sigma_t"]
        sigma_s_teacher_vec = sample_dict["sigma_s"]
        sigma_bdry_vec = sample_dict["sigma_bdry"]
        is_terminal = sample_dict["is_terminal"].bool()
        is_boundary_snap = sample_dict["is_boundary_snap"].bool()
        is_general = (~is_terminal) & (~is_boundary_snap)

        sigma_s_eff = sigma_s_teacher_vec.clone()
        sigma_s_eff = torch.where(is_boundary_snap, sigma_bdry_vec, sigma_s_eff)
        sigma_s_eff = torch.where(
            is_general,
            torch.maximum(sigma_s_teacher_vec, sigma_bdry_vec),
            sigma_s_eff,
        )

        sigma_t = sigma_t_vec.to(torch.float64).view(batch_size, 1, 1, 1)
        sigma_s = sigma_s_eff.to(torch.float64).view(batch_size, 1, 1, 1)
        sigma_bdry = sigma_bdry_vec.to(torch.float64).view(batch_size, 1, 1, 1)

        tol = 1e-8
        sigma_upper_vec = student_sigmas[j]
        interior_mask = (j > 0) & (~is_terminal)
        if interior_mask.any():
            assert (sigma_t_vec[interior_mask] < sigma_upper_vec[interior_mask] - tol).all(), (
                "Ordering: sigma_t must be strictly < segment upper boundary (interior segments)"
            )
        assert (sigma_t_vec > sigma_s_teacher_vec + tol).all(), (
            "Ordering: sigma_t must be strictly > sigma_s (raw)"
        )
        non_term = ~is_terminal
        if non_term.any():
            assert (sigma_t_vec[non_term] > sigma_s_eff[non_term] + tol).all(), (
                "Ordering: sigma_t must be strictly > sigma_s_eff for non-terminal"
            )
        assert (sigma_s_eff >= sigma_bdry_vec - tol).all(), (
            "Ordering: sigma_s_eff must be >= sigma_bdry"
        )

        if self.debug_invariants:
            assert (sigma_s_eff[is_terminal] == 0).all(), "Terminal edges must have sigma_s_eff == 0"
            assert (is_terminal & is_boundary_snap).sum() == 0, "Terminal and boundary_snap must be disjoint"
            assert (n_rel[is_boundary_snap] == 1).all(), "Boundary snap edges must have n_rel == 1"
            if is_boundary_snap.any():
                assert (sigma_t_vec[is_boundary_snap] > sigma_bdry_vec[is_boundary_snap] + tol).all(), (
                    "Boundary snap: sigma_t must be strictly > sigma_bdry"
                )

        eps = torch.randn_like(y).to(torch.float64)
        y64 = y.to(torch.float64)
        x_t = y64 + sigma_t * eps

        non_terminal = ~is_terminal
        boundary_mask = is_boundary_snap
        general_mask = (~is_terminal) & (~is_boundary_snap)

        x_s_teach = torch.zeros_like(x_t)
        x_ref_bdry = torch.zeros_like(x_t)
        sigma_ref_vec = sigma_t_vec.new_zeros(batch_size).to(torch.float64)
        tol = 1e-12

        if non_terminal.any():
            idx = non_terminal
            with torch.no_grad():
                x_s_teach_nt = heun_hop_edm(
                    net=self.teacher_net,
                    x_t=x_t[idx],
                    sigma_t=sigma_t_vec[idx],
                    sigma_s=sigma_s_eff[idx],
                    class_labels=labels[idx] if labels is not None else None,
                )
            x_s_teach[idx] = x_s_teach_nt

        if self.terminal_teacher_hop and is_terminal.any():
            with torch.no_grad():
                x_ref_bdry[is_terminal] = self.teacher_net(
                    x_t[is_terminal].float(),
                    sigma_t_vec[is_terminal],
                    labels[is_terminal] if labels is not None else None,
                ).to(torch.float64)
        else:
            x_ref_bdry[is_terminal] = y64[is_terminal]
        sigma_ref_vec[is_terminal] = 0.0

        x_ref_bdry[boundary_mask] = x_s_teach[boundary_mask]
        sigma_ref_vec[boundary_mask] = sigma_s_eff[boundary_mask]

        if self.sync_dropout:
            rng_state = torch.cuda.get_rng_state()

        x_hat_t = net(x_t.float(), sigma_t, labels).to(torch.float32)

        if general_mask.any():
            with torch.no_grad():
                # Disable in-place forced weight normalization on the no-grad
                # pass so that MPConv doesn't mutate weights as a side-effect.
                # The traditional WN path still normalizes in the forward graph.
                token = inplace_norm_flag.set(False)
                try:
                    if self.sync_dropout:
                        torch.cuda.set_rng_state(rng_state)
                        target_x = x_t.clone()
                        target_x[general_mask] = x_s_teach[general_mask]
                        target_sigma = sigma_t.clone()
                        target_sigma[general_mask] = sigma_s[general_mask]
                        x_hat_full = net(
                            target_x.float(), target_sigma, labels,
                        ).to(torch.float64)
                        x_hat_s_ng = x_hat_full[general_mask]
                    else:
                        net.eval()
                        x_hat_s_ng = net(
                            x_s_teach[general_mask].float(),
                            sigma_s[general_mask],
                            labels[general_mask] if labels is not None else None,
                        ).to(torch.float64)
                        net.train()
                finally:
                    inplace_norm_flag.reset(token)

            ratio_s_b = sigma_bdry[general_mask] / torch.clamp(sigma_s[general_mask], min=tol)
            x_ref_bdry[general_mask] = x_hat_s_ng + ratio_s_b * (x_s_teach[general_mask] - x_hat_s_ng)
            sigma_ref_vec[general_mask] = sigma_bdry_vec[general_mask]

        ratio_ref = sigma_ref_vec / torch.clamp(sigma_t_vec, min=1e-12)
        gain = 1.0 / torch.clamp(1.0 - ratio_ref, min=1e-6)

        try:
            x_hat_t_star = inv_ddim_edm(
                x_ref=x_ref_bdry,
                x_t=x_t,
                sigma_t=sigma_t_vec,
                sigma_ref=sigma_ref_vec,
            ).to(torch.float32)
        except ValueError as e:
            error_msg = str(e) + "\n\n  Sampling context for affected samples (first 5):\n"
            bad_idx = (torch.abs(sigma_ref_vec - sigma_t_vec) < 1e-8).nonzero(as_tuple=False).view(-1)[:5]
            for idx in bad_idx:
                i = int(idx.item())
                error_msg += (
                    f"    Sample {i}: seg_j={j[i].item()}, n_rel={n_rel[i].item()}, "
                    f"terminal={is_terminal[i].item()}, boundary_snap={is_boundary_snap[i].item()}\n"
                    f"              sigma_t={sigma_t_vec[i].item():.9f}, "
                    f"sigma_s_eff={sigma_s_eff[i].item():.9f}, "
                    f"sigma_bdry={sigma_bdry_vec[i].item():.9f}, "
                    f"sigma_ref={sigma_ref_vec[i].item():.9f}\n"
                )
            raise ValueError(error_msg) from e

        weight = self._weight(sigma_t_vec.to(torch.float32))
        weight = weight.view(batch_size, 1, 1, 1)
        diff = x_hat_t - x_hat_t_star
        if self.loss_type == "huber":
            per_elem = _huber_loss(diff)
            loss = weight * per_elem
        elif self.loss_type == "pseudo_huber":
            per_sample = _pseudo_huber_vector_norm(diff, eps=1e-4)
            per_elem = per_sample.view(batch_size, 1, 1, 1)
            loss = weight * per_elem
        elif self.loss_type == "l2_root":
            per_sample = torch.sqrt(torch.clamp((diff * diff).sum(dim=[1, 2, 3]), min=1e-12))
            per_elem = per_sample.view(batch_size, 1, 1, 1)
            loss = weight * per_elem
        else:
            per_elem = diff * diff
            loss = weight * per_elem

        num_terminal = int(is_terminal.sum().item())
        num_boundary = int(is_boundary_snap.sum().item())
        num_general = int((~is_terminal & ~is_boundary_snap).sum().item())
        num_edges = num_terminal + num_boundary + num_general

        if self.debug_invariants:
            assert num_edges == batch_size, f"Edge counts don't sum to batch_size: {num_edges} != {batch_size}"

        self._count_total_calls += 1
        self._count_total_edges += num_edges
        self._count_terminal_edges += num_terminal
        self._count_boundary_match += num_boundary
        self._count_general_edges += num_general

        if self.enable_stats:
            with torch.no_grad():
                training_stats.report('Loss/cd', loss)
                training_stats.report('CD/sigma_t', sigma_t_vec.mean())
                training_stats.report('CD/sigma_s', sigma_s_eff.mean())
                training_stats.report('CD/sigma_bdry', sigma_bdry_vec.mean())
                training_stats.report('CD/seg_id', j.float().mean())
                training_stats.report('CD/T_edges', torch.as_tensor(float(T_edges), device=device))

                training_stats.report('CD/gain_mean', gain.mean())
                training_stats.report('CD/gain_max', gain.max())
                training_stats.report('CD/gain_95p', gain.quantile(0.95))
                training_stats.report('CD/gain_99p', gain.quantile(0.99))

                gain_terminal = gain[is_terminal]
                gain_boundary = gain[is_boundary_snap]
                gain_general = gain[general_mask]

                training_stats.report('CD/gain_all', gain)
                training_stats.report('CD/gain_terminal_mean', gain_terminal.mean() if gain_terminal.numel() > 0 else [])
                training_stats.report('CD/gain_boundary_mean', gain_boundary.mean() if gain_boundary.numel() > 0 else [])
                training_stats.report('CD/gain_general_mean', gain_general.mean() if gain_general.numel() > 0 else [])

                loss_mean_per_sample = loss.mean(dim=(1, 2, 3))
                training_stats.report('CD/loss_mean', loss_mean_per_sample.mean())

                per_sample_l2 = (diff * diff).sum(dim=[1, 2, 3])
                per_sample_l2_sqrt = torch.sqrt(per_sample_l2.clamp(min=1e-12))
                training_stats.report('CD/l2_error_all', per_sample_l2_sqrt.mean())

                training_stats.report('CD/frac_terminal', torch.as_tensor(float(num_terminal) / max(num_edges, 1), device=device))
                training_stats.report('CD/frac_boundary', torch.as_tensor(float(num_boundary) / max(num_edges, 1), device=device))
                training_stats.report('CD/frac_general', torch.as_tensor(float(num_general) / max(num_edges, 1), device=device))

        return loss
