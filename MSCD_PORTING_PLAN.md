# MSCD Porting Plan: EDM → EDM2

This document is a comprehensive, file-by-file plan for porting the **Multi-Step
Consistency Distillation (MSCD)** implementation from the `edm/` codebase into
`edm2/`.  Every section below maps an EDM source to the corresponding EDM2
target, calls out every non-trivial adaptation, and flags open design decisions.

---

## Table of Contents

1. [High-Level Architecture Delta](#1-high-level-architecture-delta)
2. [New Files to Create](#2-new-files-to-create)
   - 2.1 `training/consistency_ops.py`
   - 2.2 `training/loss_cd.py`
3. [Files to Modify](#3-files-to-modify)
   - 3.1 `training/networks_edm2.py` — add `round_sigma` + expose `sigma_min`/`sigma_max`
   - 3.2 `training/training_loop.py` — CD-aware training loop
   - 3.3 `train_edm2.py` — CLI + CD wiring
   - 3.4 `generate_images.py` — persistence hooks
4. [Detailed Component Adaptations](#4-detailed-component-adaptations)
   - 4.1 Teacher loading & the encoder pipeline
   - 4.2 Preconditioning differences (`round_sigma`, `sigma_min/max`)
   - 4.3 Loss function: EDM2Loss logvar vs CD loss
   - 4.4 Student/teacher forward signature (`augment_labels` removal)
   - 4.5 EMA: exponential half-life vs PowerFunctionEMA
   - 4.6 LR schedule
   - 4.7 Dropout sync (CUDA RNG)
   - 4.8 DDP configuration
   - 4.9 Checkpoint / resume system
   - 4.10 Validation / FID hooks
5. [Consistency Ops: Adaptation Notes](#5-consistency-ops-adaptation-notes)
6. [Loss CD: Line-by-Line Adaptation Notes](#6-loss-cd-line-by-line-adaptation-notes)
7. [Training Loop: Exact Changelist](#7-training-loop-exact-changelist)
8. [train_edm2.py: Exact Changelist](#8-train_edm2py-exact-changelist)
9. [Testing & Validation Checklist](#9-testing--validation-checklist)
10. [Open Design Decisions](#10-open-design-decisions)

---

## 1. High-Level Architecture Delta

| Aspect | EDM | EDM2 | Impact on MSCD Port |
|--------|-----|------|---------------------|
| **Network** | `EDMPrecond` wrapping `DhariwalUNet` or `SongUNet` | `Precond` wrapping magnitude-preserving `UNet` | Teacher & student are both EDM2 `Precond`. Need `round_sigma`/`sigma_min`/`sigma_max` stubs. |
| **Forward signature** | `net(x, sigma, class_labels, augment_labels=...)` | `net(x, sigma, class_labels)` — no `augment_labels` | Strip `augment_labels` from all CD code paths. |
| **Encoder pipeline** | None; raw pixels scaled inline `/ 127.5 - 1` | `encoder.encode_latents(images.to(device))` — may be VAE | Teacher Heun hops and student evals operate in **latent space**. `inv_ddim_edm` targets are latents, not pixels. Terminal anchor `y` = clean latent (not pixel). |
| **Loss** | `EDMLoss`: fixed weight, no logvar | `EDM2Loss`: logvar head, uncertainty-weighted | CD loss replaces EDM2Loss entirely; **logvar head is unused** during CD. |
| **Augmentation** | `AugmentPipe` with `augment_dim=9` | None (no augmentation in EDM2) | Simplification: remove all `augment_pipe` / `augment_labels` from CD loss. |
| **EMA** | Exponential half-life EMA (`ema_halflife_kimg`) | `PowerFunctionEMA` (power-function profiles, multiple stds) | CD must call `ema.update(cur_nimg=..., batch_size=...)` per step, same as base EDM2. |
| **LR schedule** | Linear rampup then constant | `learning_rate_schedule()`: rampup + 1/sqrt decay | Keep EDM2's LR schedule for CD; it's better for long runs. |
| **Checkpoint** | Manual `pickle.dump` + `torch.save` | `dist.CheckpointIO` (auto-resume from `run_dir`) | CD teacher is NOT checkpointed (frozen, loaded once). Student + optimizer + EMA use existing `CheckpointIO`. |
| **Progress units** | `kimg` (thousands of images) | `nimg` (raw image count) | Teacher annealing `set_global_kimg()` needs nimg→kimg conversion. |
| **DDP** | Custom kwargs (bucket_cap_mb, etc.) | Default `DistributedDataParallel` | May want to adopt EDM's DDP tuning. |

---

## 2. New Files to Create

### 2.1 `training/consistency_ops.py`

**Source:** `edm/training/consistency_ops.py` (370 lines)

**Copy verbatim with these changes:**

| Function | Change needed |
|----------|---------------|
| `make_karras_sigmas()` | `round_fn` parameter: EDM2's `Precond` has no `round_sigma`. Add a no-op default: `round_fn = round_fn or (lambda x: x)`. This makes the function work with both EDM and EDM2 networks. |
| `filter_teacher_edges_by_sigma()` | No change. Pure tensor math. |
| `partition_edges_by_sigma()` | No change. Pure tensor math. |
| `compute_importance_weights()` | No change. Pure tensor math. |
| `sample_segment_and_teacher_pair()` | No change. Pure tensor math. |
| `ddim_step_edm()` | No change. Pure tensor math. |
| `inv_ddim_edm()` | No change. Pure tensor math. |
| `heun_hop_edm()` | **KEY CHANGE**: Remove `net.round_sigma(...)` calls. EDM2's `Precond` does not have `round_sigma`. Replace with identity or add a `getattr(net, 'round_sigma', lambda x: x)` fallback. Also: remove `augment_labels` kwarg — EDM2 `Precond.forward()` doesn't accept it. |
| `heun_hop_edm_stochastic()` | Copy as backward-compat alias (or omit if not needed). |

**Detailed changes for `heun_hop_edm`:**

```python
# EDM version:
sigma_t_r = net.round_sigma(torch.as_tensor(sigma_t, device=x64.device))
sigma_s_r = net.round_sigma(torch.as_tensor(sigma_s, device=x64.device))
denoised_t = net(x64.float(), sigma_t_r, class_labels=class_labels,
                 augment_labels=augment_labels).to(torch.float64)

# EDM2 version:
sigma_t_r = torch.as_tensor(sigma_t, device=x64.device)
sigma_s_r = torch.as_tensor(sigma_s, device=x64.device)
denoised_t = net(x64.float(), sigma_t_r,
                 class_labels=class_labels).to(torch.float64)
```

### 2.2 `training/loss_cd.py`

**Source:** `edm/training/loss_cd.py` (889 lines)

**Copy with the following adaptations** (each numbered, detailed in §6):

| Area | EDM | EDM2 adaptation |
|------|-----|-----------------|
| **Import** | `from .consistency_ops import ...` | Same (new file from 2.1). |
| **`@persistence.persistent_class`** | Uses EDM's persistence | Same — EDM2 also has `torch_utils.persistence`. |
| **Constructor: `teacher_net`** | Accepts any EDM-preconditioned net | Must accept EDM2 `Precond` instance. No `round_sigma` → see `_build_student_grid` / `_build_teacher_grid`. |
| **`_build_student_grid()`** | Calls `net.round_sigma` via fallback | EDM2 `Precond` has no `round_sigma`. Use identity lambda: `round_fn = getattr(net_unwrapped, 'round_sigma', lambda x: x)`. |
| **`_build_teacher_grid()`** | Calls `self.teacher_net.round_sigma` | Same fix: `getattr(self.teacher_net, 'round_sigma', lambda x: x)`. |
| **`__call__` — augmentation** | `y, augment_labels = augment_pipe(images) if augment_pipe ...` | EDM2 has no augmentation. Remove `augment_pipe` parameter entirely. `y = images` directly. |
| **`__call__` — teacher hop** | `heun_hop_edm(..., augment_labels=...)` | Remove `augment_labels` kwarg. |
| **`__call__` — student forward** | `net(x_t.float(), sigma_t, labels, augment_labels=augment_labels)` | `net(x_t.float(), sigma_t, labels)` — no augment_labels. |
| **`__call__` — nograd student** | Same pattern with `augment_labels` | Remove `augment_labels`. |
| **`__call__` — terminal anchor** | `x_ref_bdry[is_terminal] = y64[is_terminal]` | `y64` is the **clean latent** (not pixel). This is correct as-is since EDM2 operates in latent space; `y` = encoder output. |
| **`__call__` — terminal teacher hop** | `self.teacher_net(x_t[...].float(), sigma_t_vec[...], labels[...], augment_labels=...)` | Remove `augment_labels`. |
| **Diagnostic `_save_image_grid`** | Saves pixel grids | May need to decode latents first for visualization, or just remove this debug helper. |

---

## 3. Files to Modify

### 3.1 `training/networks_edm2.py`

**Add `round_sigma` and `sigma_min`/`sigma_max` to `Precond`:**

The MSCD code (both `consistency_ops.py` and `loss_cd.py`) calls
`net.round_sigma(...)` and reads `net.sigma_min` / `net.sigma_max`. EDM's
`EDMPrecond` has these; EDM2's `Precond` does not.

**Option A (recommended):** Add no-op stubs to `Precond`:

```python
class Precond(torch.nn.Module):
    def __init__(self, ...):
        ...
        self.sigma_min = 0            # no clamping
        self.sigma_max = float('inf') # no clamping

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
```

This matches `EDMPrecond.round_sigma` exactly and is the least invasive change.

**Option B:** Make `consistency_ops.py` never call `round_sigma` / `sigma_min` /
`sigma_max` on the network and instead accept them as explicit arguments. More
refactoring but cleaner separation.

**Recommendation:** Option A — it's 4 lines and keeps CD code identical.

### 3.2 `training/training_loop.py`

This is the largest change. The EDM2 training loop is ~240 lines; the EDM CD
training loop is ~745 lines. We need to weave the CD-specific logic into the
EDM2 loop without breaking the base diffusion training path.

See §7 for the exact changelist.

### 3.3 `train_edm2.py`

Add CLI options and wiring for CD mode. See §8 for the exact changelist.

### 3.4 `generate_images.py`

Add persistence import hooks so that pickled snapshots containing
`training.loss_cd.EDMConsistencyDistillLoss` and `training.consistency_ops.*`
can be loaded:

```python
# At top of file, after imports:
from torch_utils import persistence
persistence.import_hook('training.loss_cd')
persistence.import_hook('training.consistency_ops')
```

Also add these hooks to `reconstruct_phema.py`.

---

## 4. Detailed Component Adaptations

### 4.1 Teacher Loading & the Encoder Pipeline

**EDM:** Teacher is loaded from a pickle containing `{'ema': <EDMPrecond>}`.
Images are raw pixels scaled to [-1, 1] inline.

**EDM2:** Teacher will be loaded from an EDM2 pickle containing
`{'ema': <Precond>, 'encoder': <Encoder>}`. Images go through
`encoder.encode_latents()` before being fed to the network.

**Adaptation:**
- In `train_edm2.py`, when `--consistency=True`:
  1. Load teacher pickle.
  2. Extract `teacher_data['ema']` as the teacher network.
  3. The encoder from the teacher pickle should match the student's encoder.
     Assert or warn if different.
- In `loss_cd.py`, the `images` passed to `__call__` are **already latents**
  (encoded by the training loop before calling the loss). So `y = images` is
  the clean latent. This is correct — no pixel-level operations needed.
- Teacher Heun hops input/output latents, not pixels. This is fine since both
  teacher and student share the same latent space.

### 4.2 Preconditioning Differences

**EDM `EDMPrecond`:**
```python
self.sigma_min = 0
self.sigma_max = float('inf')
def round_sigma(self, sigma): return torch.as_tensor(sigma)
```

**EDM2 `Precond`:**
- No `sigma_min`, `sigma_max`, or `round_sigma`.
- Same c_skip/c_out/c_in/c_noise formulas as EDM.

**Resolution:** Add the 3 stubs to `Precond` (§3.1). The preconditioning math
is identical — `D(x) = c_skip·x + c_out·F(x)` — so no functional change.

### 4.3 Loss Function: EDM2Loss logvar vs CD loss

**EDM2Loss** (current):
```python
denoised, logvar = net(..., return_logvar=True)
loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
```

**CD loss** (port):
```python
x_hat_t = net(x_t.float(), sigma_t, labels)  # no return_logvar
loss = weight * loss_fn(x_hat_t - x_hat_t_star)
```

The CD loss does **not** use the logvar head. When the student is called,
`return_logvar=False` (the default). The logvar branch in `Precond.forward()`
is simply never triggered.

**No changes to `Precond` needed** — `return_logvar` defaults to `False`.

### 4.4 Student/Teacher Forward Signature

**EDM:**
```python
net(x, sigma, class_labels=..., augment_labels=..., force_fp32=...)
```

**EDM2:**
```python
net(x, sigma, class_labels=..., force_fp32=..., return_logvar=...)
```

**Key difference:** EDM2 has **no `augment_labels`** parameter. The CD code
must strip all `augment_labels=` kwargs from:
- `heun_hop_edm()` calls to `net(...)`
- Student forward `net(x_t.float(), sigma_t, labels)`
- Nograd student forward
- Terminal teacher hop `self.teacher_net(...)`

This is purely mechanical — delete every `augment_labels=...` argument.

### 4.5 EMA: Exponential Half-Life vs PowerFunctionEMA

**EDM CD:** Uses standard exponential EMA with `ema_halflife_kimg` and optional
rampup. Single EMA network. Updated manually per step:
```python
ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
for p_ema, p_net in zip(ema.parameters(), net.parameters()):
    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
```

**EDM2:** Uses `PowerFunctionEMA` from `training/phema.py`. Multiple EMA
profiles tracked simultaneously. Updated per step:
```python
ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
```

**Adaptation:** Keep EDM2's `PowerFunctionEMA` for CD. It's strictly more
general (you get multiple EMA snapshots at different stds for post-hoc
reconstruction). The CD loss itself doesn't care about EMA — it only sees the
live `net` (with grad) and the frozen `teacher_net`.

**Decision needed:** What EMA `stds` to use for CD snapshots? Defaults in EDM2
are typically `[0.050, 0.100]` or similar. The EDM CD branch used
`ema_halflife_kimg=500` (i.e., `ema=0.5`). The EDM PHEMA adaptation in the
edm branch (`training/phema.py`) already supports this. Keep EDM2 defaults or
let the user override via CLI.

### 4.6 LR Schedule

**EDM CD:**
```python
lr = base_lr * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
```
Linear rampup then constant.

**EDM2:**
```python
lr = ref_lr / sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
```
Rampup + 1/sqrt decay.

**Recommendation:** Keep EDM2's `learning_rate_schedule()` for CD. The 1/sqrt
decay is well-suited for long distillation runs. Expose `ref_lr` and
`ref_batches` as CLI overrides for CD.

### 4.7 Dropout Sync (CUDA RNG)

**EDM CD `sync_dropout`:** Saves/restores `torch.cuda.get_rng_state()` so the
no-grad student target pass has identical dropout masks to the student forward.

**EDM2:** The `Block` class in `networks_edm2.py` uses
`torch.nn.functional.dropout(y, p=self.dropout)` in its forward. This is
standard PyTorch dropout and is controlled by the same CUDA RNG.

**Adaptation:** The sync_dropout mechanism ports directly. The RNG save/restore
pattern works identically:
```python
if self.sync_dropout:
    rng_state = torch.cuda.get_rng_state()
# ... student forward ...
if self.sync_dropout:
    torch.cuda.set_rng_state(rng_state)
# ... nograd target forward ...
```

**Caveat:** EDM2's `Block.forward()` applies dropout via:
```python
if self.training and self.dropout != 0:
    y = torch.nn.functional.dropout(y, p=self.dropout)
```
The no-grad target pass must run with `.train()` mode (not `.eval()`) for
dropout to fire, matching the EDM CD approach. The sync_dropout full-batch
trick (run full batch through net to consume same RNG) works the same way.

### 4.8 DDP Configuration

**EDM CD:** Tuned DDP kwargs:
```python
DistributedDataParallel(net, device_ids=[device],
    broadcast_buffers=False, gradient_as_bucket_view=True,
    static_graph=False, find_unused_parameters=False, bucket_cap_mb=100)
```

**EDM2:** Default DDP:
```python
DistributedDataParallel(net, device_ids=[device])
```

**Recommendation:** Adopt EDM's tuned DDP kwargs for CD mode (the teacher
forward + multiple student forwards per step benefit from larger bucket caps
and gradient-as-bucket-view). Make this conditional on CD mode to avoid
regressing the base training path.

### 4.9 Checkpoint / Resume System

**EDM:** Manual pickle + torch.save/load. Resume logic in training loop.

**EDM2:** `dist.CheckpointIO` — automatic checkpoint discovery and resume from
`run_dir`.

**Adaptation:**
- The frozen teacher is **not** part of `CheckpointIO`. It's loaded once at
  startup from the user-provided `--teacher` path.
- The CD loss object (`EDMConsistencyDistillLoss`) holds a reference to the
  teacher but should be excluded from checkpoint serialization (it's huge and
  frozen). On resume, the teacher is re-loaded from the original path.
- The student `net`, `optimizer`, `ema`, and `loss_fn` (minus teacher) are
  checkpointed normally via `CheckpointIO`.

**Implementation detail:** `CheckpointIO` pickles `loss_fn`. Since
`EDMConsistencyDistillLoss` holds `teacher_net` as an attribute, we need to
either:
  - (a) Override `__getstate__` to exclude `teacher_net` and re-attach on load, or
  - (b) Store teacher path in the loss object and re-load on `__setstate__`, or
  - (c) Don't include `loss_fn` in `CheckpointIO` when in CD mode; reconstruct
    it on resume.

**Recommendation:** Option (a) — add `__getstate__`/`__setstate__` to
`EDMConsistencyDistillLoss` that serializes `teacher_net` as `None` and
requires re-attachment after unpickling.

### 4.10 Validation / FID Hooks

**EDM CD:** Full FID validation system (`validation.py`) with teacher baseline,
student EMA sampling, Inception-v3 features, distributed reduce.

**EDM2:** No built-in validation during training. FID is computed externally
via `calculate_metrics.py` after generating images.

**Decision:** Port the validation system or not?
- **Minimum viable:** Skip validation for the initial port. Users can generate
  images with `generate_images.py` and compute FID externally.
- **Full port:** Adapt `validation.py` to EDM2 (needs encoder decode, different
  sampler interface, EDM2's `generate_images.edm_sampler`).

**Recommendation:** Skip for initial port. Add as a follow-up. This is large
and not core to MSCD training correctness.

---

## 5. Consistency Ops: Adaptation Notes

File: `edm2/training/consistency_ops.py` (new file, ported from EDM)

### Functions — no change needed (pure math)
- `make_karras_sigmas()` — make `round_fn` default to identity
- `filter_teacher_edges_by_sigma()`
- `partition_edges_by_sigma()`
- `compute_importance_weights()`
- `sample_segment_and_teacher_pair()`
- `_expand_sigma_to_bchw()`
- `ddim_step_edm()`
- `inv_ddim_edm()`

### Functions — changes needed

**`heun_hop_edm(net, x_t, sigma_t, sigma_s, class_labels, augment_labels)`**

Changes:
1. Remove `augment_labels` parameter.
2. Replace `net.round_sigma(...)` with `getattr(net, 'round_sigma', lambda x: torch.as_tensor(x, device=...))`.
   Or, since we add `round_sigma` to `Precond` (§3.1), this becomes a no-op
   change (the fallback is still good practice).
3. Remove `augment_labels=augment_labels` from both `net(...)` calls.

New signature:
```python
def heun_hop_edm(net, x_t, sigma_t, sigma_s, class_labels=None):
```

---

## 6. Loss CD: Line-by-Line Adaptation Notes

File: `edm2/training/loss_cd.py` (new file, ported from EDM)

### Constructor `__init__`

| Parameter | EDM | EDM2 change |
|-----------|-----|-------------|
| `teacher_net` | Any EDM-precond net | EDM2 `Precond` instance |
| All sigma/rho/S/T params | Same | Same |
| `sigma_data` | 0.5 | 0.5 (same) |

No constructor changes needed beyond ensuring the teacher is an EDM2 Precond.

### `_build_student_grid(self, net, device)`

```python
# EDM:
round_fn = getattr(net, 'round_sigma', None)
if round_fn is None and hasattr(net, 'module'):
    round_fn = getattr(net.module, 'round_sigma', None)

# EDM2: Same code works IF we add round_sigma to Precond (§3.1).
# If not, add fallback:
if round_fn is None:
    round_fn = lambda x: torch.as_tensor(x)
```

### `_build_teacher_grid(self, student_sigmas, device)`

```python
# EDM:
round_fn=self.teacher_net.round_sigma

# EDM2: Same if we add round_sigma to Precond.
# Safety: round_fn=getattr(self.teacher_net, 'round_sigma', lambda x: torch.as_tensor(x))
```

### `__call__(self, net, images, labels=None, augment_pipe=None)`

**Change signature to:** `__call__(self, net, images, labels=None)`

Remove `augment_pipe` parameter entirely.

Line-by-line changes:

| Line(s) | EDM code | EDM2 change |
|---------|----------|-------------|
| 365 | `y, augment_labels = augment_pipe(images) if augment_pipe ...` | `y = images` (latents, already encoded by training loop) |
| 489 | `augment_labels=augment_labels[idx] if augment_labels is not None else None` in `heun_hop_edm()` | Remove this kwarg |
| 502-507 | `self.teacher_net(x_t[...].float(), ..., augment_labels=...)` | Remove `augment_labels` kwarg |
| 522 | `net(x_t.float(), sigma_t, labels, augment_labels=augment_labels)` | `net(x_t.float(), sigma_t, labels)` |
| 536-539 | `net(target_x.float(), target_sigma, labels, augment_labels=augment_labels)` | `net(target_x.float(), target_sigma, labels)` |
| 543-548 | `net(x_s_teach[...].float(), ..., augment_labels=...)` | Remove `augment_labels` kwarg |
| 361 | `loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)` (call site in training loop) | `loss = loss_fn(net=ddp, images=images, labels=labels)` |

### Debug helper `_save_image_grid`

This function saves pixel images. In EDM2, the data is latents (possibly VAE
latents). Options:
- (a) Remove the function entirely (it's for debugging only).
- (b) Keep it but note it will produce nonsensical images for VAE latents.
- (c) Accept an encoder and decode before saving.

**Recommendation:** Remove it. It's not essential.

---

## 7. Training Loop: Exact Changelist

File: `edm2/training/training_loop.py`

The current EDM2 training loop is clean and minimal (~240 lines). We need to
add CD-specific logic conditionally, triggered by the loss function type.

### 7.1 Add to function signature

```python
def training_loop(
    ...,
    # Existing params unchanged.
    # New CD params:
    teacher_pkl         = None,     # Path to teacher checkpoint. None = normal training.
    cd_kwargs           = None,     # Dict of CD hyperparams (S, T_start, T_end, etc.)
):
```

### 7.2 After network construction: teacher loading

```python
# After: net.train().requires_grad_(True).to(device)

# CD setup: load teacher, construct CD loss, seed student.
is_cd_mode = (teacher_pkl is not None)
teacher_net = None
if is_cd_mode:
    dist.print0(f'Loading teacher from {teacher_pkl}...')
    with dnnlib.util.open_url(teacher_pkl, verbose=(dist.get_rank() == 0)) as f:
        teacher_data = pickle.load(f)
    teacher_net = teacher_data['ema'].eval().requires_grad_(False).to(device)

    # Seed student from teacher if shapes match.
    try:
        misc.copy_params_and_buffers(src_module=teacher_net, dst_module=net, require_all=False)
        dist.print0('[CD INIT] Seeded student from teacher.')
    except Exception as e:
        dist.print0(f'[CD INIT] Could not seed from teacher: {e}')
```

### 7.3 Loss function construction

```python
# CD mode: override loss_kwargs with CD loss.
if is_cd_mode:
    from training.loss_cd import EDMConsistencyDistillLoss
    loss_fn = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        **cd_kwargs,
    )
else:
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
```

### 7.4 In the training loop body: teacher annealing

```python
# Before optimizer.zero_grad():
if is_cd_mode and hasattr(loss_fn, 'set_global_kimg'):
    loss_fn.set_global_kimg(state.cur_nimg / 1e3)
```

### 7.5 Forward pass: no augmentation for CD

The current EDM2 loop does:
```python
images = encoder.encode_latents(images.to(device))
loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
```

For CD, the call is the same signature:
```python
loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
```

No change needed here because our `EDMConsistencyDistillLoss.__call__` has the
same `(net, images, labels)` signature as `EDM2Loss.__call__`.

### 7.6 DDP tuning for CD

```python
if is_cd_mode:
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=100,
    )
else:
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
```

### 7.7 Gradient sanitization

EDM2 already has `force_finite` nan-to-num. Keep it. EDM CD also does this.
No change.

### 7.8 Snapshot: exclude teacher from pickle

When saving snapshots, the `loss_fn` is included. For CD, the teacher inside
`loss_fn` is huge. Either:
- Don't save `loss_fn` in CD mode, or
- Implement `__getstate__` on `EDMConsistencyDistillLoss` to exclude teacher.

```python
# In snapshot saving:
if is_cd_mode:
    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
else:
    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
    data.ema = ...
```

### 7.9 CheckpointIO: handle CD loss

`CheckpointIO` saves `loss_fn`. For CD, add `__getstate__` to exclude teacher:

```python
# In EDMConsistencyDistillLoss:
def __getstate__(self):
    state = self.__dict__.copy()
    state['teacher_net'] = None  # don't serialize frozen teacher
    state['_filter_cache'] = {}  # clear cache
    return state
```

On resume, re-attach teacher after loading checkpoint.

---

## 8. train_edm2.py: Exact Changelist

### 8.1 New CLI options

```python
# Consistency Distillation options
@click.option('--teacher',          help='Teacher checkpoint (EDM2 pickle)', metavar='PKL|URL', type=str, default=None)
@click.option('--S',                help='Student steps', metavar='INT', type=click.IntRange(min=2), default=8, show_default=True)
@click.option('--T_start',         help='Initial teacher edges', metavar='INT', type=click.IntRange(min=2), default=256, show_default=True)
@click.option('--T_end',           help='Final teacher edges', metavar='INT', type=click.IntRange(min=2), default=1024, show_default=True)
@click.option('--T_anneal_kimg',   help='Anneal horizon (kimg)', metavar='INT', type=click.IntRange(min=0), default=750, show_default=True)
@click.option('--cd_loss',         help='CD loss type', metavar='STR', type=click.Choice(['huber','l2','l2_root','pseudo_huber']), default='pseudo_huber', show_default=True)
@click.option('--cd_weight_mode',  help='CD weight mode', metavar='STR', type=click.Choice(['edm','sqrt_karras','flat','snr','karras','uniform']), default='sqrt_karras', show_default=True)
@click.option('--sampling_mode',   help='Edge sampling', metavar='STR', type=click.Choice(['uniform','vp','edm']), default='vp', show_default=True)
@click.option('--sync_dropout/--no_sync_dropout', default=True)
@click.option('--terminal_anchor/--no_terminal_anchor', default=True)
```

### 8.2 Wiring in `setup_training_config`

```python
def setup_training_config(preset='edm2-img512-s', **opts):
    ...
    # If teacher is provided, wire CD mode.
    if opts.get('teacher'):
        c.teacher_pkl = opts['teacher']
        c.cd_kwargs = dict(
            S=opts.get('S', 8),
            T_start=opts.get('T_start', 256),
            T_end=opts.get('T_end', 1024),
            T_anneal_kimg=opts.get('T_anneal_kimg', 750),
            loss_type=opts.get('cd_loss', 'pseudo_huber'),
            weight_mode=opts.get('cd_weight_mode', 'sqrt_karras'),
            sampling_mode=opts.get('sampling_mode', 'vp'),
            sync_dropout=opts.get('sync_dropout', True),
            terminal_anchor=opts.get('terminal_anchor', True),
        )
    ...
```

### 8.3 Config presets for CD

Consider adding CD-specific presets:
```python
'edm2-img512-s-cd8':  dnnlib.EasyDict(
    duration=200<<20, batch=2048, channels=192,
    lr=0.0020, decay=0, dropout=0.00,
    P_mean=-0.4, P_std=1.0,
),
```

These would use a lower LR (typical for distillation), shorter duration, and
no dropout (CD uses sync_dropout, not random dropout).

---

## 9. Testing & Validation Checklist

### Smoke tests (single-GPU, tiny dataset)

- [ ] Base EDM2 training still works (no regression).
- [ ] CD mode launches: teacher loads, student seeds, grids build.
- [ ] CD loss computes without NaN/Inf for 10 steps.
- [ ] Snapshot saves and loads (teacher excluded from pickle).
- [ ] Resume from checkpoint works (teacher re-loaded from original path).

### Functional tests

- [ ] `heun_hop_edm` produces same output as EDM version on same inputs.
- [ ] `inv_ddim_edm` produces same output as EDM version on same inputs.
- [ ] `make_karras_sigmas` with identity round_fn matches EDM output.
- [ ] Student grid + teacher grid have correct structure (descending, terminal 0).
- [ ] Edge sampling distribution matches expectations (terminal anchor ~1/T).
- [ ] Sync dropout: student and nograd target produce identical outputs when
  fed same input and RNG state (verify by comparing outputs with dropout=0.1).

### Multi-GPU tests

- [ ] DDP training runs without NCCL timeouts for 100 steps on 2+ GPUs.
- [ ] Gradient allreduce produces finite values.
- [ ] Snapshot + checkpoint writing doesn't cause rank desync.

### Quality tests

- [ ] Train CD on CIFAR-10 or ImageNet-64 for ~50k images; loss decreases.
- [ ] Generate images with S steps; visual quality is reasonable.
- [ ] FID improves over training (compare to random init).

---

## 10. Open Design Decisions

### OD-1: Default loss type and weight mode

EDM CD defaults to `huber` + `edm` weighting. MSCD paper uses `pseudo_huber` +
`sqrt_karras`. Which should be the default for EDM2?

**Recommendation:** `pseudo_huber` + `sqrt_karras` (matches MSCD paper).

### OD-2: EMA configuration for CD

EDM CD uses exponential EMA (`halflife=500 kimg`). EDM2 uses PowerFunctionEMA.
Should we:
- (a) Keep PowerFunctionEMA for CD snapshots, or
- (b) Add an option to use standard exponential EMA for CD?

**Recommendation:** (a) Keep PowerFunctionEMA. It's strictly more flexible and
gives post-hoc EMA reconstruction for free.

### OD-3: Encoder consistency between teacher and student

What if the teacher was trained with a different encoder (e.g., VAE) than the
student dataset? Should we:
- (a) Assert same encoder type, or
- (b) Allow mixed (and document that it's the user's responsibility)?

**Recommendation:** (a) Assert and warn. Latent space must be compatible.

### OD-4: Validation during CD training

Should we port the full FID validation system from EDM?

**Recommendation:** Defer to follow-up. External validation via
`generate_images.py` + `calculate_metrics.py` is sufficient initially.

### OD-5: `sigma_min` and `sigma_max` defaults

EDM uses `sigma_min=0.002, sigma_max=80` (matching the sampler). EDM2's
`Precond` has no explicit bounds. The CD grids need explicit values.

**Recommendation:** Use the same defaults `sigma_min=0.002, sigma_max=80` as
EDM. These match the `generate_images.py` sampler defaults.

### OD-6: `dropout` during CD training

EDM CD disables augmentation and uses `sync_dropout` for the nograd target.
EDM2 models may have dropout=0.10 (e.g., M/L/XL presets). Should CD training:
- (a) Use the teacher's dropout setting, or
- (b) Allow independent dropout setting for student?

**Recommendation:** (b) Allow override. Default to 0 for CD (simpler), with
sync_dropout available for non-zero dropout.

### OD-7: Learning rate for CD

EDM CD uses a fixed LR (2e-6) with linear rampup. EDM2 uses 1/sqrt decay.
For distillation, a constant LR after rampup may be better since training is
shorter.

**Recommendation:** Add a `--cd_lr` override. Default to EDM2's schedule but
allow constant LR via `--decay=0`.

---

## Appendix: File-Level Summary

| Action | File | Description |
|--------|------|-------------|
| **CREATE** | `training/consistency_ops.py` | Karras grids, Heun hop, invDDIM, DDIM, edge sampling (ported from EDM, remove `augment_labels` and `round_sigma` calls) |
| **CREATE** | `training/loss_cd.py` | `EDMConsistencyDistillLoss` class (ported from EDM, remove `augment_pipe`/`augment_labels`, add `__getstate__` for checkpoint) |
| **MODIFY** | `training/networks_edm2.py` | Add `round_sigma()`, `sigma_min`, `sigma_max` stubs to `Precond` |
| **MODIFY** | `training/training_loop.py` | Add CD-conditional logic: teacher loading, CD loss construction, teacher annealing, DDP tuning, snapshot exclusion |
| **MODIFY** | `train_edm2.py` | Add CD CLI options, wire `cd_kwargs` and `teacher_pkl` into training config |
| **MODIFY** | `generate_images.py` | Add persistence import hooks for `training.loss_cd` and `training.consistency_ops` |
| **MODIFY** | `reconstruct_phema.py` | Add same persistence import hooks |
| **SKIP** | `validation.py` | Defer to follow-up (not core to MSCD training) |
| **SKIP** | `debug_cd*.py`, `test_debug*.py` | Debug/test scripts — port later if needed |
