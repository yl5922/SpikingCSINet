# -*- coding: utf-8 -*-
"""
Training / evaluation script for SpikingCSINet / SpikingCSINetPR on COST2100.

Modes
-----
  default          : direct training at a single ``--T``.
  --prog_T         : progressive-T training (only meaningful for the PR model).
                     One model is built at ``--prog_T_end``; each stage sets
                     ``model.active_steps`` to the current T and trains for the
                     allocated epochs. Newly activated per-step modules are
                     warm-started from the previous stage.
  --eval_only      : skip training, load the checkpoint at ``--init_ckpt`` and
                     report val / test NMSE (in dB).

Models
------
  * spikingcsinetpr : the proposed progressive-residual model.
  * spikingcsinet   : ablation baseline (no progressive residual).

Example training command (reproduces our reported numbers at CR=64, indoor):

  CUDA_VISIBLE_DEVICES=0 python train.py \\
      --model spikingcsinetpr --file_path ./data --envir indoor \\
      --cr 64 --T 6 --tau 2 --enc_channels 4 --scale_factor 50 \\
      --rx_hidden 8196 --skip_alpha 0.5 --skip_norm_align \\
      --lr 0.0005 --lr_schedule cosine --cosine_cycles 1 --epochs 200 \\
      --batch_size 64 --snn_dropout 0.2 \\
      --step_mse_lambda 0.5 --step_mse_mode adaptive \\
      --rate_reg_lambda 0 --aug_phase_bins 16 --seed 2026

Example eval-only command:

  python train.py --eval_only --init_ckpt ./checkpoints/indoor_cr64_T6_spikingcsinetpr.pt \\
      --model spikingcsinetpr --file_path ./data --envir indoor --cr 64 --T 6 \\
      --tau 2 --enc_channels 4 --scale_factor 50 --rx_hidden 8196 \\
      --skip_alpha 0.5 --skip_norm_align --batch_size 64 --snn_dropout 0.2
"""
import csv
import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from dataset import load_data, data_preprocess, restore_channel, aug_csi
from networks import SpikingCSINetPR, SpikingCSINet, reset_net


MODELS = {"spikingcsinetpr", "spikingcsinet"}

# Dataset preprocessing was originally keyed off the legacy class name
# "snnconvenccsinet"; both new model names map to the same preprocessing path.
_DATASET_KEY = "snnconvenccsinet"


# ======================== Reproducibility ========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================== Evaluation ========================

@torch.no_grad()
def eval_nmse_db(model, loader, device, scale_factor):
    model.eval()
    nmse_list = []
    for x, _ in loader:
        x = x.to(device).float()
        x_in = data_preprocess(x, _DATASET_KEY, scale_factor)
        yhat, _ = model(x_in)

        H    = x - 0.5
        Hhat = restore_channel(yhat, _DATASET_KEY, scale_factor)

        err = ((H - Hhat) ** 2).sum(dim=(1, 2, 3))
        sig = (H ** 2).sum(dim=(1, 2, 3)) + 1e-16
        nmse_list.append((err / sig).cpu().numpy())
        reset_net(model)

    return 10.0 * np.log10(float(np.concatenate(nmse_list).mean()) + 1e-16)


@torch.no_grad()
def _batch_nmse_linear(x, yhat):
    err = ((x - yhat) ** 2).sum(dim=(1, 2, 3))
    sig = (x ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-16)
    return (err / sig).mean().item()


@torch.no_grad()
def compute_avg_firing_rate(model, loader, device, scale_factor):
    """Average per-neuron firing rate of every spiking layer over ``loader``.

    Resets the model's running spike-rate buffers, runs one eval pass over
    the loader (each LIF call inside ``forward`` accumulates into the
    buffers via ``_acc_spike_vec``), then averages each layer's per-neuron
    rate vector down to a single scalar.

    Returns a dict::

        {"lif_tx":   <mean spike rate of the transmitter LIF>,
         "rx_lif1":  <mean spike rate of the decoder LIF>,
         "overall":  <neuron-count-weighted mean across all layers>}
    """
    model.eval()
    if hasattr(model, "reset_running_spike_rate"):
        model.reset_running_spike_rate()

    for x, _ in loader:
        x = x.to(device).float()
        x_in = data_preprocess(x, _DATASET_KEY, scale_factor)
        model(x_in)
        reset_net(model)

    rate_dict = model.get_running_rate_dict()
    out = {}
    total_sum, total_cnt = 0.0, 0
    for name, vec in rate_dict.items():
        scalar = float(vec.float().mean().item())
        out[name] = scalar
        total_sum += scalar * vec.numel()
        total_cnt += vec.numel()
    out["overall"] = (total_sum / total_cnt) if total_cnt > 0 else float("nan")
    return out


# ======================== LR Scheduler ========================

def _make_lr_scheduler(opt, total_batches, args, tag=""):
    """Build the LR scheduler. Returns (scheduler, description) or (None, ...)."""
    sched = args.lr_schedule
    total_batches = max(1, int(total_batches))
    if sched == "cosine":
        cycles = max(1, int(args.cosine_cycles))
        if cycles == 1:
            s = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_batches, eta_min=0)
        else:
            period = max(1, total_batches // cycles)
            s = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=period, T_mult=1, eta_min=0)
        return s, (f"{tag}CosineAnnealingLR: T_max={total_batches} batches, "
                   f"cycles={cycles}")
    if sched == "linear":
        s = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=float(args.linear_end_factor),
            total_iters=total_batches)
        return s, (f"{tag}LinearLR: total_iters={total_batches} batches, "
                   f"end_factor={args.linear_end_factor}")
    return None, f"{tag}LR schedule: none (constant lr={args.lr})"


# ======================== Training Epoch ========================

def train_epoch(model, loader, opt, loss_fn, args, scheduler=None):
    model.train()
    total_loss, nmse_sum, nmse_batches = 0.0, 0.0, 0

    # Reset epoch-level running stats so adaptive step-MSE weights and the
    # per-step residual scaling are computed from this epoch only.
    for attr in ("reset_running_residual_norm", "reset_running_spike_rate",
                 "reset_running_step_mse"):
        if hasattr(model, attr):
            getattr(model, attr)()

    for x, _ in loader:
        x = x.to(args.device).float()
        x = data_preprocess(x, _DATASET_KEY, args.scale_factor)

        if args.aug:
            x = aug_csi(x, args.aug_phase, args.aug_noise_std, args.aug_phase_bins)

        yhat, _ = model(x)
        loss = loss_fn(yhat, x)

        # Auxiliary per-step MSE loss
        if getattr(model, "step_mse_vec", None) is not None and args.step_mse_lambda > 0:
            if args.step_mse_mode == "adaptive" and \
                    getattr(model, "step_mse_weights", None) is not None:
                w = model.step_mse_weights.to(model.step_mse_vec.device)
                loss = loss + (model.step_mse_vec * w).mean() * args.step_mse_lambda
            else:
                loss = loss + model.step_mse_vec.mean() * args.step_mse_lambda

        # Spike-rate regularisation (firing-rate penalty)
        if abs(args.rate_reg_lambda) > 1e-12:
            last_rate = getattr(model, "last_rate", None)
            if isinstance(last_rate, dict) and last_rate:
                total_sum, total_cnt = None, None
                for r in last_rate.values():
                    if r is None:
                        continue
                    s = r.sum()
                    c = torch.tensor(float(r.numel()), device=r.device)
                    total_sum = s if total_sum is None else (total_sum + s)
                    total_cnt = c if total_cnt is None else (total_cnt + c)
                if total_sum is not None:
                    loss = loss + args.rate_reg_lambda * total_sum / (total_cnt + 1e-12)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            nmse_sum += _batch_nmse_linear(x, yhat)
            nmse_batches += 1
        reset_net(model)

    avg_loss      = total_loss / max(1, len(loader))
    train_nmse_db = 10.0 * np.log10(nmse_sum / max(1, nmse_batches) + 1e-16)

    # Update next-epoch scale factors and adaptive step-MSE weights
    if hasattr(model, "update_scale_factors_from_running_norms"):
        model.update_scale_factors_from_running_norms()
    if hasattr(model, "update_step_mse_weights"):
        model.update_step_mse_weights()

    return avg_loss, train_nmse_db


# ======================== Model Builder ========================

def _build_model(args, H, W, M):
    if args.model == "spikingcsinetpr":
        return SpikingCSINetPR(
            H=H, W=W, M=M, T=args.T, tau=args.tau, vth=args.vth,
            enc_channels=args.enc_channels,
            skip_alpha=args.skip_alpha,
            skip_norm_align=args.skip_norm_align,
            rx_hidden=args.rx_hidden,
            use_scale=args.use_scale,
            neuron_type=args.neuron_type,
            dropout=args.snn_dropout,
        )
    if args.model == "spikingcsinet":
        return SpikingCSINet(
            H=H, W=W, M=M, T=args.T, tau=args.tau, vth=args.vth,
            enc_channels=args.enc_channels,
            skip_alpha=args.skip_alpha,
            skip_norm_align=args.skip_norm_align,
            rx_hidden=args.rx_hidden,
            neuron_type=args.neuron_type,
            dropout=args.snn_dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


# ======================== Direct Single-T Training ========================

def train_single_ratio(train_loader, val_loader, test_loader, args, H, W):
    cr = args.cr
    N  = H * W * 2
    M  = max(1, int(round(N / cr)))
    print(f"\n{'='*60}\n[Train] CR={cr} (M={M}, T={args.T}, model={args.model})\n{'='*60}")

    model = _build_model(args, H, W, M).to(args.device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=args.device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f"[Train] Loaded init: {args.init_ckpt}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {n_params:,} total, {n_train:,} trainable")
    print(f"[Train] skip_alpha={args.skip_alpha}, T={args.T}, tau={args.tau}, "
          f"vth={args.vth}")
    print(f"[Train] enc_channels={args.enc_channels} (per-step conv)")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    total_batches = args.epochs * len(train_loader)
    scheduler, desc = _make_lr_scheduler(opt, total_batches, args, tag="[Train] ")
    print(desc)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_name = f"{args.envir}_cr{cr}_T{args.T}_{args.model}.pt"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

    # Last-phase evaluation window (top-K test-mean reporting)
    n_last     = max(10, int(np.ceil(args.epochs * 0.10)))
    last_start = args.epochs - n_last + 1
    print(f"[Train] Last-phase epochs: {last_start}..{args.epochs} (n_last={n_last})")

    best_val, best_ep, best_state = float("inf"), -1, None
    last_records = []

    for ep in range(1, args.epochs + 1):
        loss, train_nmse_db = train_epoch(
            model, train_loader, opt, loss_fn, args, scheduler=scheduler)

        rate_str = ""
        if isinstance(getattr(model, "last_rate", None), dict):
            parts = [f"{layer}:{float(r.mean()):.4f}"
                     for layer, r in model.last_rate.items() if r is not None]
            if parts:
                rate_str = " | rate " + " ".join(parts)

        if ep < last_start:
            if not (ep % args.eval_every == 0 or ep == 1):
                continue
            reset_net(model)
            val_nmse = eval_nmse_db(model, val_loader, args.device, args.scale_factor)
            print(f"[CR={cr}] ep {ep:4d} | loss {loss:.6f} "
                  f"| train {train_nmse_db:.2f} dB | val {val_nmse:.2f} dB{rate_str}")
            if val_nmse < best_val:
                best_val   = float(val_nmse)
                best_ep    = ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        else:
            reset_net(model)
            val_nmse = eval_nmse_db(model, val_loader, args.device, args.scale_factor)
            reset_net(model)
            test_nmse = eval_nmse_db(model, test_loader, args.device, args.scale_factor)
            print(f"[CR={cr}] ep {ep:4d} | loss {loss:.6f} "
                  f"| train {train_nmse_db:.2f} dB | val {val_nmse:.2f} dB "
                  f"| test {test_nmse:.2f} dB{rate_str}")
            last_records.append({"ep": ep, "val": float(val_nmse),
                                 "test": float(test_nmse)})
            if val_nmse < best_val:
                best_val   = float(val_nmse)
                best_ep    = ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No validation was run, cannot select best-val model.")

    top10_stats = None
    if last_records:
        topk = sorted(last_records, key=lambda d: d["val"])[:10]
        test_vals = np.array([d["test"] for d in topk], dtype=np.float32)
        top10_stats = {
            "cr": cr, "M": M, "T": args.T, "model": args.model,
            "last_phase_start_epoch": last_start, "n_last": n_last,
            "topk": topk,
            "test_mean": float(test_vals.mean()),
            "test_std":  float(test_vals.std(ddof=0)),
        }
        print(f"[CR={cr}] Last {n_last} epochs: "
              f"Top-{len(topk)} by val -> test mean {top10_stats['test_mean']:.2f} dB, "
              f"std {top10_stats['test_std']:.2f} dB")

    save_dict = {
        "best_val_nmse_db":    float(best_val),
        "best_val_epoch":      int(best_ep),
        "cr": int(cr), "M": int(M), "T": int(args.T),
        "H": int(H), "W": int(W),
        "model_name":          args.model,
        "top10val_test_stats": top10_stats,
    }

    # ---- Average firing rate of the best-val model on val + test ----
    # Load best-val weights into the model first; firing rate of an
    # under-trained snapshot would be misleading.
    model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
    fr_val  = compute_avg_firing_rate(model, val_loader,  args.device, args.scale_factor)
    fr_test = compute_avg_firing_rate(model, test_loader, args.device, args.scale_factor)
    save_dict["avg_firing_rate"] = {"val": fr_val, "test": fr_test}
    print(f"[CR={cr}] Avg firing rate (best-val model):")
    for split, fr in (("val", fr_val), ("test", fr_test)):
        parts = [f"{k}={v:.4f}" for k, v in fr.items()]
        print(f"           {split:5s}: " + " ".join(parts))

    if args.save_state_dict:
        save_dict["state_dict"] = best_state
        tag = "with state_dict"
    else:
        tag = "metadata only (state_dict omitted)"

    torch.save(save_dict, ckpt_path)
    print(f"[CR={cr}] Saved {tag}: {ckpt_path} "
          f"(best val {best_val:.2f} dB @ ep {best_ep})")


# ======================== Progressive-T Training ========================

def train_progressive(train_loader, val_loader, test_loader, args, H, W):
    """Progressive-T training: build one model at ``T = prog_T_end`` and train
    in stages with ``model.active_steps`` set to the current T. Newly activated
    per-step modules are warm-started from the previous stage. After all
    stages finish, the model is evaluated at every T in the schedule and a
    table compares the progressive-T model against any directly-trained
    checkpoints sitting in ``--ckpt_dir`` (named e.g. ``indoor_cr8_T4_<model>.pt``).
    """
    cr = args.cr
    N  = H * W * 2
    M  = max(1, int(round(N / cr)))

    T_end = args.prog_T_end
    T_seq = list(range(args.prog_T_start, T_end + 1, args.prog_T_step))
    if T_seq[-1] != T_end:
        T_seq.append(T_end)

    print(f"\n{'='*60}")
    print(f"[ProgT] CR={cr} M={M}  T sequence: {T_seq}")
    print(f"[ProgT] Init epochs={args.prog_epochs_init}, "
          f"step epochs={args.prog_epochs_step}")
    print(f"{'='*60}")

    # Build one model at T_end; active_steps is set per stage
    _orig_T = args.T
    args.T  = T_end
    model   = _build_model(args, H, W, M).to(args.device)
    args.T  = _orig_T

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ProgT] Model built at T={T_end}: {n_params:,} params")

    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    stage_results    = {}      # T_cur -> {"best_val", "best_ep", "best_test"}
    best_state_final = None     # best-val state across the final stage
    prev_T_cur       = None

    for stage_idx, T_cur in enumerate(T_seq):
        epochs = args.prog_epochs_init if stage_idx == 0 else args.prog_epochs_step

        # Warm-start newly activated per-step modules from the previous stage
        if prev_T_cur is not None and T_cur > prev_T_cur:
            if hasattr(model, "warm_start_per_step_modules"):
                model.warm_start_per_step_modules(prev_T_cur, T_cur)
                print(f"[ProgT] Warm-started per-step modules "
                      f"[{prev_T_cur}, {T_cur}) from step {prev_T_cur - 1}")
            # Rebuild the optimiser so Adam moments are consistent — momentum /
            # variance from the old stage no longer match the new param values
            # for the reused steps.
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            print(f"[ProgT] Optimiser rebuilt (Adam state reset) at stage T={T_cur}")

        model.active_steps = T_cur
        prev_T_cur = T_cur

        # The length of step_mse_vec changes with active_steps, so any weights
        # left over from the previous stage would have the wrong shape.
        if hasattr(model, "step_mse_weights"):
            model.step_mse_weights = None
        for attr in ("reset_running_step_mse", "reset_running_residual_norm",
                     "reset_running_spike_rate"):
            if hasattr(model, attr):
                getattr(model, attr)()

        print(f"\n[ProgT] ── Stage T={T_cur} | epochs={epochs} ──")

        total_batches  = epochs * len(train_loader)
        scheduler, desc = _make_lr_scheduler(
            opt, total_batches, args, tag=f"[ProgT T={T_cur}] ")
        print(desc)

        n_last     = max(5, int(np.ceil(epochs * 0.10)))
        last_start = epochs - n_last + 1
        best_val, best_ep, best_state = float("inf"), -1, None
        last_records = []

        # train_epoch reads args.T only for logging; harmless either way
        args.T = T_cur
        for ep in range(1, epochs + 1):
            loss, train_nmse_db = train_epoch(
                model, train_loader, opt, loss_fn, args, scheduler=scheduler)

            rate_str = ""
            if isinstance(getattr(model, "last_rate", None), dict):
                parts = [f"{lyr}:{float(r.mean()):.4f}"
                         for lyr, r in model.last_rate.items() if r is not None]
                if parts:
                    rate_str = " | rate " + " ".join(parts)

            if ep < last_start:
                if not (ep % args.eval_every == 0 or ep == 1):
                    continue
                reset_net(model)
                val_nmse = eval_nmse_db(model, val_loader, args.device, args.scale_factor)
                print(f"[ProgT T={T_cur}] ep {ep:4d}/{epochs} | loss {loss:.6f} "
                      f"| train {train_nmse_db:.2f} dB | val {val_nmse:.2f} dB{rate_str}")
                if val_nmse < best_val:
                    best_val  = float(val_nmse)
                    best_ep   = ep
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                reset_net(model)
                val_nmse  = eval_nmse_db(model, val_loader, args.device, args.scale_factor)
                reset_net(model)
                test_nmse = eval_nmse_db(model, test_loader, args.device, args.scale_factor)
                print(f"[ProgT T={T_cur}] ep {ep:4d}/{epochs} | loss {loss:.6f} "
                      f"| train {train_nmse_db:.2f} dB | val {val_nmse:.2f} dB "
                      f"| test {test_nmse:.2f} dB{rate_str}")
                last_records.append({"ep": ep, "val": float(val_nmse),
                                     "test": float(test_nmse)})
                if val_nmse < best_val:
                    best_val  = float(val_nmse)
                    best_ep   = ep
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
        args.T = _orig_T

        if best_state is None:
            raise RuntimeError(f"[ProgT T={T_cur}] No validation was run.")

        best_test = None
        if last_records:
            topk      = sorted(last_records, key=lambda d: d["val"])[:10]
            test_vals = np.array([d["test"] for d in topk], dtype=np.float32)
            best_test = float(test_vals.mean())
            print(f"[ProgT T={T_cur}] top-{len(topk)} by val → "
                  f"test mean {best_test:.2f} dB ± {test_vals.std(ddof=0):.2f} dB")

        stage_results[T_cur] = {"best_val": best_val, "best_ep": best_ep,
                                 "best_test": best_test}

        # Carry best-val weights into the next stage
        model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
        if T_cur == T_end:
            best_state_final = best_state

    # ---- Save the final progressive-T checkpoint ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    model.active_steps = T_end
    ckpt_name = f"{args.envir}_cr{cr}_T{T_end}_{args.model}_progt.pt"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    save_dict = {
        "prog_T_seq":   T_seq,
        "prog_T_end":   T_end,
        "stage_results": stage_results,
        "cr": int(cr), "M": int(M), "T": int(T_end),
        "H": int(H), "W": int(W),
        "model_name":   args.model,
    }

    # Average firing rate of the final-stage best-val model
    if best_state_final is not None:
        model.load_state_dict({k: v.to(args.device) for k, v in best_state_final.items()})
    fr_val  = compute_avg_firing_rate(model, val_loader,  args.device, args.scale_factor)
    fr_test = compute_avg_firing_rate(model, test_loader, args.device, args.scale_factor)
    save_dict["avg_firing_rate"] = {"val": fr_val, "test": fr_test}
    print(f"[ProgT] Avg firing rate (final stage best-val model, T={T_end}):")
    for split, fr in (("val", fr_val), ("test", fr_test)):
        parts = [f"{k}={v:.4f}" for k, v in fr.items()]
        print(f"           {split:5s}: " + " ".join(parts))

    if args.save_state_dict and best_state_final is not None:
        save_dict["state_dict"] = best_state_final
    torch.save(save_dict, ckpt_path)
    print(f"\n[ProgT] Final checkpoint saved: {ckpt_path}")

    # ---- Comparison table: prog-T model at every T vs direct ckpts ----
    print(f"\n{'='*72}")
    print(f"[ProgT] Comparison: progressive-T (one model)  vs  direct (CR={cr})")
    print(f"{'='*72}")
    print(f"{'T':>4}  {'ProgT val':>12}  {'ProgT test':>14}  "
          f"{'Direct val':>12}  {'Direct test':>12}")
    print(f"{'-'*4}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*12}")

    for T_cur in T_seq:
        model.active_steps = T_cur
        reset_net(model)
        p_val_eval  = eval_nmse_db(model, val_loader,  args.device, args.scale_factor)
        reset_net(model)
        p_test_eval = eval_nmse_db(model, test_loader, args.device, args.scale_factor)

        d_val = d_test = "(no ckpt)"
        direct_path = os.path.join(
            args.ckpt_dir, f"{args.envir}_cr{cr}_T{T_cur}_{args.model}.pt")
        if os.path.isfile(direct_path):
            try:
                d = torch.load(direct_path, map_location="cpu", weights_only=False)
                d_val = f"{d.get('best_val_nmse_db', float('nan')):.2f} dB"
                s = d.get("top10val_test_stats")
                d_test = f"{s['test_mean']:.2f} dB" if s else "—"
            except Exception as e:
                d_val = d_test = f"err({e})"

        print(f"{T_cur:>4}  {p_val_eval:>9.2f} dB  {p_test_eval:>11.2f} dB  "
              f"{d_val:>12}  {d_test:>12}")

    model.active_steps = T_end
    print(f"{'='*72}")


# ======================== Eval-Only ========================

def eval_only(train_loader, val_loader, test_loader, args, H, W):
    """Load a saved checkpoint, compute train/val/test NMSE and average firing
    rate, and write the results to a CSV alongside the checkpoint.
    """
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise FileNotFoundError(
            f"--eval_only requires --init_ckpt to point at a checkpoint file; "
            f"got: {args.init_ckpt!r}")

    cr = args.cr
    N  = H * W * 2
    M  = max(1, int(round(N / cr)))
    print(f"\n{'='*60}\n[Eval] CR={cr} (M={M}, T={args.T}, model={args.model})")
    print(f"[Eval] Loading checkpoint: {args.init_ckpt}\n{'='*60}")

    ckpt = torch.load(args.init_ckpt, map_location=args.device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if isinstance(ckpt, dict):
        for key in ("model_name", "cr", "M", "T", "H", "W"):
            if key in ckpt:
                print(f"[Eval]   ckpt {key} = {ckpt[key]}")

    model = _build_model(args, H, W, M).to(args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Eval] WARNING — missing keys when loading: {missing}")
    if unexpected:
        print(f"[Eval] WARNING — unexpected keys in checkpoint: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Eval] Model params: {n_params:,}")

    # ---- NMSE on each split ----
    reset_net(model)
    val_nmse  = eval_nmse_db(model, val_loader,  args.device, args.scale_factor)
    reset_net(model)
    test_nmse = eval_nmse_db(model, test_loader, args.device, args.scale_factor)
    reset_net(model)
    train_nmse = eval_nmse_db(model, train_loader, args.device, args.scale_factor)

    # ---- Average firing rate on each split ----
    fr_train = compute_avg_firing_rate(model, train_loader, args.device, args.scale_factor)
    fr_val   = compute_avg_firing_rate(model, val_loader,   args.device, args.scale_factor)
    fr_test  = compute_avg_firing_rate(model, test_loader,  args.device, args.scale_factor)

    # ---- Console summary ----
    layer_names = sorted(fr_test.keys())  # consistent column order
    print(f"\n{'='*60}")
    print(f"[Eval] CR={cr} | T={args.T} | model={args.model}")
    print(f"[Eval]   train NMSE: {train_nmse:.2f} dB   |   firing rate: " +
          " ".join(f"{n}={fr_train[n]:.4f}" for n in layer_names))
    print(f"[Eval]   val   NMSE: {val_nmse:.2f} dB   |   firing rate: " +
          " ".join(f"{n}={fr_val[n]:.4f}"   for n in layer_names))
    print(f"[Eval]   test  NMSE: {test_nmse:.2f} dB   |   firing rate: " +
          " ".join(f"{n}={fr_test[n]:.4f}"  for n in layer_names))
    print(f"{'='*60}")

    # ---- Write CSV alongside the checkpoint ----
    # Filename keeps the same identifying fields as the checkpoint:
    # <envir>_cr<CR>_T<T>_<model_name>.csv (matching the .pt filename root).
    out_dir = (args.eval_csv_dir if args.eval_csv_dir
               else os.path.dirname(os.path.abspath(args.init_ckpt)))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        out_dir, f"eval_{args.envir}_cr{cr}_T{args.T}_{args.model}.csv")
    nmse_by_split = {"train": train_nmse, "val": val_nmse, "test": test_nmse}
    fr_by_split   = {"train": fr_train,   "val": fr_val,   "test": fr_test}
    header = ["split", "nmse_db"] + [f"firing_rate_{n}" for n in layer_names]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for split in ("train", "val", "test"):
            row = [split, f"{nmse_by_split[split]:.6f}"] + \
                  [f"{fr_by_split[split][n]:.6f}" for n in layer_names]
            w.writerow(row)
    print(f"[Eval] CSV written: {csv_path}")

    return {
        "nmse_db":         nmse_by_split,
        "avg_firing_rate": fr_by_split,
        "csv_path":        csv_path,
    }


# ======================== Main ========================

def main(args):
    args.device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data(
        file_path=args.file_path, batch_size=args.batch_size,
        num_workers=args.num_workers, dataset_name="cost2100",
        envir=args.envir, gpu_preload=args.gpu_preload, device=args.device)

    xb, _ = next(iter(train_loader))
    _, C, H, W = xb.shape
    assert C == 2, f"Expected 2 channels, got {C}"

    os.makedirs(args.ckpt_dir, exist_ok=True)
    assert args.cr >= 1, "--cr must be a positive integer"

    if args.eval_only:
        eval_only(train_loader, val_loader, test_loader, args, H, W)
    elif args.prog_T:
        train_progressive(train_loader, val_loader, test_loader, args, H, W)
    else:
        train_single_ratio(train_loader, val_loader, test_loader, args, H, W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training / evaluation for SpikingCSINet / SpikingCSINetPR.")

    # ---- Run mode ----
    parser.add_argument("--eval_only", action="store_true", default=False,
                        help="Skip training; load --init_ckpt and report "
                             "train/val/test NMSE.")
    parser.add_argument("--prog_T", action="store_true", default=False,
                        help="Progressive-T training: start at T=prog_T_start "
                             "and grow to T=prog_T_end. Overrides --epochs.")

    # ---- Data ----
    parser.add_argument("--file_path",     default="./data")
    parser.add_argument("--num_workers",   type=int,   default=1)
    parser.add_argument("--batch_size",    type=int,   default=200)
    parser.add_argument("--scale_factor",  type=float, default=50,
                        help="SNN preprocessing scale: x_in = (x - 0.5) * scale_factor")
    parser.add_argument("--gpu_preload",   action="store_true", default=False)
    parser.add_argument("--envir",         default="indoor",
                        choices=["indoor", "outdoor"])

    # ---- Model selection ----
    parser.add_argument("--model",
                        choices=sorted(MODELS),
                        default="spikingcsinetpr")

    # ---- SNN architecture ----
    parser.add_argument("--T", type=int, default=8, help="SNN time steps")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="LIF leak time constant (ignored when --neuron_type if).")
    parser.add_argument("--vth", type=float, default=1.0)
    parser.add_argument("--neuron_type", choices=["lif", "if"], default="lif")
    parser.add_argument("--enc_channels", type=int, default=16,
                        help="Hidden channels in the per-step conv encoder.")
    parser.add_argument("--rx_hidden", type=int, default=4096,
                        help="Width of the decoder spiking layer rx_fc1 / rx_lif1.")
    parser.add_argument("--skip_alpha", type=float, default=1.0,
                        help="Weight on the rx_skip output: y = y_main + alpha * y_skip.")
    parser.add_argument("--skip_norm_align", action="store_true", default=False,
                        help="Rescale rx_skip output to match the L2 norm of y_main "
                             "before the weighted sum.")
    parser.add_argument("--snn_dropout", type=float, default=0.0,
                        help="Dropout probability on the rx_lif1 spike output. "
                             "0.0 disables dropout entirely.")
    parser.add_argument("--no_scale", dest="use_scale", action="store_false",
                        help="Disable residual rescaling in SpikingCSINetPR (only "
                             "affects the PR model; the baseline never rescales).")
    parser.set_defaults(use_scale=True)

    # ---- Training ----
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--eval_every", type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lr_schedule", choices=["none", "cosine", "linear"],
                        default="cosine")
    parser.add_argument("--cosine_cycles", type=int, default=1,
                        help="Number of cosine cycles (1 = single decay, "
                             ">1 = warm restarts).")
    parser.add_argument("--linear_end_factor", type=float, default=0.0,
                        help="Final-LR multiplier for --lr_schedule=linear.")
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--ckpt_dir",   default="./checkpoints")
    parser.add_argument("--init_ckpt",  default="",
                        help="Path to a checkpoint to load. Required for "
                             "--eval_only; optional for training (warm start).")
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--save_state_dict", action="store_true", default=False,
                        help="Save state_dict in checkpoint (default OFF to save space)")
    parser.add_argument("--eval_csv_dir", default="",
                        help="Directory to write the eval CSV in --eval_only mode. "
                             "If empty, writes alongside --init_ckpt.")

    # ---- Augmentation ----
    parser.add_argument("--aug",            action="store_true", default=False)
    parser.add_argument("--no_aug_phase",   action="store_true", default=False)
    parser.add_argument("--aug_noise_std",  type=float, default=0.0)
    parser.add_argument("--aug_phase_bins", type=int,   default=16)

    # ---- Auxiliary losses ----
    parser.add_argument("--step_mse_lambda", type=float, default=0.2,
                        help="Weight on the per-step MSE auxiliary loss.")
    parser.add_argument("--step_mse_mode",   default="uniform",
                        choices=["uniform", "adaptive"],
                        help="adaptive uses inverse-MSE weights from the previous epoch.")
    parser.add_argument("--rate_reg_lambda", type=float, default=0.0,
                        help="Weight on the firing-rate regulariser (mean spike rate).")

    # ---- Compression ratio ----
    parser.add_argument("--cr", type=int, default=8,
                        help="Compression ratio: M = round(2*H*W / cr).")

    # ---- Progressive-T training ----
    parser.add_argument("--prog_T_start", type=int, default=2,
                        help="Starting T for --prog_T (default 2).")
    parser.add_argument("--prog_T_end",   type=int, default=10,
                        help="Final T for --prog_T (default 10).")
    parser.add_argument("--prog_T_step",  type=int, default=2,
                        help="T increment per stage (default 2 → 2,4,6,8,10).")
    parser.add_argument("--prog_epochs_init", type=int, default=1000,
                        help="Epochs for the first stage T=prog_T_start.")
    parser.add_argument("--prog_epochs_step", type=int, default=400,
                        help="Epochs for each subsequent stage.")

    args = parser.parse_args()
    args.aug_phase = not args.no_aug_phase
    print(args)
    set_seed(args.seed)
    main(args)
