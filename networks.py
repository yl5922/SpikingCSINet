# -*- coding: utf-8 -*-
"""
Spiking neural networks for CSI feedback.

Two models are provided:

  * SpikingCSINetPR : the proposed model with progressive-residual encoding.
                      At every time step the encoder sees ``x - y_est`` (the
                      reconstruction residual so far) and emits an additive
                      correction.

  * SpikingCSINet   : ablation baseline that feeds the *original* channel ``x``
                      to the encoder at every step and averages the per-step
                      reconstructions.

Both share the same network blocks:
  * encoder    : per-step 2-layer ANN conv (channel-light) + a single shared FC
                 ``tx_fc`` that maps the conv features to the M-dim latent.
  * spike tx   : a single LIF/IF neuron over the latent — this is the bit that
                 transmits actual binary spikes between encoder and decoder.
  * decoder    : a single FC ``rx_fc1`` + LIF, plus a single linear skip
                 ``rx_skip`` that bypasses the spiking layer to inject a
                 continuous-valued correction at the output. The final
                 reconstruction is ``rx_main + skip_alpha * rx_skip``.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


# ======================== Helpers ========================

def reset_net(net: nn.Module):
    """Reset the membrane potential of every spiking neuron in ``net``."""
    for m in net.modules():
        if hasattr(m, "reset") and callable(m.reset):
            m.reset()


def _make_lif(tau, vth, neuron_type="lif"):
    """Build a spiking neuron.

    neuron_type:
      * "lif" — leaky integrate-and-fire with time constant ``tau``.
      * "if"  — non-leaky integrate-and-fire (``tau`` ignored).
    """
    nt = str(neuron_type).lower()
    if nt == "if":
        return neuron.IFNode(v_threshold=vth,
                             surrogate_function=surrogate.ATan(),
                             detach_reset=True)
    return neuron.LIFNode(tau=tau, v_threshold=vth,
                          surrogate_function=surrogate.ATan(),
                          detach_reset=True)


# ======================== SpikingCSINetPR (progressive-residual) ========================

class SpikingCSINetPR(nn.Module):
    """SNN CSI feedback with per-step ANN conv encoder and progressive residual.

    At time step ``t``:
      1. Compute the residual  r_t = x - y_est_{t-1}  (r_0 = x).
      2. Optionally rescale  x_t = r_t * scale_factors[t]  to keep the conv
         input amplitude roughly constant across steps.
      3. Encode with per-step conv ``enc_convs[t]`` then shared linear ``tx_fc``.
      4. Drive a LIF/IF over the latent to produce binary spikes ``s_tx``.
      5. Decode through ``rx_fc1 + rx_lif1`` (main path) and the linear bypass
         ``rx_skip`` (continuous-valued correction). The reconstruction at this
         step is  y_main + skip_alpha * rx_skip(s_tx).
      6. Undo the rescale to obtain the residual prediction and accumulate it
         into ``y_est``.
    """

    def __init__(self, H, W, M, T=8, tau=2.0, vth=1.0,
                 enc_channels=16,
                 skip_alpha=1.0, skip_norm_align=False,
                 rx_hidden=4096,
                 scale_clamp=(0.25, 256.0), scale_momentum=0.2, scale_eps=1e-6,
                 use_scale=True, neuron_type="lif", dropout=0.0):
        super().__init__()
        assert str(neuron_type).lower() in ("lif", "if")
        assert 0.0 <= float(dropout) < 1.0, "dropout prob must be in [0, 1)"

        self.H, self.W, self.M = H, W, M
        self.time_step       = T
        self.skip_alpha      = float(skip_alpha)
        self.skip_norm_align = bool(skip_norm_align)
        self.enc_channels    = enc_channels
        self.scale_clamp     = scale_clamp
        self.scale_momentum  = scale_momentum
        self.scale_eps       = scale_eps
        self.use_scale       = bool(use_scale)
        self.neuron_type     = str(neuron_type).lower()
        # Number of time steps actually used in forward (≤ T). Buffers and
        # per-step modules are sized for T; we just iterate up to active_steps.
        # This is the hook for progressive-T training: stage k sets
        # ``active_steps`` to T_k while the underlying buffers stay at T_end.
        self.active_steps    = T
        self.dropout_p       = float(dropout)
        # Dropout applied to the rx_lif1 spike output (the rx_hidden-wide
        # tensor — the largest activation in the decoder). Identity when 0.
        self.rx_drop1 = (nn.Dropout(p=self.dropout_p)
                         if self.dropout_p > 0.0 else nn.Identity())

        in_dim = 2 * H * W

        # ---- Encoder: per-step ANN conv + shared FC ----
        self.enc_convs = nn.ModuleList()
        for _ in range(T):
            self.enc_convs.append(nn.Sequential(
                nn.Conv2d(2, enc_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(enc_channels),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(enc_channels, 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(2),
                nn.LeakyReLU(0.3, inplace=True),
            ))
        self.tx_fc  = nn.Linear(in_dim, M, bias=True)
        self.lif_tx = _make_lif(tau, vth, neuron_type=self.neuron_type)

        # ---- Decoder: main spiking path + linear skip ----
        self.rx_fc1  = nn.Linear(M, rx_hidden, bias=False)
        self.rx_lif1 = _make_lif(tau, vth, neuron_type=self.neuron_type)
        self.fc_out  = nn.Linear(rx_hidden, in_dim, bias=True)
        self.rx_skip = nn.Linear(M, in_dim, bias=True)

        # ---- Spike-rate running stats (for logging / rate regularisation) ----
        self._spk_layer_dims = {"lif_tx": M, "rx_lif1": rx_hidden}
        for name, dim in self._spk_layer_dims.items():
            self.register_buffer(f"run_spk_sum_{name}", torch.zeros(dim))
            self.register_buffer(f"run_spk_cnt_{name}", torch.zeros(1))
        self.last_rate = None

        # ---- Scale factors & step-MSE running stats ----
        self.register_buffer("scale_factors",    torch.ones(T))
        self.register_buffer("run_res_norm_sum", torch.zeros(T))
        self.register_buffer("run_res_norm_cnt", torch.zeros(T))
        self.register_buffer("run_step_mse_sum", torch.zeros(T))
        self.register_buffer("run_step_mse_cnt", torch.zeros(T))
        self.step_mse_weights = None
        self.step_mse_vec  = None
        self.step_mse_mean = None

    # ---- Spike-rate helpers ----
    @torch.no_grad()
    def reset_running_spike_rate(self):
        for name in self._spk_layer_dims:
            getattr(self, f"run_spk_sum_{name}").zero_()
            getattr(self, f"run_spk_cnt_{name}").zero_()

    @torch.no_grad()
    def _acc_spike_vec(self, name, spk):
        getattr(self, f"run_spk_sum_{name}").add_(spk.detach().float().sum(dim=0))
        getattr(self, f"run_spk_cnt_{name}").add_(
            torch.tensor([spk.shape[0]], device=spk.device))

    @torch.no_grad()
    def get_running_rate_vec(self, name):
        s = getattr(self, f"run_spk_sum_{name}")
        c = getattr(self, f"run_spk_cnt_{name}").clamp_min(1.0)
        return (s / c).clone()

    @torch.no_grad()
    def get_running_rate_dict(self):
        return {n: self.get_running_rate_vec(n) for n in self._spk_layer_dims}

    # ---- Residual-norm / step-MSE / scale-factor helpers ----
    @torch.no_grad()
    def reset_running_residual_norm(self):
        self.run_res_norm_sum.zero_()
        self.run_res_norm_cnt.zero_()

    @torch.no_grad()
    def reset_running_step_mse(self):
        self.run_step_mse_sum.zero_()
        self.run_step_mse_cnt.zero_()

    @torch.no_grad()
    def update_step_mse_weights(self):
        """Inverse-MSE weights (normalised to mean 1) over the active steps."""
        K = int(getattr(self, "active_steps", self.time_step))
        K = max(1, min(K, int(self.time_step)))
        if (self.run_step_mse_cnt[:K] <= 0).any():
            return
        cnt = self.run_step_mse_cnt[:K].clamp_min(1.0)
        avg = self.run_step_mse_sum[:K] / cnt
        inv = 1.0 / (avg + 1e-12)
        self.step_mse_weights = inv * (float(K) / (inv.sum() + 1e-12))

    @torch.no_grad()
    def update_scale_factors_from_running_norms(self):
        """Set scale_factors[t] = ||r_0|| / ||r_t|| (clamped, EMA-smoothed)."""
        K = int(getattr(self, "active_steps", self.time_step))
        K = max(1, min(K, int(self.time_step)))
        if (self.run_res_norm_cnt[:K] <= 0).any():
            return
        eps = self.scale_eps
        cnt = self.run_res_norm_cnt[:K].clamp_min(1.0)
        avg = self.run_res_norm_sum[:K] / cnt
        target = avg[0].clamp_min(eps)
        new_scale_active = torch.ones_like(avg)
        if K > 1:
            new_scale_active[1:] = target / (avg[1:] + eps)
        lo, hi = self.scale_clamp
        new_scale_active = new_scale_active.clamp(lo, hi)
        new_scale_active[0] = 1.0
        new_scale = self.scale_factors.clone()
        new_scale[:K] = new_scale_active
        m = float(self.scale_momentum)
        if m > 0:
            self.scale_factors.mul_(1 - m).add_(m * new_scale)
        else:
            self.scale_factors.copy_(new_scale)

    @torch.no_grad()
    def warm_start_per_step_modules(self, prev_T, new_T):
        """Copy weights from step ``prev_T - 1`` into newly activated steps.

        Used by progressive-T training when ``active_steps`` grows from
        ``prev_T`` to ``new_T``: the conv encoders at indices
        ``[prev_T, new_T)`` still hold their random initialisation while the
        rest of the model is already well trained. Seeding them from the last
        trained step puts them on a sensible starting manifold so the new
        stage continues to learn rather than recover from random output.

        Also seeds ``scale_factors[prev_T:new_T]`` from the last trained slot
        so the residual rescaling does not start from 1.0 on the new steps.
        """
        prev_T, new_T = int(prev_T), int(new_T)
        if not (0 < prev_T < new_T <= int(self.time_step)):
            return
        src_idx = prev_T - 1
        for t in range(prev_T, new_T):
            self.enc_convs[t].load_state_dict(self.enc_convs[src_idx].state_dict())
        if self.use_scale:
            self.scale_factors[prev_T:new_T] = self.scale_factors[src_idx]

    def forward(self, x):
        reset_net(self)
        B = x.size(0)
        y_est = torch.zeros_like(x)
        residual = x
        rx_feat_acc = None
        step_mse_list = []
        scales = self.scale_factors.to(device=x.device, dtype=x.dtype)
        # grad_sum retains gradients so a firing-rate regulariser can
        # back-propagate through the surrogate.
        grad_sum = {k: None for k in self._spk_layer_dims}

        for t in range(self.active_steps):
            residual_t = residual
            with torch.no_grad():
                norms = residual_t.detach().view(B, -1).norm(p=2, dim=1)
                self.run_res_norm_sum[t] += norms.sum()
                self.run_res_norm_cnt[t] += B

            x_scaled = residual_t * scales[t] if self.use_scale else residual_t

            # Encoder
            e = self.enc_convs[t](x_scaled)
            s = self.tx_fc(e.view(B, -1))

            # Spike transmission (LIF/IF)
            s_spk = self.lif_tx(s)
            self._acc_spike_vec("lif_tx", s_spk)
            spk_sum = s_spk.float().sum(dim=0)
            grad_sum["lif_tx"] = spk_sum if grad_sum["lif_tx"] is None \
                else (grad_sum["lif_tx"] + spk_sum)
            s_tx = s_spk

            # Decoder: linear skip in parallel with the spiking main path
            y_skip = self.rx_skip(s_tx)

            r1_spk = self.rx_lif1(self.rx_fc1(s_tx))
            self._acc_spike_vec("rx_lif1", r1_spk)
            r1_sum = r1_spk.float().sum(dim=0)
            grad_sum["rx_lif1"] = r1_sum if grad_sum["rx_lif1"] is None \
                else (grad_sum["rx_lif1"] + r1_sum)

            # Dropout on the rx_hidden-wide spike feature. Spike-rate stats
            # above are computed on the un-dropped spikes because firing rate
            # is a property of the neuron, not of the dropout mask.
            rx_feat_t = self.rx_drop1(r1_spk)
            rx_feat_acc = rx_feat_t if rx_feat_acc is None \
                else (rx_feat_acc + rx_feat_t)

            y_main = self.fc_out(rx_feat_t)

            if self.skip_norm_align:
                mn = y_main.detach().norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                sn = y_skip.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                y_skip = y_skip * (mn / sn)
            y_out = y_main + self.skip_alpha * y_skip
            y_step = y_out.view(B, 2, self.H, self.W)

            # Undo residual rescale to obtain the actual delta prediction
            delta_hat = y_step / scales[t] if self.use_scale else y_step
            mse_t = (delta_hat - residual_t).pow(2).mean(dim=(1, 2, 3)).mean()
            step_mse_list.append(mse_t)
            with torch.no_grad():
                self.run_step_mse_sum[t] += mse_t.detach()
                self.run_step_mse_cnt[t] += 1

            y_est = y_est + delta_hat
            residual = x - y_est

        self.last_rate = {name: grad_sum[name] / float(B * self.active_steps)
                          for name in self._spk_layer_dims}
        self.step_mse_vec  = torch.stack(step_mse_list, dim=0)
        self.step_mse_mean = float(self.step_mse_vec.mean().detach())
        return y_est, rx_feat_acc / float(self.active_steps)


# ======================== SpikingCSINet (no progressive residual) ========================

class SpikingCSINet(nn.Module):
    """Ablation baseline of :class:`SpikingCSINetPR`.

    Same architecture, but at every time step the encoder consumes the
    *original* channel ``x`` rather than the residual ``x - y_est``. The
    per-step decoder outputs are then averaged to produce the final
    reconstruction.

    The constructor signature, attributes and forward signature are kept
    aligned with :class:`SpikingCSINetPR` so the same training loop works for
    both models.
    """

    def __init__(self, H, W, M, T=8, tau=2.0, vth=1.0,
                 enc_channels=16,
                 skip_alpha=1.0, skip_norm_align=False,
                 rx_hidden=4096,
                 neuron_type="lif", dropout=0.0):
        super().__init__()
        assert str(neuron_type).lower() in ("lif", "if")
        assert 0.0 <= float(dropout) < 1.0, "dropout prob must be in [0, 1)"

        self.H, self.W, self.M = H, W, M
        self.time_step       = T
        self.skip_alpha      = float(skip_alpha)
        self.skip_norm_align = bool(skip_norm_align)
        self.enc_channels    = enc_channels
        self.neuron_type     = str(neuron_type).lower()
        self.active_steps    = T
        self.dropout_p       = float(dropout)
        self.rx_drop1 = (nn.Dropout(p=self.dropout_p)
                         if self.dropout_p > 0.0 else nn.Identity())

        in_dim = 2 * H * W

        # ---- Encoder ----
        self.enc_convs = nn.ModuleList()
        for _ in range(T):
            self.enc_convs.append(nn.Sequential(
                nn.Conv2d(2, enc_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(enc_channels),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(enc_channels, 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(2),
                nn.LeakyReLU(0.3, inplace=True),
            ))
        self.tx_fc  = nn.Linear(in_dim, M, bias=True)
        self.lif_tx = _make_lif(tau, vth, neuron_type=self.neuron_type)

        # ---- Decoder ----
        self.rx_fc1  = nn.Linear(M, rx_hidden, bias=False)
        self.rx_lif1 = _make_lif(tau, vth, neuron_type=self.neuron_type)
        self.fc_out  = nn.Linear(rx_hidden, in_dim, bias=True)
        self.rx_skip = nn.Linear(M, in_dim, bias=True)

        # ---- Spike-rate running stats ----
        self._spk_layer_dims = {"lif_tx": M, "rx_lif1": rx_hidden}
        for name, dim in self._spk_layer_dims.items():
            self.register_buffer(f"run_spk_sum_{name}", torch.zeros(dim))
            self.register_buffer(f"run_spk_cnt_{name}", torch.zeros(1))
        self.last_rate = None

        # ---- Step-MSE running stats (kept for loss / logging compatibility) ----
        self.register_buffer("run_step_mse_sum", torch.zeros(T))
        self.register_buffer("run_step_mse_cnt", torch.zeros(T))
        self.step_mse_weights = None
        self.step_mse_vec  = None
        self.step_mse_mean = None

    # ---- Spike-rate helpers (identical API to SpikingCSINetPR) ----
    @torch.no_grad()
    def reset_running_spike_rate(self):
        for name in self._spk_layer_dims:
            getattr(self, f"run_spk_sum_{name}").zero_()
            getattr(self, f"run_spk_cnt_{name}").zero_()

    @torch.no_grad()
    def _acc_spike_vec(self, name, spk):
        getattr(self, f"run_spk_sum_{name}").add_(spk.detach().float().sum(dim=0))
        getattr(self, f"run_spk_cnt_{name}").add_(
            torch.tensor([spk.shape[0]], device=spk.device))

    @torch.no_grad()
    def get_running_rate_vec(self, name):
        s = getattr(self, f"run_spk_sum_{name}")
        c = getattr(self, f"run_spk_cnt_{name}").clamp_min(1.0)
        return (s / c).clone()

    @torch.no_grad()
    def get_running_rate_dict(self):
        return {n: self.get_running_rate_vec(n) for n in self._spk_layer_dims}

    # ---- API parity with SpikingCSINetPR ----
    @torch.no_grad()
    def reset_running_residual_norm(self):
        pass  # no residual rescaling in baseline

    @torch.no_grad()
    def reset_running_step_mse(self):
        self.run_step_mse_sum.zero_()
        self.run_step_mse_cnt.zero_()

    @torch.no_grad()
    def update_step_mse_weights(self):
        K = int(getattr(self, "active_steps", self.time_step))
        K = max(1, min(K, int(self.time_step)))
        if (self.run_step_mse_cnt[:K] <= 0).any():
            return
        cnt = self.run_step_mse_cnt[:K].clamp_min(1.0)
        avg = self.run_step_mse_sum[:K] / cnt
        inv = 1.0 / (avg + 1e-12)
        self.step_mse_weights = inv * (float(K) / (inv.sum() + 1e-12))

    @torch.no_grad()
    def update_scale_factors_from_running_norms(self):
        pass  # baseline has no residual rescaling

    @torch.no_grad()
    def warm_start_per_step_modules(self, prev_T, new_T):
        """Same warm-start as the PR model, sans scale-factor handling."""
        prev_T, new_T = int(prev_T), int(new_T)
        if not (0 < prev_T < new_T <= int(self.time_step)):
            return
        src_idx = prev_T - 1
        for t in range(prev_T, new_T):
            self.enc_convs[t].load_state_dict(self.enc_convs[src_idx].state_dict())

    def forward(self, x):
        """Each time step encodes the original channel ``x``; per-step outputs
        are averaged. Returns ``(y_est, rx_feat_acc)`` matching the PR model.
        """
        reset_net(self)
        B = x.size(0)
        K = int(getattr(self, "active_steps", self.time_step))
        K = max(1, min(K, int(self.time_step)))

        y_steps = []
        rx_feat_acc = None
        step_mse_list = []
        grad_sum = {k: None for k in self._spk_layer_dims}

        for t in range(K):
            e = self.enc_convs[t](x)
            s = self.tx_fc(e.view(B, -1))

            s_spk = self.lif_tx(s)
            self._acc_spike_vec("lif_tx", s_spk)
            spk_sum = s_spk.float().sum(dim=0)
            grad_sum["lif_tx"] = spk_sum if grad_sum["lif_tx"] is None \
                else (grad_sum["lif_tx"] + spk_sum)
            s_tx = s_spk

            y_skip = self.rx_skip(s_tx)

            r1_spk = self.rx_lif1(self.rx_fc1(s_tx))
            self._acc_spike_vec("rx_lif1", r1_spk)
            r1_sum = r1_spk.float().sum(dim=0)
            grad_sum["rx_lif1"] = r1_sum if grad_sum["rx_lif1"] is None \
                else (grad_sum["rx_lif1"] + r1_sum)

            rx_feat_t = self.rx_drop1(r1_spk)
            rx_feat_acc = rx_feat_t if rx_feat_acc is None \
                else (rx_feat_acc + rx_feat_t)

            y_main = self.fc_out(rx_feat_t)
            if self.skip_norm_align:
                mn = y_main.detach().norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                sn = y_skip.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                y_skip = y_skip * (mn / sn)
            y_out = y_main + self.skip_alpha * y_skip

            y_step = y_out.view(B, 2, self.H, self.W)
            y_steps.append(y_step)

            mse_t = (y_step - x).pow(2).mean(dim=(1, 2, 3)).mean()
            step_mse_list.append(mse_t)
            with torch.no_grad():
                self.run_step_mse_sum[t] += mse_t.detach()
                self.run_step_mse_cnt[t] += 1

        y_est = torch.stack(y_steps, dim=0).mean(dim=0)

        self.last_rate = {name: grad_sum[name] / float(B * K)
                          for name in self._spk_layer_dims}
        self.step_mse_vec  = torch.stack(step_mse_list, dim=0)
        self.step_mse_mean = float(self.step_mse_vec.mean().detach())
        return y_est, rx_feat_acc / float(K)
