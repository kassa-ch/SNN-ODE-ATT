import torch
import torch.nn as nn


class _BaseStatModel(nn.Module):
    """
    Statistical sequence model with native 7-output contract:
    (recons, preds, z, state_seq_1, state_seq_2, aux_seq_1, h_seq)

    This class keeps statistical fitting via closed-form data statistics and
    does not rely on gradient training.
    """

    def __init__(self, input_dim, hidden_dim=32, variant="linear"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.variant = variant
        self.event_seq_kind = "none"
        self.is_stat_model = True

        self.register_buffer("ar_a", torch.ones(self.input_dim))
        self.register_buffer("ar_b", torch.zeros(self.input_dim))
        self.register_buffer("feat_mean", torch.zeros(self.input_dim))
        self.register_buffer("feat_std", torch.ones(self.input_dim))

        self.state_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        for p in self.state_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def fit_statistical(self, train_loader, device=None):
        x_all = []
        m_all = []
        for x, m, _dt, _ in train_loader:
            x_all.append(x.float().cpu())
            m_all.append(m.float().cpu())

        if not x_all:
            return

        x = torch.cat(x_all, dim=0)   # [N,T,D]
        m = torch.cat(m_all, dim=0)   # [N,T,D]

        x_prev = x[:, :-1, :]
        y_next = x[:, 1:, :]
        pair_mask = (m[:, :-1, :] * m[:, 1:, :]).clamp_min(0.0)

        denom = pair_mask.sum(dim=(0, 1)).clamp_min(1.0)
        mx = (x_prev * pair_mask).sum(dim=(0, 1)) / denom
        my = (y_next * pair_mask).sum(dim=(0, 1)) / denom
        cov = (((x_prev - mx) * (y_next - my)) * pair_mask).sum(dim=(0, 1)) / denom
        var = (((x_prev - mx) ** 2) * pair_mask).sum(dim=(0, 1)).clamp_min(1e-6) / denom

        a = cov / var
        b = my - a * mx

        if self.variant == "lasso":
            # soft-threshold shrinkage to mimic sparse AR
            lam = 0.02
            a = torch.sign(a) * torch.clamp(torch.abs(a) - lam, min=0.0)
        elif self.variant == "krr":
            # smooth AR gain to mimic kernel regularization
            a = 0.8 * a + 0.2 * torch.tanh(a)
        elif self.variant == "fpca":
            # lower-gain linear dynamics for stable low-rank-like reconstruction
            a = torch.clamp(a, -0.95, 0.95) * 0.7
        elif self.variant == "cumint":
            # conservative integral-style trend memory
            a = torch.clamp(a, -0.5, 0.5)

        target_device = self.ar_a.device
        self.ar_a.copy_(a.to(target_device))
        self.ar_b.copy_(b.to(target_device))

        obs = x[m > 0]
        if obs.numel() > 0:
            mean = obs.view(-1, self.input_dim).mean(dim=0)
            std = obs.view(-1, self.input_dim).std(dim=0).clamp_min(1e-6)
            self.feat_mean.copy_(mean.to(target_device))
            self.feat_std.copy_(std.to(target_device))

    def _build_outputs(self, x, mask):
        # Reconstruct (denoise-like): blend AR estimate with current value.
        ar_recon = self.ar_a.view(1, 1, -1) * x + self.ar_b.view(1, 1, -1)
        recons = 0.5 * x + 0.5 * ar_recon

        # One-step prediction with AR on previous observation.
        preds = torch.zeros_like(x)
        preds[:, 0, :] = ar_recon[:, 0, :]
        preds[:, 1:, :] = self.ar_a.view(1, 1, -1) * x[:, :-1, :] + self.ar_b.view(1, 1, -1)

        recons = recons * mask
        preds = preds * mask

        # Unified state semantics for downstream loss/demo compatibility.
        norm_state = (recons - self.feat_mean.view(1, 1, -1)) / self.feat_std.view(1, 1, -1)
        mem_seq = torch.tanh(self.state_proj(norm_state))
        mem_reset_seq = mem_seq
        aux_seq = torch.zeros_like(mem_seq)
        h_seq = torch.cat([mem_reset_seq, aux_seq], dim=-1)
        z = mem_reset_seq.mean(dim=1)
        return recons, preds, z, mem_seq, mem_reset_seq, aux_seq, h_seq

    def forward(self, x, mask, delta_t=None, delta_ts=None):
        return self._build_outputs(x.float(), mask.float())


class PredictiveStatLinearAR_Model(_BaseStatModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, variant="linear")


class PredictiveStatLassoAR_Model(_BaseStatModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, variant="lasso")


class PredictiveStatKRR_Model(_BaseStatModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, variant="krr")


class PredictiveStatFPCA_Model(_BaseStatModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, variant="fpca")


class PredictiveStatCumInt_Model(_BaseStatModel):
    def __init__(self, input_dim, hidden_dim=32, **kwargs):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, variant="cumint")
