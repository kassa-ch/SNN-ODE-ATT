import torch
import torch.nn as nn


class EnhancedSNNLoss(nn.Module):
    """
    Unified loss for all models:
    L_total = alpha_recon * L_recon + beta_pred * L_pred + gamma_stability * L_stability

    Notes:
    - No anomaly-density term is introduced.
    - Stability is applied to one unified semantic variable: mem_seq = M_pre(t_k).
    """

    def __init__(
        self,
        alpha_recon=0.6,
        beta_pred=0.3,
        gamma_stability=0.003,
        exclude_positional_encoding_from_loss=True,
        position_encoding_dims=2,
    ):
        super().__init__()
        self.alpha_recon = alpha_recon
        self.beta_pred = beta_pred
        self.gamma_stability = gamma_stability
        self.exclude_positional_encoding_from_loss = exclude_positional_encoding_from_loss
        self.position_encoding_dims = position_encoding_dims

    def _select_signal_dims(self, tensor):
        if tensor is None:
            return None
        if not self.exclude_positional_encoding_from_loss:
            return tensor
        if tensor.dim() < 2:
            return tensor
        if tensor.dim() == 2:
            return tensor

        feat_dim = tensor.shape[-1]
        keep_dim = max(feat_dim - self.position_encoding_dims, 1)
        return tensor[..., :keep_dim]

    def _masked_mse(self, pred, target, mask=None):
        err = (pred - target) ** 2
        if mask is not None:
            if mask.dim() == err.dim() - 1:
                mask = mask.unsqueeze(-1)
            if mask.dim() == err.dim() and mask.shape[-1] == 1 and err.shape[-1] != 1:
                mask = mask.expand_as(err)
            err = err * mask
            denom = mask.sum().clamp_min(1.0)
            return err.sum() / denom
        return err.mean()

    def _compute_reconstruction_loss(self, reconstructions, x, mask):
        reconstructions = self._select_signal_dims(reconstructions)
        x = self._select_signal_dims(x)
        mask = self._select_signal_dims(mask) if mask is not None else None
        return self._masked_mse(reconstructions, x, mask)

    def _compute_prediction_loss(self, predictions, x, mask):
        if predictions is None:
            return x.new_tensor(0.0)

        _, seq_len, _ = x.shape
        if seq_len <= 1:
            return x.new_tensor(0.0)

        predictions = self._select_signal_dims(predictions)
        x = self._select_signal_dims(x)
        mask = self._select_signal_dims(mask) if mask is not None else None

        pred_input = predictions[:, :-1, :]
        pred_target = x[:, 1:, :]

        if mask is not None:
            if mask.dim() == 3:
                pred_mask = mask[:, 1:, :]
            elif mask.dim() == 2:
                pred_mask = mask[:, 1:]
            else:
                raise ValueError(f"Illegal mask shape: {tuple(mask.shape)}")
            return self._masked_mse(pred_input, pred_target, pred_mask)

        return self._masked_mse(pred_input, pred_target, None)

    def _compute_stability_loss(self, mem_seq, mask=None, delta_ts=None):
        """
        mem_seq corresponds to M_pre(t_k). The regularizer is:
            L_stability = mean ||M_pre(t_k)||^2
        """
        if mem_seq is None:
            raise ValueError("EnhancedSNNLoss requires mem_seq for stability regularization.")

        if mem_seq.dim() != 3:
            raise ValueError(f"mem_seq shape must be [B,T,H], got {tuple(mem_seq.shape)}")

        state_sq = mem_seq ** 2

        if mask is None:
            return state_sq.mean()

        if mask.dim() == 3:
            state_mask = mask.float().mean(dim=-1, keepdim=True)   # [B,T,1]
        elif mask.dim() == 2:
            state_mask = mask.float().unsqueeze(-1)                # [B,T,1]
        else:
            raise ValueError(f"Illegal mask shape: {tuple(mask.shape)}")

        state_sq = state_sq * state_mask
        denom = state_mask.sum().clamp_min(1.0) * mem_seq.size(-1)
        return state_sq.sum() / denom

    def forward(
        self,
        reconstructions,
        predictions,
        x,
        mask,
        z_mean=None,
        mem_seq=None,
        mem_reset_seq=None,
        spike_seq=None,
        rate_seq=None,
        h_seq=None,
        delta_t=None,
        delta_ts=None,
        **kwargs,
    ):
        total_loss = x.new_tensor(0.0)
        loss_info = {}

        recon_loss = self._compute_reconstruction_loss(reconstructions, x, mask)
        total_loss = total_loss + self.alpha_recon * recon_loss
        loss_info["recon_loss"] = float(recon_loss.item())

        pred_loss = self._compute_prediction_loss(predictions, x, mask)
        total_loss = total_loss + self.beta_pred * pred_loss
        loss_info["pred_loss"] = float(pred_loss.item())

        if delta_ts is None:
            delta_ts = delta_t

        stability_loss = self._compute_stability_loss(
            mem_seq=mem_seq,
            mask=mask,
            delta_ts=delta_ts,
        )
        weighted_stability_loss = self.gamma_stability * stability_loss
        total_loss = total_loss + weighted_stability_loss

        loss_info["stability_loss"] = float(stability_loss.item())
        loss_info["weighted_stability"] = float(weighted_stability_loss.item())
        loss_info["total_loss"] = float(total_loss.item())

        return total_loss, loss_info

    def print_config(self):
        print("=== EnhancedSNNLoss Config ===")
        print(f"alpha_recon: {self.alpha_recon}")
        print(f"beta_pred: {self.beta_pred}")
        print(f"gamma_stability: {self.gamma_stability}")
        print(f"exclude_positional_encoding_from_loss: {self.exclude_positional_encoding_from_loss}")
        print(f"position_encoding_dims: {self.position_encoding_dims}")
        print("stability variable: mem_seq (= M_pre(t_k))")
        print("==============================")

    def enable_debug(self):
        self._debug_count = 0
