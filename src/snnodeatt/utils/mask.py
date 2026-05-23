"""Mask utilities for sequence-level SNN-ODE-ATT data flow.

The canonical multidataset path uses sample-level masks with shape ``[B, T]``.
Legacy wafer code may still provide ``[B, T, D]`` masks. That legacy form is
accepted only when all feature channels share the same validity at each time
step, which matches the current non-channel-asynchronous sampling assumption.
"""

from __future__ import annotations

import torch


def normalize_sequence_mask(
    mask: torch.Tensor | None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
    atol: float = 1e-6,
    strict: bool = True,
) -> torch.Tensor:
    """Return a canonical sample-level mask with shape ``[B, T]``.

    Accepted input shapes:
    - ``[B, T]``
    - ``[B, T, 1]``
    - ``[B, T, D]`` only if all channels are equal at each time step.

    ``[B, T, D]`` masks with channel-wise asynchronous missingness are rejected
    because the current mainline assumes all sensors in a sample share the same
    observed times after non-homogeneous Poisson sampling.
    """

    if mask is None:
        if batch_size is None or seq_len is None:
            raise ValueError("mask=None requires batch_size and seq_len.")
        out = torch.ones(batch_size, seq_len, dtype=dtype or torch.float32, device=device)
        return out

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    if device is not None:
        mask = mask.to(device=device)

    if mask.dim() == 2:
        out = mask
    elif mask.dim() == 3:
        if mask.shape[-1] == 1:
            out = mask[..., 0]
        else:
            ref = mask[..., :1]
            max_diff = (mask - ref).abs().max()
            if bool(max_diff > atol):
                msg = (
                    "Channel-wise asynchronous/value-level masks are not "
                    f"supported by this mainline; max channel diff={float(max_diff):.6g}."
                )
                if strict:
                    raise ValueError(msg)
                print(f"[WARN] {msg} Using mask.mean(-1)>0 as a fallback.")
                out = (mask.float().mean(dim=-1) > 0).to(mask.dtype)
            else:
                out = ref[..., 0]
    else:
        raise ValueError(f"Illegal mask shape {tuple(mask.shape)}; expected [B,T], [B,T,1], or [B,T,D].")

    if batch_size is not None and out.shape[0] != batch_size:
        raise ValueError(f"mask batch mismatch: expected {batch_size}, got {out.shape[0]}.")
    if seq_len is not None and out.shape[1] != seq_len:
        raise ValueError(f"mask time mismatch: expected {seq_len}, got {out.shape[1]}.")

    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def is_tail_padded(mask_bt: torch.Tensor) -> bool:
    """Return True when every row has all valid positions before padding."""

    mask_bool = normalize_sequence_mask(mask_bt, dtype=torch.float32) > 0
    seen_pad = torch.cumsum((~mask_bool).to(torch.int64), dim=1) > 0
    invalid_after_pad = mask_bool & seen_pad
    return not bool(invalid_after_pad.any())
