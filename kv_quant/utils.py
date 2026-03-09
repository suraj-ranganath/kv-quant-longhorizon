from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Iterator, Tuple

import torch

EPS = 1e-8


def _reshape_blocks(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int]:
    # x shape: [B, L, H, D]
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B, L, H, D], got shape={tuple(x.shape)}")
    b, l, h, d = x.shape
    pad_len = (block_size - (l % block_size)) % block_size
    if pad_len:
        pad = torch.zeros((b, pad_len, h, d), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    nb = x.shape[1] // block_size
    return x.view(b, nb, block_size, h, d), pad_len


def _unshape_blocks(xb: torch.Tensor, pad_len: int, orig_len: int) -> torch.Tensor:
    x = xb.reshape(xb.shape[0], xb.shape[1] * xb.shape[2], xb.shape[3], xb.shape[4])
    if pad_len:
        x = x[:, :orig_len]
    return x


def quantize_asym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmin, qmax = 0, (1 << bits) - 1
    x_min = x.amin(dim=reduce_dims, keepdim=True)
    x_max = x.amax(dim=reduce_dims, keepdim=True)
    scale = ((x_max - x_min) / max(qmax - qmin, 1)).clamp_min(EPS)
    zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zp).clamp(qmin, qmax).to(torch.int8)
    return q, scale.to(torch.float16), zp.to(torch.float16)


def dequantize_asym(
    q: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (q.to(dtype) - zp.to(dtype)) * scale.to(dtype)


def quantize_sym(x: torch.Tensor, bits: int, reduce_dims: Tuple[int, ...]):
    qmax = (1 << (bits - 1)) - 1
    x_abs = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (x_abs / max(qmax, 1)).clamp_min(EPS)
    q = torch.round(x / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_sym(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("Last dimension must be power of two for Hadamard transform.")
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, h * 2)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*x.shape)
        h <<= 1
    return y / math.sqrt(n)


@contextmanager
def timed() -> Iterator[list]:
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
