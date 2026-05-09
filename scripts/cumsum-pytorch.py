# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified: Replace Triton cumsum kernels with PyTorch torch.cumsum wrappers.
# Avoids Triton codegen bugs (triton-lang/triton#3017: tl.cumsum + tl.sum interaction)
# that cause crashes on gfx1151 and H100. Negligible throughput impact since cumsum
# is a lightweight operation. See FLA #734 for production validation on 128x H100.

import torch
import torch.nn.functional as F

from fla.utils import input_guard


def _cumsum_local_scalar_impl(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: float | None,
    head_first: bool,
) -> torch.Tensor:
    """Local cumsum within chunks for scalar (3D) tensors, no cu_seqlens."""
    if head_first:
        B, H, T = g.shape
        time_dim = 2
    else:
        B, T, H = g.shape
        time_dim = 1

    BT = chunk_size
    pad_size = (BT - T % BT) % BT

    g = g.float()
    if pad_size > 0:
        if head_first:
            g = F.pad(g, (0, pad_size))  # pad last dim (T)
        else:
            g = F.pad(g, (0, 0, 0, pad_size))  # pad T dim (second-to-last)

    if head_first:
        # [B, H, T_padded] -> [B, H, NT, BT]
        g = g.reshape(B, H, -1, BT)
        chunk_dim = 3
    else:
        # [B, T_padded, H] -> [B, NT, BT, H]
        g = g.reshape(B, -1, BT, H)
        chunk_dim = 2

    if reverse:
        g = g.flip(chunk_dim).cumsum(chunk_dim).flip(chunk_dim)
    else:
        g = g.cumsum(chunk_dim)

    if scale is not None:
        g = g * scale

    if head_first:
        g = g.reshape(B, H, -1)
        if pad_size > 0:
            g = g[:, :, :T]
    else:
        g = g.reshape(B, -1, H)
        if pad_size > 0:
            g = g[:, :T, :]

    return g


def _cumsum_local_vector_impl(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: float | None,
    head_first: bool,
) -> torch.Tensor:
    """Local cumsum within chunks for vector (4D) tensors, no cu_seqlens."""
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape

    BT = chunk_size
    pad_size = (BT - T % BT) % BT

    g = g.float()
    if pad_size > 0:
        if head_first:
            g = F.pad(g, (0, 0, 0, pad_size))  # pad T dim
        else:
            g = F.pad(g, (0, 0, 0, 0, 0, pad_size))  # pad T dim

    if head_first:
        # [B, H, T_padded, S] -> [B, H, NT, BT, S]
        g = g.reshape(B, H, -1, BT, S)
        chunk_dim = 3
    else:
        # [B, T_padded, H, S] -> [B, NT, BT, H, S]
        g = g.reshape(B, -1, BT, H, S)
        chunk_dim = 2

    if reverse:
        g = g.flip(chunk_dim).cumsum(chunk_dim).flip(chunk_dim)
    else:
        g = g.cumsum(chunk_dim)

    if scale is not None:
        g = g * scale

    if head_first:
        g = g.reshape(B, H, -1, S)
        if pad_size > 0:
            g = g[:, :, :T, :]
    else:
        g = g.reshape(B, -1, H, S)
        if pad_size > 0:
            g = g[:, :T, :, :]

    return g


def _cumsum_global_scalar_impl(
    s: torch.Tensor,
    reverse: bool,
    scale: float | None,
    head_first: bool,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Global cumsum across full sequences for scalar (3D) tensors."""
    s = s.float()
    if head_first:
        time_dim = 2
    else:
        time_dim = 1

    if cu_seqlens is not None:
        # Variable-length: process each sequence separately
        result = torch.zeros_like(s)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if head_first:
                seq = s[:, :, bos:eos]
                if reverse:
                    seq = seq.flip(2).cumsum(2).flip(2)
                else:
                    seq = seq.cumsum(2)
                if scale is not None:
                    seq = seq * scale
                result[:, :, bos:eos] = seq
            else:
                seq = s[:, bos:eos, :]
                if reverse:
                    seq = seq.flip(1).cumsum(1).flip(1)
                else:
                    seq = seq.cumsum(1)
                if scale is not None:
                    seq = seq * scale
                result[:, bos:eos, :] = seq
        return result
    else:
        if reverse:
            z = s.flip(time_dim).cumsum(time_dim).flip(time_dim)
        else:
            z = s.cumsum(time_dim)
        if scale is not None:
            z = z * scale
        return z


def _cumsum_global_vector_impl(
    s: torch.Tensor,
    reverse: bool,
    scale: float | None,
    head_first: bool,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Global cumsum across full sequences for vector (4D) tensors."""
    s = s.float()
    if head_first:
        time_dim = 2
    else:
        time_dim = 1

    if cu_seqlens is not None:
        result = torch.zeros_like(s)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if head_first:
                seq = s[:, :, bos:eos, :]
                if reverse:
                    seq = seq.flip(2).cumsum(2).flip(2)
                else:
                    seq = seq.cumsum(2)
                if scale is not None:
                    seq = seq * scale
                result[:, :, bos:eos, :] = seq
            else:
                seq = s[:, bos:eos, :, :]
                if reverse:
                    seq = seq.flip(1).cumsum(1).flip(1)
                else:
                    seq = seq.cumsum(1)
                if scale is not None:
                    seq = seq * scale
                result[:, bos:eos, :, :] = seq
        return result
    else:
        if reverse:
            z = s.flip(time_dim).cumsum(time_dim).flip(time_dim)
        else:
            z = s.cumsum(time_dim)
        if scale is not None:
            z = z * scale
        return z


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    if cu_seqlens is not None:
        # Variable-length: process each sequence separately, then stitch back
        result = torch.empty_like(g, dtype=output_dtype or g.dtype)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if head_first:
                seq = g[:, :, bos:eos]
            else:
                seq = g[:, bos:eos, :]
            seq_out = _cumsum_local_scalar_impl(seq, chunk_size, reverse, scale, head_first)
            out_dtype = output_dtype or g.dtype
            if head_first:
                result[:, :, bos:eos] = seq_out.to(out_dtype)
            else:
                result[:, bos:eos, :] = seq_out.to(out_dtype)
        return result
    else:
        out = _cumsum_local_scalar_impl(g, chunk_size, reverse, scale, head_first)
        return out.to(output_dtype or g.dtype)


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    if cu_seqlens is not None:
        result = torch.empty_like(g, dtype=output_dtype or g.dtype)
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if head_first:
                seq = g[:, :, bos:eos, :]
            else:
                seq = g[:, bos:eos, :, :]
            seq_out = _cumsum_local_vector_impl(seq, chunk_size, reverse, scale, head_first)
            out_dtype = output_dtype or g.dtype
            if head_first:
                result[:, :, bos:eos, :] = seq_out.to(out_dtype)
            else:
                result[:, bos:eos, :, :] = seq_out.to(out_dtype)
        return result
    else:
        out = _cumsum_local_vector_impl(g, chunk_size, reverse, scale, head_first)
        return out.to(output_dtype or g.dtype)


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    out = _cumsum_global_scalar_impl(s, reverse, scale, head_first, cu_seqlens)
    return out.to(output_dtype or s.dtype)


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    out = _cumsum_global_vector_impl(s, reverse, scale, head_first, cu_seqlens)
    return out.to(output_dtype or s.dtype)


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
            f"or [B, H, T]/[B, H, T, D] otherwise",
        )


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise",
        )
