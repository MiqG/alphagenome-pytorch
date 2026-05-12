"""Unified training utilities for AlphaGenome fine-tuning.

Provides common training functions for both RNA-seq and ATAC-seq modalities.
Includes enhanced versions with DDP support, profiling, and Pearson R metrics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

from alphagenome_pytorch.losses import (
    multinomial_loss,
    cross_entropy_loss,
    cross_entropy_loss_normalized,
    cross_entropy_loss_from_logits,
    binary_crossentropy_from_logits,
    poisson_loss,
)
from alphagenome_pytorch.heads import (
    SpliceSitesClassificationHead,
    SpliceSitesUsageHead,
    SpliceSitesJunctionHead,
)

# Number of segments for multinomial loss computation.
# AlphaGenome divides sequences into 8 equal segments for numerical stability.
NUM_SEGMENTS = 8

# Tuple of all splice head types for isinstance checks
SPLICE_HEAD_TYPES = (SpliceSitesClassificationHead, SpliceSitesUsageHead, SpliceSitesJunctionHead)

if TYPE_CHECKING:
    from torch.optim import Optimizer


def collate_genomic(
    batch: list[tuple[Tensor, dict[int, Tensor]]],
) -> tuple[Tensor, dict[int, Tensor]]:
    """Collate function for genomic fine-tuning datasets.

    Args:
        batch: List of (sequence, targets_dict) tuples from dataset.

    Returns:
        Tuple of (sequences, targets_dict) where:
            - sequences: Stacked sequences tensor (batch, seq_len, 4)
            - targets_dict: Dict mapping resolution to targets (batch, out_len, n_tracks)

    Example:
        >>> batch = [(seq1, {1: t1_1bp, 128: t1_128bp}), (seq2, {1: t2_1bp, 128: t2_128bp})]
        >>> sequences, targets = collate_genomic(batch)
        >>> targets[1].shape, targets[128].shape
    """
    sequences = torch.stack([item[0] for item in batch])

    # Targets are always a dict
    first_targets = batch[0][1]
    targets_dict: dict[int, Tensor] = {}
    for res in first_targets.keys():
        targets_dict[res] = torch.stack([item[1][res] for item in batch])

    return sequences, targets_dict


def _top_k_positions_from_logits(logits_ncl: torch.Tensor, top_k: int) -> torch.Tensor:
    """Derive splice-site positions from classification head logits via top-k selection.

    Args:
        logits_ncl: (B, 5, S) — NCL logits from SpliceSitesClassificationHead
                    (channels: 0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-).
        top_k: Maximum number of positions to select per role per batch item.

    Returns:
        positions: (B, 4, top_k) int32 tensor — [pos_donors, pos_acceptors,
                   neg_donors, neg_acceptors], padded with -1 where fewer than
                   top_k sites exist.
    """
    B, C, S = logits_ncl.shape
    device = logits_ncl.device
    positions = torch.full((B, 4, top_k), -1, dtype=torch.int32, device=device)
    k = min(top_k, S)
    for role_idx in range(4):  # Donor+, Acceptor+, Donor-, Acceptor-
        scores = logits_ncl[:, role_idx, :]  # (B, S)
        topk_idx = torch.topk(scores, k, dim=-1, sorted=True).indices
        # Sort positions in ascending genomic order so RoPE distances are stable.
        sorted_idx, _ = topk_idx.sort(dim=-1)
        positions[:, role_idx, :k] = sorted_idx.to(torch.int32)
    return positions


def _call_splice_head(
    head,
    embeddings_dict,
    organism_idx,
    positions,
    channels_last,
    cls_head=None,
    junction_top_k: int | None = None,
):
    """Call a splice head with the training-loop's embeddings_dict interface.

    Unwraps embeddings_dict[1] and calls the correct forward signature per head type.

    Args:
        head: SpliceSitesClassificationHead, SpliceSitesUsageHead, or SpliceSitesJunctionHead.
        embeddings_dict: Dict with key 1 → embeddings tensor (B, C, S) or (B, S, C).
        organism_idx: Organism indices, shape (B,) or (B, 1).
        positions: Annotated splice-site positions (B, 4, K) or None.
            Ignored for SpliceSitesJunctionHead when junction_top_k is set.
        channels_last: If True, embeddings are (B, S, C); if False, (B, C, S).
        cls_head: SpliceSitesClassificationHead used to derive positions when
            junction_top_k is not None. Required when junction_top_k is set.
        junction_top_k: If set, positions for SpliceSitesJunctionHead are derived
            from the top-k scoring sites predicted by cls_head rather than from
            the annotated positions tensor.

    Returns:
        Dict compatible with _compute_splice_loss():
        - {1: logits} for classification/usage heads
        - {pos_counts: ..., neg_counts: ...} for junction head
        - {} if junction head and no positions available
    """
    if 1 not in embeddings_dict:
        available_keys = list(embeddings_dict.keys())
        raise ValueError(
            f"embeddings_dict missing key 1 for splice heads. Available: {available_keys}. "
            f"Make sure resolutions include 1bp for splice modalities."
        )
    emb = embeddings_dict[1]
    org = organism_idx[:, 0] if organism_idx.ndim > 1 else organism_idx
    org = torch.zeros_like(org)

    # Debug: Check embedding shape
    assert emb.ndim == 3, f"Expected 3D embeddings, got shape {emb.shape}"

    if isinstance(head, SpliceSitesJunctionHead):
        if junction_top_k is not None:
            if cls_head is None:
                raise ValueError(
                    "junction_top_k requires cls_head (SpliceSitesClassificationHead) "
                    "to be passed to _call_splice_head."
                )
            # Run classification head to get per-position scores; always NCL internally.
            emb_for_cls = emb if not channels_last else emb.transpose(1, 2)
            cls_out = cls_head(emb_for_cls, org, channels_last=False)
            positions = _top_k_positions_from_logits(cls_out["logits"], junction_top_k)
        if positions is None:
            return {}
        # Clamp -1 padding to 0 to avoid PyTorch negative indexing wrapping.
        # Padded positions use -1, but negative indices wrap to last position in PyTorch.
        # Clamping to 0 ensures a safe dummy index; output predictions are masked anyway.
        positions_clamped = positions.clamp(min=0)
        out = head(emb, org, splice_site_positions=positions_clamped, channels_last=channels_last)
        n_tissues = head._num_tissues
        return {
            "pos_counts": out["pred_counts"][..., :n_tissues],
            "neg_counts": out["pred_counts"][..., n_tissues:],
        }
    else:
        out = head(emb, org, channels_last=channels_last)
        logits = out["logits"]  # (B, S, C) if channels_last, else (B, C, S)
        # Always transpose to NLC (B, S, C) for training loop compatibility
        if channels_last:
            # Already NLC, no transpose needed
            pass
        else:
            # NCL to NLC: (B, C, S) → (B, S, C)
            logits = logits.transpose(1, 2)
        return {1: logits}


def _ce_loss_with_smoothing(pred: torch.Tensor, target: torch.Tensor, label_smoothing: float, n_classes: int) -> torch.Tensor:
    target_smooth = (1.0 - label_smoothing) * target.float() + label_smoothing / n_classes
    mask = target.any(dim=-1, keepdim=True).expand_as(pred)
    return cross_entropy_loss_from_logits(y_pred_logits=pred, y_true=target_smooth, mask=mask, axis=-1)


def _partitioned_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_partitions: int,
    loss_fn,
    mask_fn,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss as sum of per-partition losses along the sequence dimension (dim=1).

    Splits pred and target into num_partitions equal chunks along dim=1. Partitions
    that have no valid positions (mask is all False) are skipped. The final loss is
    the sum over non-empty partitions, upweighting the signal relative to a global mean.
    """
    seq_len = pred.shape[1]
    chunk_size = seq_len // num_partitions
    chunk_losses = []
    for i in range(num_partitions):
        start = i * chunk_size
        end = start + chunk_size if i < num_partitions - 1 else seq_len
        p_chunk = pred[:, start:end, :]
        t_chunk = target[:, start:end, :]
        # Skip partitions with no valid positions
        if not mask_fn(t_chunk).any():
            continue
        chunk_losses.append(loss_fn(p_chunk, t_chunk))
    if not chunk_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(chunk_losses).sum()


def _soft_clip_counts(counts: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    return torch.where(counts > clip, 2.0 * torch.sqrt(counts * clip) - clip, counts)


def _compute_junction_strand_loss(pred_counts, target_counts, donor_pos, accept_pos, device,
                                   junction_loss: str = "original"):
    """Strand-specific junction loss matching JAX SpliceSitesJunctionHead.loss.

    loss = 0.2 * (CE(axis=donor) + CE(axis=acceptor)) + 0.04 * (Poisson(axis=donor) + Poisson(axis=acceptor))

    pairs_mask[b,d,a,s] = (donor_pos[b,d] >= 0) & (accept_pos[b,a] >= 0)

    Args:
        junction_loss: "original" uses cross_entropy_loss (JAX pre-de264f5);
                       "normalized" uses cross_entropy_loss_normalized (JAX post-de264f5,
                       both targets and predictions normalized to ratios within mask).
    """
    valid_d = (donor_pos >= 0).float()
    valid_a = (accept_pos >= 0).float()
    pairs_mask = torch.einsum('bd,ba->bda', valid_d, valid_a).bool()
    pairs_mask = pairs_mask.unsqueeze(-1).expand_as(pred_counts)

    if not pairs_mask.any():
        return torch.tensor(0.0, device=device, dtype=pred_counts.dtype)

    target = torch.where(pairs_mask, target_counts, torch.zeros_like(target_counts))
    pred   = torch.where(pairs_mask, pred_counts,   torch.zeros_like(pred_counts))

    # Skip intervals with no observed junction counts — the JAX cross-entropy
    # goes negative when targets are all zero, polluting the loss with noise.
    if not (target > 0).any():
        return torch.tensor(0.0, device=device, dtype=pred_counts.dtype)

    _ce = cross_entropy_loss_normalized if junction_loss == "normalized" else cross_entropy_loss
    donor_ratios_loss    = _ce(y_true=target, y_pred=pred, mask=pairs_mask, axis=1)
    acceptor_ratios_loss = _ce(y_true=target, y_pred=pred, mask=pairs_mask, axis=2)

    sum_pred_d = pred.sum(dim=1)
    sum_tgt_d  = _soft_clip_counts(target.sum(dim=1))
    sum_pred_a = pred.sum(dim=2)
    sum_tgt_a  = _soft_clip_counts(target.sum(dim=2))
    donor_total_loss  = poisson_loss(y_true=sum_tgt_d, y_pred=sum_pred_d, mask=pairs_mask.any(dim=1))
    accept_total_loss = poisson_loss(y_true=sum_tgt_a, y_pred=sum_pred_a, mask=pairs_mask.any(dim=2))

    return 0.2 * (donor_ratios_loss + acceptor_ratios_loss) + 0.04 * (donor_total_loss + accept_total_loss)


def _compute_splice_loss(head, predictions, targets_dict, device, num_segments: int = 1,
                         junction_loss: str = "original"):
    """Compute loss for any of the three splice head types.

    Args:
        head: SpliceSitesClassificationHead, SpliceSitesUsageHead, or SpliceSitesJunctionHead.
        predictions: Dict returned by _call_splice_head.
        targets_dict: Dict with string keys: 'probs', 'usage',
            'junction_positions', 'junction_matrix'.
        device: Torch device.
        num_segments: Number of equal-length partitions to split the sequence
            into before computing loss. Each partition's loss is computed independently
            and the results are averaged. Values > 1 upweight sequence regions with
            fewer splice sites relative to a global mean, since each partition
            contributes equally regardless of how many valid positions it contains.
            Defaults to 1 (standard global mean, unchanged behaviour).
        junction_loss: Cross-entropy variant for the junction head. "original" matches
            JAX pre-de264f5; "normalized" matches JAX post-de264f5 (ratio CE).

    Returns:
        (loss_tensor, components_dict) where components_dict has keys like
        'cls_loss', 'usage_loss', 'junction_pos_loss', 'junction_neg_loss'.
    """
    N_CLASSES = 5
    label_smoothing = 1e-7

    if isinstance(head, SpliceSitesClassificationHead):
        pred = predictions[1]
        target = targets_dict["probs"].to(device)
        if num_segments > 1:
            loss = _partitioned_loss(
                pred, target,
                num_partitions=num_segments,
                loss_fn=lambda p, t: _ce_loss_with_smoothing(p, t, label_smoothing, N_CLASSES),
                mask_fn=lambda t: t.any(dim=-1, keepdim=True).expand_as(t),
                device=device,
            )
        else:
            target_smooth = (1.0 - label_smoothing) * target.float() + label_smoothing / N_CLASSES
            mask = target.any(dim=-1, keepdim=True).expand_as(pred)
            loss = cross_entropy_loss_from_logits(
                y_pred_logits=pred,
                y_true=target_smooth,
                mask=mask,
                axis=-1,
            )
        return loss, {"cls_loss": loss.item()}

    elif isinstance(head, SpliceSitesUsageHead):
        pred = predictions[1]
        target = targets_dict["usage"].to(device)
        if num_segments > 1:
            loss = _partitioned_loss(
                pred, target,
                num_partitions=num_segments,
                loss_fn=lambda p, t: binary_crossentropy_from_logits(
                    y_pred=p,
                    y_true=t.float(),
                    mask=(t > 0).any(dim=-1, keepdim=True).expand_as(p),
                ),
                mask_fn=lambda t: (t > 0).any(dim=-1, keepdim=True).expand_as(t),
                device=device,
            )
        else:
            mask = (target > 0).any(dim=-1, keepdim=True).expand_as(pred)
            loss = binary_crossentropy_from_logits(
                y_pred=pred,
                y_true=target.float(),
                mask=mask,
            )
        return loss, {"usage_loss": loss.item()}

    elif isinstance(head, SpliceSitesJunctionHead):
        if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
            return torch.tensor(0.0, device=device), {}
        junc_matrix = targets_dict["junction_matrix"].to(device)
        positions = targets_dict["junction_positions"].to(device)
        n_s = head._num_tissues
        pos_loss = _compute_junction_strand_loss(
            predictions["pos_counts"], junc_matrix[..., :n_s],
            positions[:, 0, :].long(), positions[:, 1, :].long(), device,
            junction_loss=junction_loss,
        )
        neg_loss = _compute_junction_strand_loss(
            predictions["neg_counts"], junc_matrix[..., n_s:],
            positions[:, 2, :].long(), positions[:, 3, :].long(), device,
            junction_loss=junction_loss,
        )
        loss = pos_loss + neg_loss
        return loss, {"junction_pos_loss": pos_loss.item(), "junction_neg_loss": neg_loss.item()}

    return torch.tensor(0.0, device=device), {}


def _extract_junction_cls_per_sample(predictions, targets_dict, device):
    """Extract per-sample (scores, binary_labels) for junction true/false classification.

    Positive: valid (donor, acceptor) pair with target count > 0 (observed in RNA-seq).
    Negative: valid pair with target count == 0 (splice sites present but junction absent).

    Returns list of (scores, labels) tensors of length 2*n_tissues
    (first n_tissues = pos strand, last n_tissues = neg strand), or None.
    """
    if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
        return None

    junc_matrix = targets_dict["junction_matrix"].to(device)  # (B, D, A, 2*n_s)
    positions   = targets_dict["junction_positions"].to(device)  # (B, 4, P)
    n_s = junc_matrix.shape[-1] // 2

    per_sample = []
    for pred_key, tgt_slice, donor_row, accept_row in [
        ("pos_counts", slice(None, n_s),  0, 1),
        ("neg_counts", slice(n_s, None),  2, 3),
    ]:
        pred_counts = predictions[pred_key]                       # (B, D, A, n_s)
        tgt_counts  = junc_matrix[:, :, :, tgt_slice]            # (B, D, A, n_s)
        donor_pos   = positions[:, donor_row,  :].long()
        accept_pos  = positions[:, accept_row, :].long()
        pairs_mask  = torch.einsum(
            "bd,ba->bda",
            (donor_pos >= 0).float(),
            (accept_pos >= 0).float(),
        ).bool()                                                   # (B, D, A)

        for s in range(n_s):
            scores_s = pred_counts[:, :, :, s][pairs_mask].float().cpu()
            labels_s = (tgt_counts[:, :, :, s][pairs_mask] > 0).float().cpu()
            per_sample.append((scores_s, labels_s))

    return per_sample  # length 2*n_s


def _extract_junction_pearson_per_sample(predictions, targets_dict, device):
    """Per-biological-sample (pred, true) tensors for junction Pearson.

    Combines pos+neg strand data for each biological sample.
    Returns list of n_s dicts {"full": (pred, true), "nonzero": (pred[nz], true[nz]) | None},
    or None if data unavailable.
    """
    if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
        return None

    junc_matrix = targets_dict["junction_matrix"].to(device)  # (B, D, A, 2*n_s)
    positions   = targets_dict["junction_positions"].to(device)
    n_s = junc_matrix.shape[-1] // 2

    per_sample_pred = [[] for _ in range(n_s)]
    per_sample_true = [[] for _ in range(n_s)]

    for pred_key, tgt_slice, donor_row, accept_row in [
        ("pos_counts", slice(None, n_s),  0, 1),
        ("neg_counts", slice(n_s, None),  2, 3),
    ]:
        pred_strand = predictions[pred_key]
        tgt_strand  = junc_matrix[:, :, :, tgt_slice]
        donor_pos   = positions[:, donor_row,  :].long()
        accept_pos  = positions[:, accept_row, :].long()
        pairs_mask  = torch.einsum(
            "bd,ba->bda",
            (donor_pos >= 0).float(),
            (accept_pos >= 0).float(),
        ).bool()
        for s in range(n_s):
            per_sample_pred[s].append(pred_strand[:, :, :, s][pairs_mask].float().cpu())
            per_sample_true[s].append(tgt_strand[:, :, :, s][pairs_mask].float().cpu())

    result = []
    for s in range(n_s):
        pred_s = torch.cat(per_sample_pred[s])
        true_s = torch.cat(per_sample_true[s])
        nz = true_s > 0
        result.append({
            "full":    (pred_s, true_s),
            "nonzero": (pred_s[nz], true_s[nz]) if nz.any() else None,
        })
    return result  # length n_s


def _extract_usage_pearson_per_sample(predictions, targets_dict, device):
    """Per-sample (pred, true) tensors for splice usage Pearson.

    Returns list of n_s (pred_flat, true_flat) tuples, or None per empty sample.
    """
    if 1 not in predictions:
        return None
    pred   = torch.sigmoid(predictions[1])        # (B, S, n_s)
    target = targets_dict["usage"].to(device)     # (B, S, n_s)
    n_s    = pred.shape[-1]

    result = []
    for s in range(n_s):
        mask_s = target[:, :, s] > 0
        if not mask_s.any():
            result.append(None)
        else:
            result.append((
                pred[:, :, s][mask_s].float().cpu(),
                target[:, :, s][mask_s].float().cpu(),
            ))
    return result  # length n_s


def _extract_splice_pearson_pairs(
    head, predictions, targets_dict, device
):
    """Extract flat (N,) pred and true tensors over valid positions for Pearson R.

    For SpliceSitesUsageHead: returns (pred_flat, true_flat) tuple.
    For SpliceSitesJunctionHead: returns dict with variants:
        - "full": all valid (donor, acceptor) cells
        - "nonzero": valid cells with target > 0
        - "donor_marginal": sums along acceptor axis
        - "acceptor_marginal": sums along donor axis
        Each value is a dict with "pred" and "true" keys, or None if empty.

    Returns tuple/dict or (None, None) if no valid entries.
    """
    if isinstance(head, SpliceSitesUsageHead):
        if 1 not in predictions:
            return None, None
        pred = torch.sigmoid(predictions[1])          # (B, S, n_samples)
        target = targets_dict["usage"].to(device)     # (B, S, n_samples)
        mask = (target > 0).any(dim=-1)                # (B, S)
        if not mask.any():
            return None, None
        pred_flat = pred[mask].reshape(-1)
        true_flat = target[mask].reshape(-1)
        return pred_flat, true_flat

    elif isinstance(head, SpliceSitesJunctionHead):
        if "junction_matrix" not in targets_dict or "pos_counts" not in predictions:
            return None, None
        junc_matrix = targets_dict["junction_matrix"].to(device)
        positions = targets_dict["junction_positions"].to(device)
        n_s = head._num_tissues

        variants = {"full": {}, "nonzero": {}, "donor_marginal": {}, "acceptor_marginal": {}}
        all_pred_full, all_true_full = [], []
        all_pred_nz, all_true_nz = [], []
        all_pred_d_marg, all_true_d_marg = [], []
        all_pred_a_marg, all_true_a_marg = [], []

        for strand_idx, (pred_key, tgt_slice, donor_row, accept_row) in enumerate([
            ("pos_counts", (slice(None), slice(None), slice(None), slice(None, n_s)),   0, 1),
            ("neg_counts", (slice(None), slice(None), slice(None), slice(n_s, None)),   2, 3),
        ]):
            pred_counts = predictions[pred_key]           # (B, D, A, n_s)
            tgt_counts = junc_matrix[tgt_slice]           # (B, D, A, n_s)
            donor_pos  = positions[:, donor_row,  :].long()
            accept_pos = positions[:, accept_row, :].long()
            valid_d = (donor_pos >= 0).float()
            valid_a = (accept_pos >= 0).float()
            pairs_mask = torch.einsum('bd,ba->bda', valid_d, valid_a).bool()
            pairs_mask4 = pairs_mask.unsqueeze(-1).expand_as(pred_counts)

            # Full variant: all valid pairs
            if pairs_mask4.any():
                all_pred_full.append(pred_counts[pairs_mask4])
                all_true_full.append(tgt_counts[pairs_mask4])

            # Nonzero variant: valid pairs with target > 0
            nonzero_mask = pairs_mask4 & (tgt_counts > 0)
            if nonzero_mask.any():
                all_pred_nz.append(pred_counts[nonzero_mask])
                all_true_nz.append(tgt_counts[nonzero_mask])

            # Donor marginal: sum over acceptors, mask to donors with any valid acceptor
            donor_mask = pairs_mask.any(dim=2)  # (B, D)
            if donor_mask.any():
                pred_d = pred_counts.sum(dim=2)  # (B, D, n_s)
                true_d = tgt_counts.sum(dim=2)   # (B, D, n_s)
                donor_mask_exp = donor_mask.unsqueeze(-1).expand_as(pred_d)
                all_pred_d_marg.append(pred_d[donor_mask_exp])
                all_true_d_marg.append(true_d[donor_mask_exp])

            # Acceptor marginal: sum over donors, mask to acceptors with any valid donor
            accept_mask = pairs_mask.any(dim=1)  # (B, A)
            if accept_mask.any():
                pred_a = pred_counts.sum(dim=1)  # (B, A, n_s)
                true_a = tgt_counts.sum(dim=1)   # (B, A, n_s)
                accept_mask_exp = accept_mask.unsqueeze(-1).expand_as(pred_a)
                all_pred_a_marg.append(pred_a[accept_mask_exp])
                all_true_a_marg.append(true_a[accept_mask_exp])

        # Aggregate variants
        if all_pred_full:
            variants["full"] = {"pred": torch.cat(all_pred_full), "true": torch.cat(all_true_full)}
        if all_pred_nz:
            variants["nonzero"] = {"pred": torch.cat(all_pred_nz), "true": torch.cat(all_true_nz)}
        if all_pred_d_marg:
            variants["donor_marginal"] = {"pred": torch.cat(all_pred_d_marg), "true": torch.cat(all_true_d_marg)}
        if all_pred_a_marg:
            variants["acceptor_marginal"] = {"pred": torch.cat(all_pred_a_marg), "true": torch.cat(all_true_a_marg)}

        if not variants["full"]:
            return None, None
        return variants

    return None, None


@dataclass
class ModalityConfig:
    """Configuration for a fine-tuning modality.

    Attributes:
        name: Modality name ('rnaseq' or 'atac').
        resolutions: Tuple of output resolutions (e.g., (1, 128) or (128,)).
        default_resolution_weights: Default weights for each resolution.
        embedding_dim: Embedding dimension for ATAC (None for RNA-seq).
        positions_arg: CLI argument name for positions ('positions' or 'peaks').
    """

    name: str
    resolutions: tuple[int, ...]
    default_resolution_weights: dict[int, float]
    embedding_dim: int | None
    positions_arg: str


# Registry of modality configurations
MODALITY_CONFIGS: dict[str, ModalityConfig] = {
    "rna_seq": ModalityConfig(
        name="rna_seq",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "atac": ModalityConfig(
        name="atac",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "dnase": ModalityConfig(
        name="dnase",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "procap": ModalityConfig(
        name="procap",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "cage": ModalityConfig(
        name="cage",
        resolutions=(1, 128),
        default_resolution_weights={1: 1.0, 128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_tf": ModalityConfig(
        name="chip_tf",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "chip_histone": ModalityConfig(
        name="chip_histone",
        resolutions=(128,),
        default_resolution_weights={128: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "splice": ModalityConfig(
        name="splice",
        resolutions=(1,),
        default_resolution_weights={1: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
    "splice_junction": ModalityConfig(
        name="splice_junction",
        resolutions=(1,),
        default_resolution_weights={1: 1.0},
        embedding_dim=3072,
        positions_arg="positions",
    ),
}


def create_lr_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> LambdaLR:
    """Create learning rate scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of warmup steps (linear ramp from 0 to lr).
        total_steps: Total number of training steps.
        schedule: Schedule type after warmup. Options:
            - "cosine": Cosine decay to 0 (default)
            - "constant": Constant learning rate

    Returns:
        LambdaLR scheduler.

    Examples:
        # Warmup + cosine decay (default)
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000)

        # Constant learning rate (no warmup, no decay)
        scheduler = create_lr_scheduler(opt, warmup_steps=0, total_steps=10000, schedule="constant")

        # Warmup then constant
        scheduler = create_lr_scheduler(opt, warmup_steps=500, total_steps=10000, schedule="constant")
    """
    if schedule not in ("cosine", "constant"):
        raise ValueError(f"Unknown schedule: {schedule}. Must be 'cosine' or 'constant'.")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "constant":
            return 1.0
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_finetuning_loss(
    predictions: dict[int, Tensor],
    targets: dict[int, Tensor],
    resolution_weights: dict[int, float],
    positional_weight: float,
    device: torch.device,
    channels_last: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute combined loss across resolutions.

    Uses dynamic multinomial_resolution = seq_len // 8 for consistent loss
    granularity across different sequence lengths.

    Args:
        predictions: Dict mapping resolution to prediction tensors.
        targets: Dict mapping resolution to target tensors.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        device: Torch device.
        channels_last: If True, assumes (B, S, C). If False, assumes (B, C, S).

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains per-resolution
        losses and other metrics.
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_dict: dict[str, Tensor] = {}

    for res, weight in resolution_weights.items():
        if res not in predictions:
            continue

        pred = predictions[res]
        target = targets[res]

        # Detect dimensions based on format
        if channels_last:
            # (B, S, C)
            current_seq_len = pred.shape[-2]
            num_channels = pred.shape[-1]
            mask_shape = (pred.shape[0], 1, num_channels)
        else:
            # (B, C, S)
            current_seq_len = pred.shape[-1]
            num_channels = pred.shape[-2]
            mask_shape = (pred.shape[0], num_channels, 1)

        # Use multinomial_resolution matching JAX for 1Mb sequences (2^17 // res),
        # but allow for fewer segments if the sequence is shorter.
        # This ensures segments are at least 131k bp (at 1bp) and that
        # multinomial_resolution always divides current_seq_len.
        num_segments = max(1, min(8, current_seq_len // (131072 // res)))
        multinomial_resolution = current_seq_len // num_segments

        # Create mask (all True)
        mask = torch.ones(*mask_shape, dtype=torch.bool, device=device)

        res_loss_dict = multinomial_loss(
            y_pred=pred,
            y_true=target,
            mask=mask,
            multinomial_resolution=multinomial_resolution,
            positional_weight=positional_weight,
            channels_last=channels_last,
        )

        total_loss = total_loss + weight * res_loss_dict["loss"]
        loss_dict[f"loss_{res}bp"] = res_loss_dict["loss"]

    loss_dict["loss"] = total_loss
    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    resolutions: tuple[int, ...] | None = None,
) -> float:
    """Train for one epoch.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision (default: True).
        accumulation_steps: Number of batches to accumulate gradients over
            before performing an optimizer step. Useful for simulating larger
            batch sizes when GPU memory is limited (default: 1, no accumulation).
        resolutions: Tuple of resolutions to train on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys. Training on 1bp resolution
            requires significantly more memory than 128bp.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    head.train()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (sequences, targets_dict) in enumerate(pbar):
        sequences = sequences.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}

        # Organism index (assume human)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with amp_context:
            # Forward through trunk
            outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

            # Only get embeddings for requested resolutions (1bp is 128x larger than 128bp)
            embeddings_dict = {}
            if 1 in resolutions:
                emb = outputs.get("embeddings_1bp")
                if emb is not None:
                    embeddings_dict[1] = emb
            if 128 in resolutions:
                emb = outputs.get("embeddings_128bp")
                if emb is not None:
                    embeddings_dict[128] = emb

            # Forward through head
            predictions = head(embeddings_dict, organism_idx)

            # Compute loss
            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets_dict,
                resolution_weights=resolution_weights,
                positional_weight=positional_weight,
                device=device,
                channels_last=True,
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # Optimizer step every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

    # Handle remaining gradients if dataset size is not divisible by accumulation_steps
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    use_amp: bool = True,
    resolutions: tuple[int, ...] | None = None,
) -> float:
    """Validate the model.

    Args:
        model: AlphaGenome trunk model.
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        use_amp: Whether to use automatic mixed precision (default: True).
        resolutions: Tuple of resolutions to validate on (e.g., (1,), (128,), or (1, 128)).
            If None, inferred from resolution_weights keys.

    Returns:
        Average validation loss.
    """
    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0

    # Determine which resolutions to use
    if resolutions is None:
        resolutions = tuple(resolution_weights.keys())
    if invalid := (set(resolutions) - {1, 128}):
        raise ValueError(f"Invalid resolutions {invalid}, must be 1 or 128")

    # Set up autocast context (bfloat16 on CUDA, no-op on CPU)
    if use_amp and device.type == "cuda":
        amp_context = autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()

    for sequences, targets_dict in tqdm(val_loader, desc="Validation"):
        sequences = sequences.to(device)
        targets_dict = {k: v.to(device) for k, v in targets_dict.items() if k in resolutions}
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        with amp_context:
            outputs = model(sequences, organism_idx, return_embeddings=True, channels_last=False)

            # Only get embeddings for requested resolutions
            embeddings_dict = {}
            if 1 in resolutions:
                emb = outputs.get("embeddings_1bp")
                if emb is not None:
                    embeddings_dict[1] = emb
            if 128 in resolutions:
                emb = outputs.get("embeddings_128bp")
                if emb is not None:
                    embeddings_dict[128] = emb

            predictions = head(embeddings_dict, organism_idx)

            loss, _ = compute_finetuning_loss(
                predictions=predictions,
                targets=targets_dict,
                resolution_weights=resolution_weights,
                positional_weight=positional_weight,
                device=device,
                channels_last=True,
            )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# Re-export save_checkpoint from checkpointing module for backward compatibility
from alphagenome_pytorch.extensions.finetuning.checkpointing import save_checkpoint


# =============================================================================
# Profiling utilities
# =============================================================================


class ProfilingStats:
    """Collect timing statistics for profiling training batches.

    Example:
        >>> stats = ProfilingStats()
        >>> t0 = time.perf_counter()
        >>> # ... some operation ...
        >>> stats.add("forward", time.perf_counter() - t0)
        >>> print(stats.report(n_batches=10))
    """

    def __init__(self) -> None:
        self.times: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, elapsed: float) -> None:
        """Add a timing measurement.

        Args:
            name: Name of the operation (e.g., "forward", "backward").
            elapsed: Elapsed time in seconds.
        """
        self.times[name].append(elapsed)

    def report(self, n_batches: int) -> str:
        """Generate a profiling report.

        Args:
            n_batches: Number of batches profiled.

        Returns:
            Formatted report string with timing breakdowns.
        """
        import numpy as np

        lines = ["\n" + "=" * 70, "PROFILING REPORT", "=" * 70]
        total_time = 0.0

        for name, times in sorted(self.times.items()):
            arr = np.array(times)
            total_time += arr.sum()
            lines.append(
                f"\n{name}:\n"
                f"  Mean:  {arr.mean()*1000:.2f} ms (+/- {arr.std()*1000:.2f} ms)\n"
                f"  Total: {arr.sum():.2f} s ({len(times)} samples)"
            )

        lines.append(f"\n{'=' * 70}")
        lines.append(f"TOTAL TIME: {total_time:.2f} s for {n_batches} batches")
        lines.append(f"AVG TIME PER BATCH: {total_time/n_batches*1000:.2f} ms")

        # Breakdown percentages
        lines.append(f"\nBREAKDOWN:")
        for name, times in sorted(self.times.items()):
            pct = np.sum(times) / total_time * 100
            lines.append(f"  {name}: {pct:.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)

    def estimated_epoch_time(self, total_batches: int) -> float:
        """Estimate total epoch time based on profiled batches.

        Args:
            total_batches: Total number of batches in the epoch.

        Returns:
            Estimated epoch time in seconds.
        """
        import numpy as np

        n_profiled = len(next(iter(self.times.values()))) if self.times else 0
        if n_profiled == 0:
            return 0.0

        total_profiled_time = sum(np.sum(t) for t in self.times.values())
        avg_batch_time = total_profiled_time / n_profiled
        return avg_batch_time * total_batches


# =============================================================================
# Enhanced training functions with DDP and profiling support
# =============================================================================


def _cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA if on GPU (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _compute_multinomial_resolution(
    seq_len: int,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
) -> int:
    """Compute positions per segment for multinomial loss.

    Args:
        seq_len: Total sequence length (number of positions).
        num_segments: Target number of segments.
        min_segment_size: Minimum positions per segment (optional).

    Returns:
        Resolution (positions per segment).
    """
    resolution = max(1, seq_len // num_segments)

    if min_segment_size is not None:
        resolution = max(resolution, min_segment_size)

    return resolution


def train_epoch_ddp(
    model: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
) -> float:
    """Train for one epoch with DDP and profiling support.

    This is the enhanced version of train_epoch() with:
    - Distributed Data Parallel (DDP) support
    - Optional profiling of first N batches
    - Gradient accumulation
    - Frozen backbone optimization (memory saving when no LoRA)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone (memory optimization).
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        train_sampler: DistributedSampler for DDP (set epoch for shuffling).
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile (0 to disable).
        log_fn: Optional function to call for step logging: log_fn(metrics_dict).
        encoder_only: If True, run only the CNN encoder (skip transformer and decoder)
            and pass the raw encoder output (B, S//128, 1536) to the head as resolution
            128. The backbone is always frozen in encoder-only mode. Requires the head
            to have been created with ``create_finetuning_head(..., encoder_only=True)``.

    Returns:
        Average training loss for the epoch (synchronized across ranks).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    head.train()

    # Set epoch for distributed sampler (important for shuffling)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0

    for batch_idx, (sequences, targets_dict) in enumerate(pbar):
        is_profiling = do_profile and batch_idx < profile_batches

        # --- Data loading time (time since last batch ended) ---
        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        # --- Transfer to GPU ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # --- Forward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # When backbone is frozen (no LoRA), we can save memory by not building
        # the computation graph for the backbone forward pass.
        resolutions = tuple(resolution_weights.keys())

        if encoder_only:
            # Run only the CNN encoder; skip transformer, decoder, OutputEmbedders.
            # Backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )
        else:
            backbone_ctx = torch.no_grad() if frozen_backbone else nullcontext()
            with backbone_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolution_weights:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    emb = outputs[emb_key]
                    embeddings_dict[res] = emb.detach() if frozen_backbone else emb

                predictions = head(
                    embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                )

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward", time.perf_counter() - t0)

        # --- Loss computation ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for res, weight in resolution_weights.items():
            if res not in predictions or res not in targets_dict:
                continue

            pred = predictions[res]
            targets = targets_dict[res].to(device)

            # Scale targets from experimental space to model space
            head_module = head.module if hasattr(head, "module") else head
            targets = head_module.scale(targets, organism_idx, resolution=res, channels_last=True)
            mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

            # Compute multinomial loss
            current_seq_len = pred.shape[-2]
            multinomial_res = _compute_multinomial_resolution(
                current_seq_len, num_segments, min_segment_size
            )

            loss_dict = multinomial_loss(
                y_pred=pred,
                y_true=targets,
                mask=mask,
                multinomial_resolution=multinomial_res,
                positional_weight=positional_weight,
                count_weight=count_weight,
                channels_last=True,
            )

            res_loss = loss_dict["loss"] * weight
            loss = loss + res_loss
            loss_components[f"loss_{res}bp"] = res_loss.item()
            # Log raw (unweighted) losses for comparability across runs
            loss_components[f"loss_{res}bp_count"] = loss_dict["loss_total"].item()
            loss_components[f"loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_loss", time.perf_counter() - t0)

        # --- Backward pass ---
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        # --- Optimizer step (only every accumulation_steps batches) ---
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            # Get trainable parameters for gradient clipping
            trainable_params = [p for p in head.parameters() if p.requires_grad]
            trainable_params += [p for p in model.parameters() if p.requires_grad]

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

        # Update totals
        raw_loss = loss.item()
        total_loss += raw_loss
        n_batches += 1

        # Update running loss
        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        # Logging (only on rank 0)
        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            # Reset running loss after logging
            running_loss = 0.0
            accumulated_batches = 0

        # Print profiling report after profiling is done
        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))

            # Estimate epoch time
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print(f"  Based on {profile_batches} profiled batches, {len(train_loader)} total batches")
            print()

        # Mark end of batch for next iteration's data loading measurement
        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce loss across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    return avg_loss


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    resolution_weights: dict[int, float],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Validate the model with DDP support and Pearson R metrics.

    This is the enhanced version of validate() with:
    - Distributed Data Parallel (DDP) support with proper tensor gathering
    - Optional Pearson R computation (profile and count correlations)

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        head: Output head module.
        val_loader: Validation data loader.
        device: Torch device.
        resolution_weights: Weight for each resolution's loss.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to the head as resolution 128. Must match the setting
            used during training.

    Returns:
        Tuple of (avg_loss, metrics_dict) where metrics_dict contains:
        - Per-resolution losses (e.g., "1bp", "128bp")
        - Pearson R metrics if compute_pearson=True (profile_pearson_r_mean, count_pearson_r, etc.)
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    head.eval()

    total_loss = 0.0
    n_batches = 0
    loss_by_resolution: dict[str, float] = defaultdict(float)

    # For Pearson R computation - accumulate across ALL batches
    accumulated_profile_r: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_pred_counts: dict[int, list[Tensor]] = defaultdict(list)
    accumulated_true_counts: dict[int, list[Tensor]] = defaultdict(list)

    # Only show progress bar on rank 0
    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    for sequences, targets_dict in pbar:
        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)
        resolutions = tuple(resolution_weights.keys())

        if encoder_only:
            outputs = model(sequences, organism_idx, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"]}
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolution_weights:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Get predictions in MODEL space for loss computation
            head_module = head.module if hasattr(head, "module") else head
            predictions_scaled = head(
                embeddings_dict, organism_idx, return_scaled=True, channels_last=True
            )

            # Get predictions in EXPERIMENTAL space for Pearson R
            if compute_pearson:
                predictions_unscaled = head(
                    embeddings_dict, organism_idx, return_scaled=False, channels_last=True
                )

        loss = torch.tensor(0.0, device=device)

        for res, weight in resolution_weights.items():
            if res not in predictions_scaled or res not in targets_dict:
                continue

            pred_scaled = predictions_scaled[res]
            targets = targets_dict[res].to(device)

            # Scale targets from experimental space to model space for loss
            targets_scaled = head_module.scale(
                targets, organism_idx, resolution=res, channels_last=True
            )
            mask = torch.ones(
                pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
            )

            # Compute multinomial loss
            current_seq_len = pred_scaled.shape[-2]
            multinomial_res = _compute_multinomial_resolution(
                current_seq_len, num_segments, min_segment_size
            )

            loss_dict = multinomial_loss(
                y_pred=pred_scaled,
                y_true=targets_scaled,
                mask=mask,
                multinomial_resolution=multinomial_res,
                positional_weight=positional_weight,
                count_weight=count_weight,
                channels_last=True,
            )

            res_loss = loss_dict["loss"] * weight
            loss = loss + res_loss
            loss_by_resolution[f"{res}bp"] += res_loss.item()
            # Log raw (unweighted) losses for comparability across runs
            loss_by_resolution[f"{res}bp_count"] += loss_dict["loss_total"].item()
            loss_by_resolution[f"{res}bp_positional"] += loss_dict["loss_positional"].item()

            # Accumulate for Pearson R (in experimental space)
            if compute_pearson:
                pred_unscaled = predictions_unscaled[res]

                # Profile Pearson R: compute per-region correlation on-the-fly, store scalars
                batch_profile_r = profile_pearson_r(pred_unscaled, targets)  # (batch, tracks)
                accumulated_profile_r[res].append(batch_profile_r.float().cpu())

                # Count Pearson R: store total counts per region (tiny memory)
                accumulated_pred_counts[res].append(pred_unscaled.sum(dim=1).float().cpu())  # (batch, tracks)
                accumulated_true_counts[res].append(targets.sum(dim=1).float().cpu())

        total_loss += loss.item()
        n_batches += 1

    # Reduce across all processes
    avg_loss = total_loss / max(1, n_batches)
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

    # Compute per-resolution loss metrics (synchronized across ranks)
    metrics: dict[str, Any] = {}
    for k, v in loss_by_resolution.items():
        res_avg = v / max(1, n_batches)
        if world_size > 1:
            res_tensor = torch.tensor(res_avg, device=device)
            res_tensor = reduce_tensor(res_tensor, world_size)
            metrics[k] = res_tensor.item()
        else:
            metrics[k] = res_avg

    # Compute Pearson R metrics (must gather across all DDP ranks)
    if compute_pearson:
        for res in resolution_weights.keys():
            # Profile Pearson R (from accumulated per-region correlations)
            if res in accumulated_profile_r and accumulated_profile_r[res]:
                all_profile_r = torch.cat(accumulated_profile_r[res], dim=0)  # (N_local, tracks)

                # Gather profile correlations from all ranks
                if world_size > 1:
                    all_profile_r = gather_tensors(all_profile_r, world_size, device)

                metrics[f"{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                metrics[f"{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                # Store full distribution for wandb histogram
                metrics[f"{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

            # Count Pearson R (from accumulated counts)
            if res in accumulated_pred_counts and accumulated_pred_counts[res]:
                all_pred_counts = torch.cat(accumulated_pred_counts[res], dim=0)  # (N_local, tracks)
                all_true_counts = torch.cat(accumulated_true_counts[res], dim=0)

                # Gather counts from all ranks
                if world_size > 1:
                    all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                    all_true_counts = gather_tensors(all_true_counts, world_size, device)

                if all_pred_counts.shape[0] > 1:
                    count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)  # (tracks,)
                    metrics[f"{res}bp_count_pearson_r"] = count_r.mean().item()
                else:
                    metrics[f"{res}bp_count_pearson_r"] = float("nan")

    return avg_loss, metrics


def train_epoch_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
    save_every_steps: int | None = None,
    save_fn: Any | None = None,
    global_step_offset: int = 0,
    skip_batches: int = 0,
    save_state: dict | None = None,
    organism_idx: int = 0,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
) -> tuple[float, dict[str, float]]:
    """Train for one epoch with multiple modality heads.

    This extends train_epoch_ddp to support multi-modality training where
    each modality has its own head and weights.

    Args:
        model: AlphaGenome trunk model (may be DDP-wrapped).
        heads: Dict mapping modality name to output head module.
        train_loader: Training data loader (yields sequences, modality_targets dict).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        modality_weights: Weight for each modality's loss (e.g., {"atac": 1.0, "rna_seq": 0.5}).
        resolution_weights: Per-modality resolution weights (e.g., {"atac": {1: 1.0, 128: 1.0}}).
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, use torch.no_grad() for backbone.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        num_segments: Number of sequence segments for both multinomial count loss and splice losses.
        train_sampler: DistributedSampler for DDP.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile.
        log_fn: Optional function for step logging.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128. Backbone is always
            frozen in encoder-only mode.

    Returns:
        Tuple of (avg_total_loss, per_modality_losses).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )

    model.train()
    for head in heads.values():
        head.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # Profiling (only on rank 0)
    do_profile = profile_batches > 0 and is_main_process(rank)
    profile_stats = ProfilingStats() if do_profile else None

    if do_profile:
        print(f"\n*** PROFILING ENABLED for first {profile_batches} batches ***\n")

    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    t_batch_start = time.perf_counter()
    running_loss = 0.0
    accumulated_batches = 0
    opt_step = 0

    for batch_idx, (sequences, modality_targets) in enumerate(pbar):
        if batch_idx < skip_batches:
            continue

        is_profiling = do_profile and batch_idx < profile_batches

        if is_profiling and batch_idx > 0:
            _cuda_sync(device)
            t_data_load = time.perf_counter() - t_batch_start
            profile_stats.add("1_data_loading", t_data_load)

        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        sequences = sequences.to(device)
        organism_idx_tensor = torch.full((sequences.shape[0],), organism_idx, dtype=torch.long, device=device)

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("2_to_device", time.perf_counter() - t0)

        # Forward through backbone
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        # Collect all needed resolutions across all modalities
        all_resolutions = set()
        for modality in heads:
            all_resolutions.update(resolution_weights.get(modality, {}).keys())
        resolutions = tuple(all_resolutions)

        if encoder_only:
            # Run only the CNN encoder; backbone is always frozen in encoder-only mode.
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx_tensor, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"].detach()}
        else:
            backbone_ctx = torch.no_grad() if frozen_backbone else nullcontext()
            with backbone_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(sequences, organism_idx_tensor, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    emb = outputs[emb_key]
                    embeddings_dict[res] = emb.detach() if frozen_backbone else emb

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("3_forward_backbone", time.perf_counter() - t0)

        # Forward through each head and compute losses
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]
            head_module = head.module if hasattr(head, "module") else head

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # Embeddings are NCL (channels-first, channels_last=False).
                # Splice heads must be told this so they don't incorrectly transpose.
                if isinstance(head_module, SPLICE_HEAD_TYPES):
                    _positions = targets_dict.get("junction_positions")
                    if _positions is not None:
                        _positions = _positions.to(device)
                    _cls_head = heads.get("splice_site") if junction_top_k is not None else None
                    if _cls_head is not None:
                        _cls_head = _cls_head.module if hasattr(_cls_head, "module") else _cls_head
                    predictions = _call_splice_head(
                        head_module, embeddings_dict, organism_idx_tensor,
                        _positions, channels_last=False,
                        cls_head=_cls_head, junction_top_k=junction_top_k,
                    )
                else:
                    predictions = head(
                        embeddings_dict, organism_idx_tensor, return_scaled=True, channels_last=True
                    )

            modality_loss = torch.tensor(0.0, device=device)

            if isinstance(head_module, SPLICE_HEAD_TYPES):
                modality_loss, splice_components = _compute_splice_loss(
                    head_module, predictions, targets_dict, device,
                    num_segments=num_segments,
                    junction_loss=junction_loss,
                )
                for k, v in splice_components.items():
                    loss_components[f"{modality}_{k}"] = v
            else:
                for res, weight in res_weights.items():
                    if res not in predictions or res not in targets_dict:
                        continue

                    pred = predictions[res]
                    targets = targets_dict[res].to(device)

                    targets = head_module.scale(
                        targets, organism_idx_tensor, resolution=res, channels_last=True
                    )
                    mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

                    current_seq_len = pred.shape[-2]
                    multinomial_res = _compute_multinomial_resolution(
                        current_seq_len, num_segments, min_segment_size
                    )

                    loss_dict = multinomial_loss(
                        y_pred=pred,
                        y_true=targets,
                        mask=mask,
                        multinomial_resolution=multinomial_res,
                        positional_weight=positional_weight,
                        count_weight=count_weight,
                        channels_last=True,
                    )

                    res_loss = loss_dict["loss"] * weight
                    modality_loss = modality_loss + res_loss
                    loss_components[f"{modality}_loss_{res}bp"] = res_loss.item()
                    loss_components[f"{modality}_loss_{res}bp_count"] = loss_dict["loss_total"].item()
                    loss_components[f"{modality}_loss_{res}bp_positional"] = loss_dict["loss_positional"].item()

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            loss_components[f"{modality}_loss"] = modality_loss.item()
            modality_loss_accum[modality] += modality_loss.item()

        scaled_loss = loss / accumulation_steps

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("4_heads_and_loss", time.perf_counter() - t0)

        # Backward
        if is_profiling:
            _cuda_sync(device)
            t0 = time.perf_counter()

        scaled_loss.backward()

        if is_profiling:
            _cuda_sync(device)
            profile_stats.add("5_backward", time.perf_counter() - t0)

        # Optimizer step
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if is_accumulation_step or is_last_batch:
            if is_profiling:
                _cuda_sync(device)
                t0 = time.perf_counter()

            trainable_params = []
            for head in heads.values():
                trainable_params.extend([p for p in head.parameters() if p.requires_grad])
            trainable_params.extend([p for p in model.parameters() if p.requires_grad])

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            if is_profiling:
                _cuda_sync(device)
                profile_stats.add("6_optimizer", time.perf_counter() - t0)

            if save_every_steps is not None and save_fn is not None:
                global_step = global_step_offset + opt_step
                if global_step % save_every_steps == 0:
                    if save_state is not None:
                        save_state["batch_idx"] = batch_idx + 1
                    save_fn()

        raw_loss = loss.item()
        total_loss_accum += raw_loss
        n_batches += 1

        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                step_metrics = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                }
                log_fn(step_metrics)

            running_loss = 0.0
            accumulated_batches = 0

        if do_profile and batch_idx == profile_batches - 1:
            print(profile_stats.report(profile_batches))
            estimated_time = profile_stats.estimated_epoch_time(len(train_loader))
            print(f"\nESTIMATED EPOCH TIME: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")
            print()

        if is_profiling:
            _cuda_sync(device)
        t_batch_start = time.perf_counter()

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    return avg_loss, per_modality_loss


@torch.no_grad()
def validate_multihead(
    model: nn.Module,
    heads: dict[str, nn.Module],
    val_loader: DataLoader,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    use_amp: bool = True,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    compute_pearson: bool = True,
    rank: int = 0,
    world_size: int = 1,
    encoder_only: bool = False,
    organism_idx: int = 0,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
    compute_per_sample: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Validate model with multiple modality heads.

    Args:
        model: AlphaGenome trunk model.
        heads: Dict mapping modality name to output head module.
        val_loader: Validation data loader.
        device: Torch device.
        modality_weights: Weight for each modality's loss.
        resolution_weights: Per-modality resolution weights.
        positional_weight: Weight for positional component.
        count_weight: Weight for count component.
        use_amp: Whether to use automatic mixed precision.
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum positions per segment.
        compute_pearson: Whether to compute Pearson R metrics.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        encoder_only: If True, run only the CNN encoder and pass raw encoder output
            (B, S//128, 1536) to all heads as resolution 128.

    Returns:
        Tuple of (avg_total_loss, metrics_dict).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        gather_tensors,
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r

    model.eval()
    for head in heads.values():
        head.eval()

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0

    # For Pearson R - per modality and resolution
    accumulated_profile_r: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_pred_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}
    accumulated_true_counts: dict[str, dict[int, list[Tensor]]] = {m: defaultdict(list) for m in heads}

    # For splice Pearson R - per variant (full, nonzero, donor_marginal, acceptor_marginal)
    # For junction head: dict[modality][variant] = {"pred": [], "true": []}
    # For usage head: dict[modality]["full"] = {"pred": [], "true": []}
    accumulated_splice: dict[str, dict[str, dict[str, list[Tensor]]]] = {
        m: {"full": {"pred": [], "true": []}} for m in heads
    }

    # For classification head auPRC: accumulate logits and one-hot targets
    accumulated_cls: dict[str, dict[str, list[Tensor]]] = {
        m: {"logits": [], "true": []} for m in heads
    }

    # For junction true/false classification auPRC: per-sample scores and labels
    # accumulated_junc_cls[modality][sample_idx] = {"scores": [], "labels": []}
    accumulated_junc_cls: dict[str, dict[int, dict[str, list]]] = {m: {} for m in heads}

    # For per-sample Pearson (only populated when compute_per_sample=True)
    # accumulated_junc_ps_pearson[modality] = list[n_s dicts], each {"full":..., "nonzero":...}
    accumulated_junc_ps_pearson: dict[str, list] = {m: [] for m in heads}
    # accumulated_usage_ps_pearson[modality] = list[n_s dicts], each {"pred":[], "true":[]}
    accumulated_usage_ps_pearson: dict[str, list] = {m: [] for m in heads}

    if is_main_process(rank):
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    for sequences, modality_targets in pbar:
        sequences = sequences.to(device)
        organism_idx_tensor = torch.full((sequences.shape[0],), organism_idx, dtype=torch.long, device=device)

        # Collect all resolutions
        all_resolutions = set()
        for modality in heads:
            all_resolutions.update(resolution_weights.get(modality, {}).keys())
        resolutions = tuple(all_resolutions)

        if encoder_only:
            outputs = model(sequences, organism_idx_tensor, encoder_only=True)
            embeddings_dict = {128: outputs["encoder_output"]}
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(sequences, organism_idx_tensor, return_embeddings=True, resolutions=resolutions, channels_last=False)

            embeddings_dict = {}
            for res in resolutions:
                emb_key = f"embeddings_{res}bp"
                if emb_key in outputs:
                    embeddings_dict[res] = outputs[emb_key]

        loss = torch.tensor(0.0, device=device)

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]

            head_module = head.module if hasattr(head, "module") else head

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                if isinstance(head_module, SPLICE_HEAD_TYPES):
                    _positions = targets_dict.get("junction_positions")
                    if _positions is not None:
                        _positions = _positions.to(device)
                    _cls_head = heads.get("splice_site") if junction_top_k is not None else None
                    if _cls_head is not None:
                        _cls_head = _cls_head.module if hasattr(_cls_head, "module") else _cls_head
                    predictions_scaled = _call_splice_head(
                        head_module, embeddings_dict, organism_idx_tensor,
                        _positions, channels_last=False,
                        cls_head=_cls_head, junction_top_k=junction_top_k,
                    )
                else:
                    predictions_scaled = head(
                        embeddings_dict, organism_idx_tensor, return_scaled=True, channels_last=True
                    )
                if compute_pearson and not isinstance(head_module, SPLICE_HEAD_TYPES):
                    predictions_unscaled = head(
                        embeddings_dict, organism_idx_tensor, return_scaled=False, channels_last=True
                    )

            modality_loss = torch.tensor(0.0, device=device)

            if isinstance(head_module, SPLICE_HEAD_TYPES):
                modality_loss, _ = _compute_splice_loss(
                    head_module, predictions_scaled, targets_dict, device,
                    num_segments=num_segments,
                    junction_loss=junction_loss,
                )
                if compute_pearson:
                    # Accumulate logits + one-hot targets for auPRC (classification head only)
                    if isinstance(head_module, SpliceSitesClassificationHead):
                        if 1 in predictions_scaled and "probs" in targets_dict:
                            accumulated_cls[modality]["logits"].append(
                                predictions_scaled[1].float().cpu()
                            )
                            accumulated_cls[modality]["true"].append(
                                targets_dict["probs"].float().cpu()
                            )

                    result = _extract_splice_pearson_pairs(
                        head_module, predictions_scaled, targets_dict, device
                    )
                    if result is not None and result != (None, None):
                        # For junction head: result is a dict of variants
                        if isinstance(head_module, SpliceSitesJunctionHead):
                            if result != (None, None):
                                for variant_name, variant_data in result.items():
                                    if variant_name not in accumulated_splice[modality]:
                                        accumulated_splice[modality][variant_name] = {"pred": [], "true": []}
                                    if variant_data:
                                        accumulated_splice[modality][variant_name]["pred"].append(variant_data["pred"].float().cpu())
                                        accumulated_splice[modality][variant_name]["true"].append(variant_data["true"].float().cpu())
                            cls_pairs = _extract_junction_cls_per_sample(
                                predictions_scaled, targets_dict, device
                            )
                            if cls_pairs is not None:
                                for s, (scores_s, labels_s) in enumerate(cls_pairs):
                                    if s not in accumulated_junc_cls[modality]:
                                        accumulated_junc_cls[modality][s] = {"scores": [], "labels": []}
                                    accumulated_junc_cls[modality][s]["scores"].append(scores_s)
                                    accumulated_junc_cls[modality][s]["labels"].append(labels_s)
                            if compute_per_sample:
                                ps_junc = _extract_junction_pearson_per_sample(
                                    predictions_scaled, targets_dict, device
                                )
                                if ps_junc is not None:
                                    if not accumulated_junc_ps_pearson[modality]:
                                        accumulated_junc_ps_pearson[modality] = [
                                            {"full": {"pred": [], "true": []}, "nonzero": {"pred": [], "true": []}}
                                            for _ in range(len(ps_junc))
                                        ]
                                    for s, data in enumerate(ps_junc):
                                        for variant in ("full", "nonzero"):
                                            if data[variant] is not None:
                                                p, t = data[variant]
                                                accumulated_junc_ps_pearson[modality][s][variant]["pred"].append(p)
                                                accumulated_junc_ps_pearson[modality][s][variant]["true"].append(t)
                        # For usage head: result is (pred_flat, true_flat) tuple
                        else:
                            _splice_pred, _splice_true = result
                            if _splice_pred is not None:
                                accumulated_splice[modality]["full"]["pred"].append(_splice_pred.float().cpu())
                                accumulated_splice[modality]["full"]["true"].append(_splice_true.float().cpu())
                            if compute_per_sample:
                                ps_usage = _extract_usage_pearson_per_sample(
                                    predictions_scaled, targets_dict, device
                                )
                                if ps_usage is not None:
                                    if not accumulated_usage_ps_pearson[modality]:
                                        accumulated_usage_ps_pearson[modality] = [
                                            {"pred": [], "true": []} for _ in range(len(ps_usage))
                                        ]
                                    for s, item in enumerate(ps_usage):
                                        if item is not None:
                                            p, t = item
                                            accumulated_usage_ps_pearson[modality][s]["pred"].append(p)
                                            accumulated_usage_ps_pearson[modality][s]["true"].append(t)
            else:
                for res, weight in res_weights.items():
                    if res not in predictions_scaled or res not in targets_dict:
                        continue

                    pred_scaled = predictions_scaled[res]
                    targets = targets_dict[res].to(device)
                    targets_scaled = head_module.scale(
                        targets, organism_idx_tensor, resolution=res, channels_last=True
                    )
                    mask = torch.ones(
                        pred_scaled.shape[0], 1, pred_scaled.shape[-1], dtype=torch.bool, device=device
                    )

                    current_seq_len = pred_scaled.shape[-2]
                    multinomial_res = _compute_multinomial_resolution(
                        current_seq_len, num_segments, min_segment_size
                    )

                    loss_dict = multinomial_loss(
                        y_pred=pred_scaled,
                        y_true=targets_scaled,
                        mask=mask,
                        multinomial_resolution=multinomial_res,
                        positional_weight=positional_weight,
                        count_weight=count_weight,
                        channels_last=True,
                    )

                    res_loss = loss_dict["loss"] * weight
                    modality_loss = modality_loss + res_loss

                    # Accumulate for Pearson R
                    if compute_pearson:
                        pred_unscaled = predictions_unscaled[res]
                        batch_profile_r = profile_pearson_r(pred_unscaled, targets)
                        accumulated_profile_r[modality][res].append(batch_profile_r.float().cpu())
                        accumulated_pred_counts[modality][res].append(pred_unscaled.sum(dim=1).float().cpu())
                        accumulated_true_counts[modality][res].append(targets.sum(dim=1).float().cpu())

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            modality_loss_accum[modality] += modality_loss.item()

        total_loss_accum += loss.item()
        n_batches += 1

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    # Build metrics dict
    metrics: dict[str, Any] = {}
    for m, v in per_modality_loss.items():
        metrics[f"{m}_loss"] = v

    # Compute Pearson R
    if compute_pearson:
        for modality in heads:
            for res in resolution_weights.get(modality, {}).keys():
                if res in accumulated_profile_r[modality] and accumulated_profile_r[modality][res]:
                    all_profile_r = torch.cat(accumulated_profile_r[modality][res], dim=0)
                    if world_size > 1:
                        all_profile_r = gather_tensors(all_profile_r, world_size, device)
                    metrics[f"{modality}_{res}bp_profile_pearson_r_mean"] = all_profile_r.mean().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_std"] = all_profile_r.std().item()
                    metrics[f"{modality}_{res}bp_profile_pearson_r_values"] = all_profile_r.flatten().tolist()

                if res in accumulated_pred_counts[modality] and accumulated_pred_counts[modality][res]:
                    all_pred_counts = torch.cat(accumulated_pred_counts[modality][res], dim=0)
                    all_true_counts = torch.cat(accumulated_true_counts[modality][res], dim=0)
                    if world_size > 1:
                        all_pred_counts = gather_tensors(all_pred_counts, world_size, device)
                        all_true_counts = gather_tensors(all_true_counts, world_size, device)
                    if all_pred_counts.shape[0] > 1:
                        count_r = pearson_r(all_pred_counts, all_true_counts, dim=0)
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = count_r.mean().item()
                    else:
                        metrics[f"{modality}_{res}bp_count_pearson_r"] = float("nan")

    # Compute splice Pearson R for each variant
    if compute_pearson:
        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            is_junction = isinstance(head_module, SpliceSitesJunctionHead)

            for variant_name, variant_data in accumulated_splice[modality].items():
                if not variant_data["pred"]:
                    continue
                all_pred = torch.cat(variant_data["pred"], dim=0)
                all_true = torch.cat(variant_data["true"], dim=0)
                if world_size > 1:
                    all_pred = gather_tensors(all_pred, world_size, device)
                    all_true = gather_tensors(all_true, world_size, device)
                if all_pred.shape[0] > 1:
                    _pred_for_r = torch.log1p(all_pred) if is_junction else all_pred
                    _true_for_r = torch.log1p(all_true) if is_junction else all_true
                    r = pearson_r(_pred_for_r.unsqueeze(0), _true_for_r.unsqueeze(0), dim=1)
                    metric_key = f"{modality}_pearson_r"
                    if variant_name != "full":
                        metric_key += f"_{variant_name}"
                    metrics[metric_key] = r.item()

            # Add target nonzero fraction diagnostic for junction head
            if is_junction and "full" in accumulated_splice[modality]:
                full_true = accumulated_splice[modality]["full"]["true"]
                if full_true:
                    all_true = torch.cat(full_true, dim=0)
                    nonzero_frac = (all_true > 0).float().mean().item()
                    metrics[f"{modality}_target_nonzero_frac"] = nonzero_frac

    # Compute junction classification auPRC (aggregate flat keys, and per-sample if requested)
    if compute_pearson:
        import numpy as np
        from sklearn.metrics import average_precision_score

        # --- aggregate auprc_sample{s} / auprc_mean (backward compat, always computed) ---
        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            if not isinstance(head_module, SpliceSitesJunctionHead):
                continue
            if not accumulated_junc_cls[modality]:
                continue

            auprc_values = []
            for s, data in sorted(accumulated_junc_cls[modality].items()):
                all_scores = torch.cat(data["scores"]).numpy()
                all_labels = torch.cat(data["labels"]).numpy()
                n_pos = all_labels.sum()
                if n_pos == 0 or n_pos == len(all_labels):
                    continue
                ap = average_precision_score(all_labels, all_scores)
                metrics[f"{modality}_auprc_sample{s}"] = ap
                auprc_values.append(ap)

            if auprc_values:
                metrics[f"{modality}_auprc_mean"] = float(np.mean(auprc_values))

        # --- per-sample rows (only when compute_per_sample=True) ---
        if compute_per_sample:
            # Infer n_s from junction accumulation, fallback to usage
            n_s = None
            junc_modality = None
            for m in heads:
                if accumulated_junc_ps_pearson[m]:
                    n_s = len(accumulated_junc_ps_pearson[m])
                    junc_modality = m
                    break
            if n_s is None:
                for m in heads:
                    if accumulated_usage_ps_pearson[m]:
                        n_s = len(accumulated_usage_ps_pearson[m])
                        break

            if n_s is not None:
                per_sample_metrics = [{} for _ in range(n_s)]

                # Junction Pearson per sample (log1p)
                for modality in heads:
                    hm = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
                    if not isinstance(hm, SpliceSitesJunctionHead):
                        continue
                    for s in range(n_s):
                        if not accumulated_junc_ps_pearson[modality] or s >= len(accumulated_junc_ps_pearson[modality]):
                            continue
                        for variant, metric_key in [
                            ("full",    f"{modality}_pearson_r"),
                            ("nonzero", f"{modality}_pearson_r_nonzero"),
                        ]:
                            data = accumulated_junc_ps_pearson[modality][s][variant]
                            if data["pred"]:
                                all_pred = torch.log1p(torch.cat(data["pred"]))
                                all_true = torch.log1p(torch.cat(data["true"]))
                                if all_pred.shape[0] > 1:
                                    r = pearson_r(all_pred.unsqueeze(0), all_true.unsqueeze(0), dim=1)
                                    per_sample_metrics[s][metric_key] = r.item()

                # Junction AUPRC per sample (average pos-strand[s] and neg-strand[s+n_s])
                for modality in heads:
                    hm = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
                    if not isinstance(hm, SpliceSitesJunctionHead):
                        continue
                    if not accumulated_junc_cls[modality]:
                        continue
                    for s in range(n_s):
                        auprc_vals = []
                        for strand_key in (s, s + n_s):
                            if strand_key not in accumulated_junc_cls[modality]:
                                continue
                            data = accumulated_junc_cls[modality][strand_key]
                            all_scores = torch.cat(data["scores"]).numpy()
                            all_labels = torch.cat(data["labels"]).numpy()
                            n_pos = all_labels.sum()
                            if n_pos == 0 or n_pos == len(all_labels):
                                continue
                            auprc_vals.append(average_precision_score(all_labels, all_scores))
                        if auprc_vals:
                            per_sample_metrics[s][f"{modality}_auprc"] = float(np.mean(auprc_vals))

                # Usage Pearson per sample
                for modality in heads:
                    hm = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
                    if not isinstance(hm, SpliceSitesUsageHead):
                        continue
                    for s in range(n_s):
                        if not accumulated_usage_ps_pearson[modality] or s >= len(accumulated_usage_ps_pearson[modality]):
                            continue
                        data = accumulated_usage_ps_pearson[modality][s]
                        if data["pred"]:
                            all_pred = torch.cat(data["pred"])
                            all_true = torch.cat(data["true"])
                            if all_pred.shape[0] > 1:
                                r = pearson_r(all_pred.unsqueeze(0), all_true.unsqueeze(0), dim=1)
                                per_sample_metrics[s][f"{modality}_pearson_r"] = r.item()

                metrics["_per_sample"] = per_sample_metrics

    # Compute auPRC for classification head
    if compute_pearson:
        import torch.nn.functional as F
        from sklearn.metrics import average_precision_score

        _CLS_NAMES = ["donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "no_site"]

        for modality in heads:
            head_module = heads[modality].module if hasattr(heads[modality], "module") else heads[modality]
            if not isinstance(head_module, SpliceSitesClassificationHead):
                continue
            if not accumulated_cls[modality]["logits"]:
                continue

            all_logits = torch.cat(accumulated_cls[modality]["logits"], dim=0)  # (N, S, 5)
            all_true   = torch.cat(accumulated_cls[modality]["true"],   dim=0)  # (N, S, 5)
            probs = F.softmax(all_logits, dim=-1)

            # Keep only positions where any class is active
            active = all_true.any(dim=-1).reshape(-1)
            probs_flat = probs.reshape(-1, 5)[active].numpy()
            true_flat  = all_true.reshape(-1, 5)[active].numpy()

            if true_flat.shape[0] == 0:
                continue

            for i, cls_name in enumerate(_CLS_NAMES):
                if true_flat[:, i].sum() > 0:
                    ap = average_precision_score(true_flat[:, i], probs_flat[:, i])
                    metrics[f"{modality}_auprc_{cls_name}"] = ap

            # Macro average over splice-site classes only (exclude no_site = index 0)
            splice_cols = [i for i, n in enumerate(_CLS_NAMES) if n != "no_site"
                           and true_flat[:, i].sum() > 0]
            if splice_cols:
                macro_ap = average_precision_score(
                    true_flat[:, splice_cols], probs_flat[:, splice_cols], average="macro"
                )
                metrics[f"{modality}_auprc_macro"] = macro_ap

    return avg_loss, metrics


def train_epoch_sequence_parallel(
    model: nn.Module,
    heads: dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    modality_weights: dict[str, float],
    resolution_weights: dict[str, dict[int, float]],
    positional_weight: float,
    count_weight: float,
    sequence_parallel: Any,
    epoch: int,
    log_every: int,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    frozen_backbone: bool = False,
    num_segments: int = NUM_SEGMENTS,
    min_segment_size: int | None = None,
    train_sampler: DistributedSampler | None = None,
    rank: int = 0,
    world_size: int = 1,
    max_grad_norm: float = 1.0,
    profile_batches: int = 0,
    log_fn: Any | None = None,
    encoder_only: bool = False,
    save_every_steps: int | None = None,
    save_fn: Any | None = None,
    global_step_offset: int = 0,
    skip_batches: int = 0,
    save_state: dict | None = None,
    junction_top_k: int | None = None,
    junction_loss: str = "original",
) -> tuple[float, dict[str, float]]:
    """Train for one epoch with sequence parallelism.

    Splits the input sequence across GPUs instead of splitting the batch (DDP).
    This enables training on longer sequences by keeping per-GPU memory constant
    regardless of world size.

    Follows the same structure as train_epoch_multihead, replacing the backbone
    forward pass with sequence_parallel.forward() which performs the distributed
    encode step and returns per-rank local embeddings in NCL format.

    Args:
        model: AlphaGenome model (may be DDP-wrapped).
        heads: Dict mapping modality name to output head module.
        train_loader: Training data loader (yields sequences, modality_targets).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        modality_weights: Weight for each modality's loss.
        resolution_weights: Per-modality resolution weights dict.
        positional_weight: Weight for positional component of multinomial loss.
        count_weight: Weight for count component of multinomial loss.
        sequence_parallel: SequenceParallelism instance.
        epoch: Current epoch number.
        log_every: Log frequency in steps.
        use_amp: Whether to use automatic mixed precision.
        accumulation_steps: Number of batches to accumulate gradients over.
        frozen_backbone: If True, run backbone under torch.no_grad().
        num_segments: Number of segments for multinomial loss.
        min_segment_size: Minimum segment size for multinomial loss.
        train_sampler: DistributedSampler for shuffling across epochs.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        max_grad_norm: Maximum gradient norm for clipping.
        profile_batches: Number of batches to profile (0 = disabled).
        log_fn: Optional step logging function.
        encoder_only: Not used in SP mode (full backbone always runs).

    Returns:
        Tuple of (avg_total_loss, per_modality_train_loss).
    """
    from alphagenome_pytorch.extensions.finetuning.distributed import (
        is_main_process,
        reduce_tensor,
    )
    from alphagenome_pytorch.sequence_parallel import SequenceParallelism

    if not isinstance(sequence_parallel, SequenceParallelism):
        raise ValueError("sequence_parallel must be a SequenceParallelism instance")

    model_module = model.module if hasattr(model, "module") else model
    model_module.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    # Collect all needed resolutions across modalities (same as train_epoch_multihead)
    all_resolutions: set[int] = set()
    for modality in heads:
        all_resolutions.update(resolution_weights.get(modality, {}).keys())
    resolutions = tuple(all_resolutions)

    total_loss_accum = 0.0
    modality_loss_accum: dict[str, float] = {m: 0.0 for m in heads}
    n_batches = 0
    running_loss = 0.0
    accumulated_batches = 0
    opt_step = 0

    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [SP]")
    else:
        pbar = train_loader

    for batch_idx, (sequences, modality_targets) in enumerate(pbar):
        if batch_idx < skip_batches:
            continue

        sequences = sequences.to(device)
        organism_idx = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

        # Align sequence length to world_size * 128 for sequence parallelism.
        # This ensures each rank's base shard size is divisible by 128, so the
        # encoder's strided convolutions produce exact token counts.
        # this will be triggered when the world_size is not a multiple of 2
        if sequence_parallel is not None and world_size > 1:
            pad_multiple = world_size * 128 * 16  # lowres length must also be divisible by 16 for pair updates
            seq_len = sequences.shape[1]
            padded_len = ((seq_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            if padded_len > seq_len:
                n_pad = padded_len - seq_len
                if rank == 0:  # Print warning only once per batch
                    import warnings
                    warnings.warn(
                        f"Sequence length {seq_len} not divisible by {pad_multiple}. "
                        f"Padding to {padded_len} (+{n_pad} bp) for sequence parallelism.",
                        stacklevel=2
                    )
                sequences = torch.nn.functional.pad(sequences, (0, 0, 0, n_pad))  # pad (S, 4) on S dim

        # ===== BACKBONE: sequence-parallel forward =====
        # Returns embeddings_dict in NCL format - same as model.encode(channels_last=False)
        original_length = seq_len if padded_len > seq_len else None

        def _build_embeddings_dict(embeddings_1bp, embeddings_128bp, need_1bp_):
            d = {128: embeddings_128bp}
            if need_1bp_ and embeddings_1bp is not None:
                d[1] = embeddings_1bp
            return d

        if frozen_backbone:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = sequence_parallel.forward(
                        model=model_module,
                        sequence=sequences,
                        organism_index=organism_idx,
                        resolutions=resolutions,
                        original_length=original_length,
                    )
            embeddings_1bp = embeddings_1bp.detach() if embeddings_1bp is not None else None
            embeddings_128bp = embeddings_128bp.detach()
            embeddings_dict = _build_embeddings_dict(embeddings_1bp, embeddings_128bp, need_1bp)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                embeddings_1bp, embeddings_128bp, embeddings_pair, need_1bp = sequence_parallel.forward(
                    model=model_module,
                    sequence=sequences,
                    organism_index=organism_idx,
                    resolutions=resolutions,
                    original_length=original_length,
                )
            embeddings_dict = _build_embeddings_dict(embeddings_1bp, embeddings_128bp, need_1bp)

        # ===== HEADS + LOSS (mirrors train_epoch_multihead exactly) =====
        loss = torch.tensor(0.0, device=device)
        loss_components: dict[str, float] = {}

        for modality, head in heads.items():
            if modality not in modality_targets:
                continue

            modality_weight = modality_weights.get(modality, 1.0)
            res_weights = resolution_weights.get(modality, {})
            targets_dict = modality_targets[modality]

            head_module = head.module if hasattr(head, "module") else head

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # contact_maps head takes embeddings_pair directly; all other
                # GenomeTracksHead variants take embeddings_dict.
                # Sequence-parallel embeddings are in NCL (channels-first) format,
                # so splice head must be called with channels_last=False.
                if modality == "contact_maps":
                    predictions = head(
                        embeddings_pair, organism_idx, channels_last=True
                    )
                elif isinstance(head_module, SPLICE_HEAD_TYPES):
                    # Junction positions are not sharded — pass directly.
                    _positions = targets_dict.get("junction_positions")
                    if _positions is not None:
                        _positions = _positions.to(device)
                    _cls_head = heads.get("splice_site") if junction_top_k is not None else None
                    if _cls_head is not None:
                        _cls_head = _cls_head.module if hasattr(_cls_head, "module") else _cls_head
                    predictions = _call_splice_head(
                        head_module, embeddings_dict, organism_idx,
                        _positions, channels_last=False,
                        cls_head=_cls_head, junction_top_k=junction_top_k,
                    )
                else:
                    predictions = head(
                        embeddings_dict, organism_idx, return_scaled=True, channels_last=True
                    )

            modality_loss = torch.tensor(0.0, device=device)

            if isinstance(head_module, SPLICE_HEAD_TYPES):
                # Slice and move targets for sequence-parallel shard, then compute splice loss.
                # Preserve string keys (junction_positions, junction_matrix) without slicing.
                splice_targets_dict = {}
                for res, tgt in targets_dict.items():
                    tgt = tgt.to(device)
                    if isinstance(res, int):
                        # Slice sequence-length dimension for integer keys (resolutions)
                        full_len = tgt.shape[1]
                        local_len = full_len // world_size
                        t_start = rank * local_len
                        tgt = tgt[:, t_start:t_start + local_len, :]
                    splice_targets_dict[res] = tgt
                modality_loss, splice_components = _compute_splice_loss(
                    head_module, predictions, splice_targets_dict, device,
                    num_segments=num_segments,
                    junction_loss=junction_loss,
                )
                for k, v in splice_components.items():
                    loss_components[f"{modality}_{k}"] = v
            else:
                for res, weight in res_weights.items():
                    if res not in predictions or res not in targets_dict:
                        continue

                    pred = predictions[res]
                    targets = targets_dict[res].to(device)

                    # Slice targets to match local shard for this rank (sequence parallel).
                    # Targets are at their native resolution (e.g. S_full for 1bp,
                    # S_full//128 for 128bp), so a single split by world_size works
                    # regardless of resolution.
                    full_len = targets.shape[1]
                    local_len = full_len // world_size
                    t_start = rank * local_len
                    targets = targets[:, t_start:t_start + local_len, :]

                    targets = head_module.scale(
                        targets, organism_idx, resolution=res, channels_last=True
                    )
                    mask = torch.ones(pred.shape[0], 1, pred.shape[-1], dtype=torch.bool, device=device)

                    current_seq_len = pred.shape[-2]
                    multinomial_res = _compute_multinomial_resolution(
                        current_seq_len, num_segments, min_segment_size
                    )

                    loss_dict = multinomial_loss(
                        y_pred=pred,
                        y_true=targets,
                        mask=mask,
                        multinomial_resolution=multinomial_res,
                        positional_weight=positional_weight,
                        count_weight=count_weight,
                        channels_last=True,
                    )

                    res_loss = loss_dict["loss"] * weight
                    modality_loss = modality_loss + res_loss
                    loss_components[f"{modality}_loss_{res}bp"] = res_loss.item()

            weighted_modality_loss = modality_loss * modality_weight
            loss = loss + weighted_modality_loss
            loss_components[f"{modality}_loss"] = modality_loss.item()
            modality_loss_accum[modality] += modality_loss.item()

        # ===== BACKWARD + OPTIMIZER =====
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if is_accumulation_step or is_last_batch:
            # Sequence parallelism bypasses DDP's forward() so its allreduce hook
            # never fires. Manually allreduce gradients so all ranks apply the same
            # parameter update.
            if world_size > 1:
                trainable_params = []
                for head in heads.values():
                    trainable_params.extend([p for p in head.parameters() if p.requires_grad])
                trainable_params.extend([p for p in model.parameters() if p.requires_grad])
                for p in trainable_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            trainable_params = []
            for head in heads.values():
                trainable_params.extend([p for p in head.parameters() if p.requires_grad])
            trainable_params.extend([p for p in model.parameters() if p.requires_grad])

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

            if save_every_steps is not None and save_fn is not None:
                global_step = global_step_offset + opt_step
                if global_step % save_every_steps == 0:
                    if save_state is not None:
                        save_state["batch_idx"] = batch_idx + 1
                    save_fn()

        # ===== LOGGING =====
        raw_loss = loss.item()
        total_loss_accum += raw_loss
        n_batches += 1
        running_loss += raw_loss
        accumulated_batches += 1

        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank) and batch_idx % log_every == 0:
            avg_running_loss = running_loss / accumulated_batches
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{raw_loss:.4f}",
                    "run_loss": f"{avg_running_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            if log_fn is not None:
                log_fn({
                    "batch": batch_idx,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "running_loss": avg_running_loss,
                    "learning_rate": current_lr,
                    **loss_components,
                })

            running_loss = 0.0
            accumulated_batches = 0

    # Reduce across processes
    avg_loss = total_loss_accum / max(1, n_batches)
    per_modality_loss = {m: v / max(1, n_batches) for m, v in modality_loss_accum.items()}

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()

        for m in per_modality_loss:
            m_tensor = torch.tensor(per_modality_loss[m], device=device)
            m_tensor = reduce_tensor(m_tensor, world_size)
            per_modality_loss[m] = m_tensor.item()

    return avg_loss, per_modality_loss


__all__ = [
    "collate_genomic",
    "ModalityConfig",
    "MODALITY_CONFIGS",
    "create_lr_scheduler",
    "compute_finetuning_loss",
    "train_epoch",
    "validate",
    "save_checkpoint",
    # Enhanced versions with DDP support
    "ProfilingStats",
    "train_epoch_ddp",
    "validate_ddp",
    # Multi-head training
    "train_epoch_multihead",
    "validate_multihead",
    # Sequence parallel training
    "train_epoch_sequence_parallel",
]
