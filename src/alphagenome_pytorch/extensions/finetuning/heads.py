"""Fine-tuning heads for AlphaGenome.

Provides a factory function to create GenomeTracksHead instances
configured for fine-tuning on specific assay types, plus a dedicated
SpliceSitesFinetuningAdapter for splice junction training (wraps original heads).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal

from alphagenome_pytorch.heads import (
    GenomeTracksHead,
    SpliceSitesClassificationHead,
    SpliceSitesUsageHead,
    SpliceSitesJunctionHead,
)
from alphagenome_pytorch.losses import cross_entropy_loss, poisson_loss


def _soft_clip_counts(counts: Tensor, clip: float = 10.0) -> Tensor:
    return torch.where(counts > clip, 2.0 * torch.sqrt(counts * clip) - clip, counts)


def _compute_junction_strand_loss(pred_counts, target_counts, donor_pos, accept_pos, device):
    """Strand-specific junction loss matching JAX SpliceSitesJunctionHead.loss.

    loss = 0.2 * (CE(axis=donor) + CE(axis=acceptor)) + 0.04 * (Poisson(axis=donor) + Poisson(axis=acceptor))

    pairs_mask[b,d,a,s] = (donor_pos[b,d] >= 0) & (accept_pos[b,a] >= 0)
    """
    valid_d = (donor_pos >= 0).float()
    valid_a = (accept_pos >= 0).float()
    pairs_mask = torch.einsum('bd,ba->bda', valid_d, valid_a).bool()
    pairs_mask = pairs_mask.unsqueeze(-1).expand_as(pred_counts)

    if not pairs_mask.any():
        return torch.tensor(0.0, device=device, dtype=pred_counts.dtype)

    target = torch.where(pairs_mask, target_counts, torch.zeros_like(target_counts))
    pred   = torch.where(pairs_mask, pred_counts,   torch.zeros_like(pred_counts))

    donor_ratios_loss    = cross_entropy_loss(y_true=target, y_pred=pred, mask=pairs_mask, axis=1)
    acceptor_ratios_loss = cross_entropy_loss(y_true=target, y_pred=pred, mask=pairs_mask, axis=2)

    sum_pred_d = pred.sum(dim=1)
    sum_tgt_d  = _soft_clip_counts(target.sum(dim=1))
    sum_pred_a = pred.sum(dim=2)
    sum_tgt_a  = _soft_clip_counts(target.sum(dim=2))
    donor_total_loss  = poisson_loss(y_true=sum_tgt_d, y_pred=sum_pred_d, mask=pairs_mask.any(dim=1))
    accept_total_loss = poisson_loss(y_true=sum_tgt_a, y_pred=sum_pred_a, mask=pairs_mask.any(dim=2))

    return 0.2 * (donor_ratios_loss + acceptor_ratios_loss) + 0.04 * (donor_total_loss + accept_total_loss)


# All supported assay types and their squashing behavior
# Only RNA-seq uses squashing (power law expansion)
ASSAY_TYPES = {
    'rna_seq': {'apply_squashing': True, 'default_resolutions': (1, 128)},
    'atac': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'dnase': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'procap': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'cage': {'apply_squashing': False, 'default_resolutions': (1, 128)},
    'chip_tf': {'apply_squashing': False, 'default_resolutions': (128,)},
    'chip_histone': {'apply_squashing': False, 'default_resolutions': (128,)},
    'splice_site': {'apply_squashing': False, 'default_resolutions': (1,)},
    'splice_usage': {'apply_squashing': False, 'default_resolutions': (1,)},
    'splice_junctions': {'apply_squashing': False, 'default_resolutions': (1,)},
}


def create_finetuning_head(
    assay_type: Literal['rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone', 'splice_site', 'splice_usage', 'splice_junctions'],
    n_tracks: int,
    resolutions: list[int] | tuple[int, ...] | None = None,
    num_organisms: int = 1,
    track_means: torch.Tensor | None = None,
    init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    encoder_only: bool = False,
) -> nn.Module:
    """Create a finetuning head configured for the given assay type.

    Args:
        assay_type: Type of assay. Splice modalities ('splice_site', 'splice_usage',
            'splice_junctions') return original head instances from alphagenome_pytorch.heads.
            All others return GenomeTracksHead.
        n_tracks: Number of output tracks.
            - For 'splice_site': ignored (always 5 classes).
            - For 'splice_usage' or 'splice_junctions': number of junction samples.
            - For others: varies by assay type.
        resolutions: Output resolutions. Valid values are 1 and/or 128.
            If None, uses default resolutions for the assay type.
        num_organisms: Number of organisms. Default: 1 for fine-tuning.
        track_means: Optional track means tensor for scaling (ignored for splice modalities).
        init_scheme: Weight initialization scheme ('truncated_normal' or 'uniform').
        encoder_only: If True, restrict to 128bp resolution only.

    Returns:
        For splice_site: SpliceSitesClassificationHead.
        For splice_usage: SpliceSitesUsageHead.
        For splice_junctions: SpliceSitesJunctionHead.
        For others: GenomeTracksHead.

    Raises:
        ValueError: If an invalid assay type or resolution is provided.
    """
    if assay_type not in ASSAY_TYPES:
        valid_types = ', '.join(sorted(ASSAY_TYPES.keys()))
        raise ValueError(f"Invalid assay type '{assay_type}'. Must be one of: {valid_types}")

    # Handle splice modalities: return original heads directly (no adapter)
    if assay_type == 'splice_site':
        return SpliceSitesClassificationHead(in_channels=1536, num_organisms=1)
    if assay_type == 'splice_usage':
        return SpliceSitesUsageHead(in_channels=1536, num_output_tracks=n_tracks, num_organisms=1)
    if assay_type == 'splice_junctions':
        return SpliceSitesJunctionHead(in_channels=1536, num_tissues=n_tracks, num_organisms=1)

    assay_config = ASSAY_TYPES[assay_type]

    if encoder_only:
        # Encoder output is at 128bp resolution only; the decoder is not run.
        if resolutions is None:
            resolutions = (128,)
        for res in resolutions:
            if res != 128:
                raise ValueError(
                    f"encoder_only heads only support resolution 128 "
                    f"(got {res}). The CNN encoder produces features at 128bp; "
                    f"the decoder is not run in encoder-only mode."
                )
        return GenomeTracksHead(
            in_channels=1536,  # raw encoder output dim (ENCODER_EMBEDDING_DIM)
            num_tracks=n_tracks,
            resolutions=list(resolutions),
            num_organisms=num_organisms,
            apply_squashing=assay_config['apply_squashing'],
            track_means=track_means,
            init_scheme=init_scheme,
        )

    # Use default resolutions for assay type if not specified
    if resolutions is None:
        resolutions = assay_config['default_resolutions']

    valid_resolutions = {1, 128}
    for res in resolutions:
        if res not in valid_resolutions:
            raise ValueError(f"Invalid resolution {res}. Must be one of {valid_resolutions}")

    apply_squashing = assay_config['apply_squashing']

    return GenomeTracksHead(
        in_channels=None,
        num_tracks=n_tracks,
        resolutions=list(resolutions),
        num_organisms=num_organisms,
        apply_squashing=apply_squashing,
        track_means=track_means,
        init_scheme=init_scheme,
    )


# Embedding dimension of the raw CNN encoder output (before transformer/decoder).
ENCODER_EMBEDDING_DIM = 1536


__all__ = [
    'ASSAY_TYPES',
    'ENCODER_EMBEDDING_DIM',
    'create_finetuning_head',
    '_compute_junction_strand_loss',
]
