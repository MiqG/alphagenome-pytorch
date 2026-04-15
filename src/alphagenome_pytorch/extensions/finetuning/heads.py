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
from alphagenome_pytorch.losses import (
    cross_entropy_loss_from_logits,
    binary_crossentropy_from_logits,
    poisson_loss,
)


def _soft_clip_counts(counts: Tensor, clip: float = 10.0) -> Tensor:
    """Soft clip junction counts following JAX implementation.

    For counts > clip, applies: 2*sqrt(counts * clip) - clip
    This squashes large counts non-linearly to reduce outlier impact on Poisson loss.
    """
    return torch.where(
        counts > clip,
        2.0 * torch.sqrt(counts * clip) - clip,
        counts,
    )


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
    'splice': {'apply_squashing': False, 'default_resolutions': (1,)},
}


def create_finetuning_head(
    assay_type: Literal['rna_seq', 'atac', 'dnase', 'procap', 'cage', 'chip_tf', 'chip_histone', 'splice'],
    n_tracks: int,
    resolutions: list[int] | tuple[int, ...] | None = None,
    num_organisms: int = 1,
    track_means: torch.Tensor | None = None,
    init_scheme: Literal['truncated_normal', 'uniform'] = 'truncated_normal',
    encoder_only: bool = False,
) -> nn.Module:
    """Create a finetuning head configured for the given assay type.

    Args:
        assay_type: Type of assay ('splice' returns SpliceSitesFinetuningAdapter;
            all others return GenomeTracksHead).
        n_tracks: Number of output tracks. For 'splice', this includes the
            5 classification classes, so the actual number of usage tracks is n_tracks - 5.
        resolutions: Output resolutions. Valid values are 1 and/or 128.
            If None, uses default resolutions for the assay type.
        num_organisms: Number of organisms. Default: 1 for fine-tuning.
        track_means: Optional track means tensor for scaling.
        init_scheme: Weight initialization scheme ('truncated_normal' or 'uniform').
        encoder_only: If True, restrict to 128bp resolution only.

    Returns:
        For 'splice': SpliceSitesFinetuningAdapter.
        For others: GenomeTracksHead.

    Raises:
        ValueError: If an invalid assay type or resolution is provided.
    """
    if assay_type not in ASSAY_TYPES:
        valid_types = ', '.join(sorted(ASSAY_TYPES.keys()))
        raise ValueError(f"Invalid assay type '{assay_type}'. Must be one of: {valid_types}")

    # Handle splice modality: use adapter wrapping original heads
    if assay_type == 'splice':
        n_samples = n_tracks - SpliceSitesFinetuningAdapter.N_CLASSES
        return SpliceSitesFinetuningAdapter(in_channels=1536, n_samples=n_samples)

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


class SpliceSitesFinetuningAdapter(nn.Module):
    """Adapter wrapping SpliceSitesClassificationHead, SpliceSitesUsageHead,
    and SpliceSitesJunctionHead from alphagenome_pytorch.heads for finetuning.

    Provides the compute_loss() interface expected by the training loop.
    Uses the JAX-correct implementations from heads.py.

    Args:
        in_channels: Embedding dimension (default 1536).
        n_samples: Number of usage/junction tracks (bigwig samples).
        label_smoothing: Cross-entropy label smoothing (default 1e-7).
    """

    N_CLASSES = 5  # Donor+, Acceptor+, Donor−, Acceptor−, None

    def __init__(self, in_channels=1536, n_samples=1, label_smoothing=1e-7):
        super().__init__()
        self.n_samples = n_samples
        self.label_smoothing = label_smoothing
        # Wrap the original JAX-ported heads with num_organisms=1
        self.classification_head = SpliceSitesClassificationHead(
            in_channels=in_channels, num_organisms=1
        )
        self.usage_head = SpliceSitesUsageHead(
            in_channels=in_channels,
            num_output_tracks=n_samples,
            num_organisms=1,
        )
        self.junction_head = SpliceSitesJunctionHead(
            in_channels=in_channels,
            num_tissues=n_samples,
            num_organisms=1,
        )

    def forward(
        self,
        embeddings_dict: dict,
        organism_idx: Tensor,
        positions: Tensor | None = None,
        return_scaled: bool = True,
        channels_last: bool = True,
    ) -> dict:
        """Run all three sub-heads.

        Args:
            embeddings_dict: ``{1: (B, S, C)}`` if channels_last else ``{1: (B, C, S)}``.
            organism_idx: ``(B,)`` organism indices (all 0 for single-organism finetuning).
            positions: Optional ``(B, 4, K)`` int32 splice-site positions.
            return_scaled: Unused; kept for interface compatibility.
            channels_last: Format of input embeddings.

        Returns:
            Dict with key ``1`` → ``(B, S, 5 + n_samples)`` (cls + usage logits),
            and optionally ``"pos_counts"`` / ``"neg_counts"`` → ``(B, K, K, n_samples)``
            when positions are provided.
        """
        emb = embeddings_dict[1]  # (B, C, S) or (B, S, C)

        # Original heads expect NCL format
        if channels_last:
            emb_ncl = emb.transpose(1, 2)  # (B, S, C) → (B, C, S)
        else:
            emb_ncl = emb  # already NCL

        # Normalize organism_idx to shape (B,) all zeros
        if organism_idx.ndim > 1:
            organism_idx = organism_idx[:, 0]
        else:
            organism_idx = organism_idx
        organism_idx = torch.zeros_like(organism_idx)  # All are organism 0

        # Call original heads
        cls_out = self.classification_head(emb_ncl, organism_idx, channels_last=False)
        usage_out = self.usage_head(emb_ncl, organism_idx, channels_last=False)

        # Combine: cls logits (B, 5, S) + usage logits (B, n_samples, S) → (B, S, 5+n_samples)
        combined = torch.cat([cls_out["logits"], usage_out["logits"]], dim=1).transpose(1, 2)
        result = {1: combined}

        if positions is not None:
            junc_out = self.junction_head(
                emb_ncl, organism_idx, channels_last=False,
                splice_site_positions=positions,
            )
            # Original returns pred_counts (B, K, K, 2*n_samples) concatenated pos+neg
            pred_counts = junc_out["pred_counts"]
            result["pos_counts"] = pred_counts[..., :self.n_samples]
            result["neg_counts"] = pred_counts[..., self.n_samples:]

        return result

    def compute_loss(
        self,
        predictions: dict,
        targets_dict: dict,
        device: torch.device,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute classification + usage + (optional) junction loss.

        Junction loss is computed when ``targets_dict`` contains
        ``"junction_positions"`` and ``"junction_matrix"``.
        """
        pred = predictions[1]
        target = targets_dict[1].to(device)

        cls_pred = pred[..., :self.N_CLASSES]
        cls_target = target[..., :self.N_CLASSES]
        usage_pred = pred[..., self.N_CLASSES:]
        usage_target = target[..., self.N_CLASSES:]

        # Classification loss (positions where any of 5 classes is true)
        cls_mask = cls_target.any(dim=-1, keepdim=True).expand_as(cls_pred)
        eps = self.label_smoothing
        cls_target_smooth = (1.0 - eps) * cls_target.float() + eps / self.N_CLASSES
        cls_loss = cross_entropy_loss_from_logits(
            y_pred_logits=cls_pred,
            y_true=cls_target_smooth,
            mask=cls_mask,
            axis=-1,
        )

        # Usage loss
        usage_mask = (usage_target > 0).any(dim=-1, keepdim=True).expand_as(usage_pred)
        usage_loss = binary_crossentropy_from_logits(
            y_pred=usage_pred,
            y_true=usage_target.float(),
            mask=usage_mask,
        )

        components: dict[str, float] = {
            "cls_loss": cls_loss.item(),
            "usage_loss": usage_loss.item(),
        }
        total_loss = cls_loss + usage_loss

        # Junction loss — only when junction targets and predictions are present
        if "junction_matrix" in targets_dict and "pos_counts" in predictions:
            junc_matrix = targets_dict["junction_matrix"].to(device)  # (B, K, K, 2*n_s)
            pos_target = junc_matrix[..., : self.n_samples]
            neg_target = junc_matrix[..., self.n_samples :]
            positions = targets_dict["junction_positions"].to(device)  # (B, 4, K)

            def _strand_loss(pred_counts, target_counts, donor_pos, accept_pos):
                # pred_counts / target_counts: (B, D, A, n_samples)
                valid_d = (donor_pos >= 0).unsqueeze(-1)   # (B, D, 1)
                valid_a = (accept_pos >= 0).unsqueeze(-1)  # (B, A, 1)

                # Apply soft clipping to targets (JAX reference: line 1014-1017)
                clipped_target = _soft_clip_counts(target_counts)

                pred_donor_total = pred_counts.sum(dim=2)   # (B, D, n_samples)
                true_donor_total = clipped_target.sum(dim=2)
                d_loss = poisson_loss(
                    y_true=true_donor_total,
                    y_pred=pred_donor_total,
                    mask=valid_d.expand_as(pred_donor_total),
                )

                pred_accept_total = pred_counts.sum(dim=1)  # (B, A, n_samples)
                true_accept_total = clipped_target.sum(dim=1)
                a_loss = poisson_loss(
                    y_true=true_accept_total,
                    y_pred=pred_accept_total,
                    mask=valid_a.expand_as(pred_accept_total),
                )

                has_reads = clipped_target.sum(dim=(1, 2, 3)) > 0
                ratio_loss = torch.tensor(0.0, device=device)
                if has_reads.any():
                    p = pred_counts[has_reads]
                    t = clipped_target[has_reads]
                    t_sum_d = t.sum(dim=1, keepdim=True).clamp(min=1e-7)
                    ratio_loss = ratio_loss - (t / t_sum_d * p.log().clamp(min=-100)).sum() / has_reads.sum()
                    t_sum_a = t.sum(dim=2, keepdim=True).clamp(min=1e-7)
                    ratio_loss = ratio_loss - (t / t_sum_a * p.log().clamp(min=-100)).sum() / has_reads.sum()

                # Return weighted loss: Poisson @ 0.2×, ratio @ 1.0× (JAX: line 1048-1051)
                return 0.2 * (d_loss + a_loss) + ratio_loss

            pos_junc_loss = _strand_loss(
                predictions["pos_counts"], pos_target,
                positions[:, 0, :].long(), positions[:, 1, :].long(),
            )
            neg_junc_loss = _strand_loss(
                predictions["neg_counts"], neg_target,
                positions[:, 2, :].long(), positions[:, 3, :].long(),
            )
            junction_loss = pos_junc_loss + neg_junc_loss
            total_loss = total_loss + junction_loss
            components["junction_pos_loss"] = pos_junc_loss.item()
            components["junction_neg_loss"] = neg_junc_loss.item()

        return total_loss, components

    def scale(
        self,
        targets: Tensor,
        organism_idx: Tensor,
        resolution: int = 1,
        channels_last: bool = True,
    ) -> Tensor:
        """No-op: targets are already in the correct space."""
        return targets


__all__ = [
    'ASSAY_TYPES',
    'ENCODER_EMBEDDING_DIM',
    'create_finetuning_head',
    'SpliceSitesFinetuningAdapter',
]
