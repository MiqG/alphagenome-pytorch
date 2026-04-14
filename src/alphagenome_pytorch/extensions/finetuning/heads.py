"""Fine-tuning heads for AlphaGenome.

Provides a factory function to create GenomeTracksHead instances
configured for fine-tuning on specific assay types, plus a dedicated
SpliceSitesFinetuningHead for splice junction training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal

from alphagenome_pytorch.heads import GenomeTracksHead
from alphagenome_pytorch.losses import (
    cross_entropy_loss_from_logits,
    binary_crossentropy_from_logits,
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
) -> GenomeTracksHead:
    """Create a GenomeTracksHead configured for fine-tuning.

    Args:
        assay_type: Type of assay. Controls whether squashing is applied.
            'rna_seq' applies power law expansion.
            All other types do not apply squashing.
        n_tracks: Number of output tracks (e.g., number of cell types).
        resolutions: Output resolutions. Valid values are 1 and/or 128.
            If None, uses default resolutions for the assay type:
            - (1, 128) for atac, dnase, procap, cage, rna_seq
            - (128,) for chip_tf, chip_histone
        num_organisms: Number of organisms. Default: 1 for fine-tuning.
        track_means: Optional track means tensor for scaling.
            Shape: (num_organisms, n_tracks). Defaults to ones.
        init_scheme: Weight initialization scheme for head parameters.
            'truncated_normal' (default): Match JAX - truncated normal for
                weights (std=1/sqrt(fan_in)), zeros for biases.
            'uniform': Legacy PyTorch-style uniform initialization for both
                weights and biases.
        encoder_only: If True, create a head that accepts raw CNN encoder output
            (B, S//128, 1536) instead of full transformer embeddings. Automatically
            restricts resolutions to (128,). Use with ``model.forward(encoder_only=True)``
            for short-sequence fine-tuning (e.g. MPRA assays).

    Returns:
        Configured GenomeTracksHead instance.

    Example:
        >>> head = create_finetuning_head('atac', n_tracks=10)
        >>> head = create_finetuning_head('rna_seq', n_tracks=5, resolutions=(1, 128))
        >>> head = create_finetuning_head('chip_tf', n_tracks=100, resolutions=(128,))
        >>> # Encoder-only head for short sequences
        >>> head = create_finetuning_head('atac', n_tracks=10, encoder_only=True)

    Raises:
        ValueError: If an invalid assay type or resolution is provided.
    """
    if assay_type not in ASSAY_TYPES:
        valid_types = ', '.join(sorted(ASSAY_TYPES.keys()))
        raise ValueError(f"Invalid assay type '{assay_type}'. Must be one of: {valid_types}")

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


class SpliceSitesFinetuningHead(nn.Module):
    """Finetuning head for splice site classification and usage prediction.

    Combines two sub-heads applied to the 1 bp resolution embeddings:

    1. **Classification sub-head** — linear (in_channels → 5) producing logits
       for five classes: Donor+, Acceptor+, Donor−, Acceptor−, None.
       Trained with cross-entropy loss and label smoothing.

    2. **Usage sub-head** — linear (in_channels → n_samples) producing logits
       whose sigmoid gives per-sample splice-site usage in [0, 1].
       Trained with binary cross-entropy loss.

    The dataset (``SpliceJunctionDataset``) must return targets of shape
    ``(seq_len, 5 + n_samples)`` where the first 5 channels are the
    classification one-hot labels and the remaining channels are per-sample
    fractional usage values.

    Args:
        in_channels: Embedding dimension of 1 bp trunk features (default 1536).
        n_samples: Number of junction files / usage tracks.
        label_smoothing: Epsilon for classification label smoothing (default 1e-7).
    """

    N_CLASSES = 5  # Donor+, Acceptor+, Donor-, Acceptor-, None

    def __init__(
        self,
        in_channels: int = 1536,
        n_samples: int = 1,
        label_smoothing: float = 1e-7,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.label_smoothing = label_smoothing

        self.classification_head = nn.Conv1d(in_channels, self.N_CLASSES, kernel_size=1)
        self.usage_head = nn.Conv1d(in_channels, n_samples, kernel_size=1)

    def forward(
        self,
        embeddings_dict: dict,
        organism_idx: Tensor,
        return_scaled: bool = True,
        channels_last: bool = True,
    ) -> dict[int, Tensor]:
        """Run classification and usage sub-heads.

        Args:
            embeddings_dict: Must contain key ``1`` mapping to the 1 bp
                embeddings, shape ``(B, S, C)`` if channels_last else ``(B, C, S)``.
            organism_idx: ``(B,)`` organism indices (unused; kept for interface
                compatibility with other finetuning heads).
            return_scaled: Unused; kept for interface compatibility.
            channels_last: Input/output tensor format.

        Returns:
            ``{1: (B, S, 5 + n_samples)}`` — classification logits followed by
            usage logits along the last dimension.
        """
        emb = embeddings_dict[1]

        # Conv1d requires (B, C, S). Input may be channels-last (B, S, C) or
        # channels-first (B, C, S) depending on the calling context.
        if channels_last:
            emb = emb.transpose(1, 2)  # (B, S, C) -> (B, C, S)

        cls_logits = self.classification_head(emb)    # (B, 5, S)
        usage_logits = self.usage_head(emb)            # (B, n_samples, S)
        combined = torch.cat([cls_logits, usage_logits], dim=1)  # (B, 5+n_samples, S)

        # Always return channels-last (B, S, 5+n_samples) so compute_loss works uniformly.
        combined = combined.transpose(1, 2)

        return {1: combined}

    def compute_loss(
        self,
        predictions: dict[int, Tensor],
        targets_dict: dict[int, Tensor],
        device: torch.device,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined classification + usage loss.

        Args:
            predictions: ``{1: (B, S, 5 + n_samples)}`` from ``forward()``.
            targets_dict: ``{1: (B, S, 5 + n_samples)}`` from the dataset.
            device: Torch device.

        Returns:
            ``(total_loss, loss_components)`` where ``loss_components`` contains
            ``"cls_loss"`` and ``"usage_loss"`` as floats.
        """
        pred = predictions[1]                          # (B, S, 5 + n_samples)
        target = targets_dict[1].to(device)            # (B, S, 5 + n_samples)

        cls_pred = pred[..., :self.N_CLASSES]          # (B, S, 5)
        cls_target = target[..., :self.N_CLASSES]      # (B, S, 5) one-hot

        usage_pred = pred[..., self.N_CLASSES:]        # (B, S, n_samples)
        usage_target = target[..., self.N_CLASSES:]    # (B, S, n_samples)

        # -- Classification loss --
        # Mask: only positions that are an actual splice site (class != None)
        cls_mask = cls_target[..., :4].any(dim=-1, keepdim=True).expand_as(cls_pred)

        eps = self.label_smoothing
        cls_target_smooth = (1.0 - eps) * cls_target.float() + eps / self.N_CLASSES

        cls_loss = cross_entropy_loss_from_logits(
            y_pred_logits=cls_pred,
            y_true=cls_target_smooth,
            mask=cls_mask,
            axis=-1,
        )

        # -- Usage loss --
        # Mask: positions where any sample has non-zero usage
        usage_mask = (usage_target > 0).any(dim=-1, keepdim=True).expand_as(usage_pred)

        usage_loss = binary_crossentropy_from_logits(
            y_pred=usage_pred,
            y_true=usage_target.float(),
            mask=usage_mask,
        )

        total_loss = cls_loss + usage_loss
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "usage_loss": usage_loss.item(),
        }

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
    'SpliceSitesFinetuningHead',
]
