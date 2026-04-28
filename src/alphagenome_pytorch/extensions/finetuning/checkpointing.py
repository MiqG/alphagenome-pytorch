"""Checkpointing utilities for AlphaGenome fine-tuning.

Provides checkpoint save/load, discovery, and preemption handling for resumable training.
"""

from __future__ import annotations

import os
import signal
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from alphagenome_pytorch.extensions.finetuning.distributed import is_main_process


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    """Save a PyTorch object atomically.

    Uses temp file + rename to prevent corrupted files from crashes or
    power failures during save. Rename is atomic on POSIX systems.

    Args:
        obj: Object to save (checkpoint dict, model state, etc.).
        path: Destination path.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)  # Close the file descriptor, torch.save will open it
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)  # Atomic rename (works cross-platform)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    val_loss: float,
    track_names: list[str] | dict[str, list[str]],
    modality: str | list[str],
    resolutions: tuple[int, ...] | dict[str, tuple[int, ...]],
    scheduler: LRScheduler | None = None,
    best_val_loss: float | None = None,
    wandb_run_id: str | None = None,
    **extra_metadata: Any,
) -> None:
    """Save a training checkpoint atomically.

    The entire model state (trunk + all heads) is saved in model_state_dict.
    This works for all training modes (linear-probe, lora, full) and supports
    both single and multi-modality training.

    Uses atomic writes (temp file + rename) to prevent corrupted checkpoints
    from crashes or power failures during save.

    Args:
        path: Path to save checkpoint.
        epoch: Current epoch number.
        model: AlphaGenome model (trunk + heads).
        optimizer: Optimizer.
        val_loss: Validation loss at this checkpoint.
        track_names: Track names - either list (single modality) or dict (multi-modality).
        modality: Modality name(s) - either str or list of str.
        resolutions: Output resolutions - either tuple or dict mapping modality to resolutions.
        scheduler: Learning rate scheduler (optional).
        best_val_loss: Best validation loss seen so far (optional).
        wandb_run_id: W&B run ID for resume support (optional).
        **extra_metadata: Additional metadata to save.

    Example:
        >>> # Single modality
        >>> save_checkpoint(
        ...     path="checkpoint.pth",
        ...     epoch=5,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     val_loss=0.123,
        ...     track_names=["track1", "track2"],
        ...     modality="atac",
        ...     resolutions=(1, 128),
        ... )
        >>> # Multi-modality
        >>> save_checkpoint(
        ...     path="checkpoint.pth",
        ...     epoch=5,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     val_loss=0.123,
        ...     track_names={"atac": ["t1"], "rna_seq": ["t2"]},
        ...     modality=["atac", "rna_seq"],
        ...     resolutions={"atac": (1,), "rna_seq": (128,)},
        ... )
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "track_names": track_names,
        "modality": modality,
        "resolutions": resolutions,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    if wandb_run_id is not None:
        checkpoint["wandb_run_id"] = wandb_run_id

    checkpoint.update(extra_metadata)

    atomic_torch_save(checkpoint, path)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the most recent checkpoint in output_dir.

    Prefers ``checkpoint_preempt.pth`` (saved mid-epoch by the signal
    handler) over ``checkpoint_epoch*.pth`` when it is newer.

    Args:
        output_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.

    Example:
        >>> ckpt_path = find_latest_checkpoint(Path("output/run_001"))
        >>> if ckpt_path:
        ...     checkpoint = torch.load(ckpt_path)
    """
    preempt = output_dir / "checkpoint_preempt.pth"

    def _epoch_num(p: Path) -> int:
        return int(p.stem.replace("checkpoint_epoch", ""))

    epoch_ckpts = list(output_dir.glob("checkpoint_epoch*.pth"))
    epoch_ckpts.sort(key=_epoch_num)

    if not epoch_ckpts and not preempt.exists():
        return None

    # If preempt checkpoint exists, prefer it when it's newer than the
    # latest epoch checkpoint (it was saved *after* the last completed epoch).
    if preempt.exists():
        if not epoch_ckpts or preempt.stat().st_mtime >= epoch_ckpts[-1].stat().st_mtime:
            return preempt

    return epoch_ckpts[-1] if epoch_ckpts else None


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Loads the entire model state (trunk + all heads) from model_state_dict.
    This works for all training modes (linear-probe, lora, full) and supports
    both single and multi-modality training.

    Args:
        path: Path to checkpoint file.
        model: AlphaGenome model to load state into.
        optimizer: Optimizer to load state into.
        scheduler: Learning rate scheduler to load state into (optional).
        device: Device to map checkpoint to.

    Returns:
        Checkpoint dict with metadata (epoch, val_loss, best_val_loss, wandb_run_id, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load entire model state (trunk + heads)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class PreemptionHandler:
    """Handler for graceful preemption via signals.

    Listens for SIGUSR1 (SLURM preemption), SIGTERM (scancel / kill),
    and SIGINT (Ctrl+C). When any signal is received, sets a flag. The
    training loop should check `handler.preempted` and call
    `handler.save_and_exit()` to save a checkpoint before exiting gracefully.

    Note:
        The save function is NOT called inside the signal handler to avoid
        deadlocks. Signal handlers can interrupt the main thread at arbitrary
        points (including during I/O), so calling torch.save() there is unsafe.
        Instead, the training loop should check the flag and save explicitly.

    Attributes:
        preempted: Whether a preemption signal was received.

    Example:
        >>> handler = PreemptionHandler(
        ...     save_fn=lambda: save_checkpoint(...),
        ...     rank=0,
        ...     world_size=1,
        ... )
        >>> handler.register()
        >>> for epoch in range(100):
        ...     if handler.preempted:
        ...         handler.save_and_exit()
        ...         break
        ...     # ... training loop ...
    """

    _SIGNALS = (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)

    def __init__(
        self,
        save_fn: Callable[[], None] | None = None,
        rank: int = 0,
        world_size: int = 1,
        signal_num: int = signal.SIGUSR1,  # kept for backward compatibility, ignored
    ) -> None:
        """Initialize the preemption handler.

        Args:
            save_fn: Function to call to save checkpoint (called from training loop, not signal handler).
            rank: Process rank for distributed training.
            world_size: Total number of processes.
            signal_num: Ignored; all of SIGUSR1, SIGTERM, and SIGINT are always registered.
        """
        self.save_fn = save_fn
        self.rank = rank
        self.world_size = world_size
        self.preempted = False
        self._original_handlers: dict[int, Any] = {}
        self._save_thread: threading.Thread | None = None

    def _handler(self, signum: int, frame) -> None:
        """Signal handler that sets preempted flag and triggers an immediate save.

        The save is dispatched to a background thread so that file I/O does not
        run inside the signal handler (which would risk deadlocks on locks held
        by the interrupted thread).
        """
        self.preempted = True

        if is_main_process(self.rank):
            print(f"\n{'='*60}")
            print(f"SIGNAL {signum} received — will save checkpoint and exit.")
            print(f"{'='*60}")

        # Dispatch save to a background thread so we don't do I/O inside the
        # signal handler. Guard against double-saving if the signal fires twice.
        if self.save_fn is not None and (
            self._save_thread is None or not self._save_thread.is_alive()
        ):
            self._save_thread = threading.Thread(target=self.save_fn, daemon=True)
            self._save_thread.start()

    def save_and_exit(self) -> None:
        """Save checkpoint and synchronize processes.

        Call this from the training loop when `preempted` is True.
        If the signal handler already dispatched a save thread, this waits
        for it to finish rather than saving twice.
        """
        if is_main_process(self.rank):
            if self._save_thread is not None and self._save_thread.is_alive():
                print("Waiting for preemption checkpoint to finish saving...")
                self._save_thread.join()
                print("Preemption checkpoint saved.")
            elif self.save_fn is not None and self._save_thread is None:
                # Signal was not received (e.g. called directly); save now.
                print("Saving preemption checkpoint...")
                self.save_fn()
                print("Preemption checkpoint saved.")
            print("Training will exit.")

        # Synchronize all processes
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def register(self) -> None:
        """Register handlers for SIGUSR1, SIGTERM, and SIGINT."""
        for sig in self._SIGNALS:
            self._original_handlers[sig] = signal.signal(sig, self._handler)

    def unregister(self) -> None:
        """Restore original signal handlers."""
        for sig, original in self._original_handlers.items():
            signal.signal(sig, original)
        self._original_handlers.clear()


def setup_preemption_handler(
    save_fn: Callable[[], None] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> PreemptionHandler:
    """Set up and register a preemption handler.

    Convenience function that creates a PreemptionHandler and registers it.

    Args:
        save_fn: Function to call when signal is received (should save checkpoint).
        rank: Process rank for distributed training.
        world_size: Total number of processes.

    Returns:
        The registered PreemptionHandler instance.

    Example:
        >>> def save():
        ...     save_checkpoint(output_dir / "checkpoint_preempt.pth", ...)
        >>> handler = setup_preemption_handler(save, rank=0, world_size=1)
        >>> # In training loop:
        >>> if handler.preempted:
        ...     break
    """
    handler = PreemptionHandler(save_fn=save_fn, rank=rank, world_size=world_size)
    handler.register()
    return handler


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "PreemptionHandler",
    "setup_preemption_handler",
]
