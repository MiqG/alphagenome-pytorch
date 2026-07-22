"""Unit tests for the adapter bundle format and `agt adapters` CLI."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from alphagenome_pytorch.cli import adapters as adapters_cli
from alphagenome_pytorch.extensions.finetuning.adapters import LoRA
from alphagenome_pytorch.extensions.finetuning.checkpointing import (
    export_delta_weights,
    save_delta_checkpoint,
)
from alphagenome_pytorch.extensions.finetuning.transfer import TransferConfig
from alphagenome_pytorch.extensions.serving.bundle import (
    BundleError,
    BundlePaths,
    DEFAULT_ADAPTER_FILENAME,
    MANIFEST_FILENAME,
    Manifest,
    SCHEMA_VERSION,
    adapter_summary_kinds,
    render_model_card,
    validate_bundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lora_model() -> nn.Module:
    """Tiny standalone model with one LoRA-wrapped linear matching default lora_targets."""
    model = nn.Module()
    model.q_proj = LoRA(nn.Linear(16, 16), rank=4)
    model.heads = nn.ModuleDict()
    return model


def _lora_config() -> TransferConfig:
    return TransferConfig(
        mode="lora", lora_rank=4, lora_targets=["q_proj"], new_heads={}
    )


def _write_delta_pth(tmp_path: Path, *, base_hash: str | None = None) -> Path:
    model = _make_lora_model()
    cfg = _lora_config()
    p = tmp_path / "src.delta.pth"
    save_delta_checkpoint(
        p, model, cfg,
        base_model_hash=base_hash,
        epoch=3, val_loss=0.2,
        track_metadata=[{"head": "atac", "track_name": "demo", "is_padding": False}],
    )
    return p


def _write_delta_safetensors(tmp_path: Path) -> Path:
    model = _make_lora_model()
    cfg = _lora_config()
    p = tmp_path / "src.safetensors"
    export_delta_weights(model, cfg, p, format="safetensors")
    return p


def _make_export_args(**overrides) -> argparse.Namespace:
    base = dict(
        adapters_command="export",
        checkpoint=None,
        out=None,
        bundle_id="demo",
        label=None,
        base_model_id=None,
        base_weights=None,
        base_model_hash=None,
        genome=None,
        organism=None,
        modality=None,
        biosample=None,
        heads=None,
        license_name=None,
        no_readme=False,
        metrics=None,
        force=False,
        json_output=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


class TestManifest:
    def test_roundtrip(self, tmp_path: Path) -> None:
        m = Manifest(
            id="demo",
            base_model_hash="sha256:abc",
            label="Demo",
            base_model_id="org/repo",
            adapter_summary={"kind": "lora", "lora_rank": 4},
            heads=["atac"],
        )
        m.dump(tmp_path)
        loaded = Manifest.load(tmp_path)
        assert loaded.id == m.id
        assert loaded.base_model_hash == m.base_model_hash
        assert loaded.label == "Demo"
        assert loaded.adapter_summary == {"kind": "lora", "lora_rank": 4}
        assert loaded.heads == ["atac"]
        assert loaded.schema_version == SCHEMA_VERSION

    def test_dump_writes_alphagenome_adapter_json(self, tmp_path: Path) -> None:
        Manifest(id="x", base_model_hash="sha256:abc").dump(tmp_path)
        assert (tmp_path / MANIFEST_FILENAME).is_file()

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(BundleError, match="id"):
            Manifest.from_dict({"schema_version": 1, "base_model_hash": "sha256:x"})
        with pytest.raises(BundleError, match="base_model_hash"):
            Manifest.from_dict({"schema_version": 1, "id": "x"})

    def test_missing_schema_version_raises(self) -> None:
        with pytest.raises(BundleError, match="schema_version"):
            Manifest.from_dict({"id": "x", "base_model_hash": "sha256:x"})

    def test_too_new_schema_version_raises(self) -> None:
        with pytest.raises(BundleError, match="newer than"):
            Manifest.from_dict({
                "schema_version": SCHEMA_VERSION + 1,
                "id": "x",
                "base_model_hash": "sha256:x",
            })

    def test_unknown_fields_ignored_for_forward_compat(self) -> None:
        m = Manifest.from_dict({
            "schema_version": SCHEMA_VERSION,
            "id": "x",
            "base_model_hash": "sha256:x",
            "future_field": "ignore-me",
        })
        assert m.id == "x"
        assert not hasattr(m, "future_field")

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="not found"):
            Manifest.load(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / MANIFEST_FILENAME
        bad.write_text("{not json")
        with pytest.raises(BundleError, match="not valid JSON"):
            Manifest.load(bad)


# ---------------------------------------------------------------------------
# BundlePaths
# ---------------------------------------------------------------------------


class TestBundlePaths:
    def test_resolve_happy_path(self, tmp_path: Path) -> None:
        Manifest(id="x", base_model_hash="sha256:x").dump(tmp_path)
        (tmp_path / DEFAULT_ADAPTER_FILENAME).write_bytes(b"")
        paths = BundlePaths.resolve(tmp_path)
        assert paths.bundle_dir == tmp_path
        assert paths.manifest.name == MANIFEST_FILENAME
        assert paths.adapter_safetensors.name == DEFAULT_ADAPTER_FILENAME
        assert paths.readme is None
        assert paths.metrics is None

    def test_resolve_picks_up_optional_files(self, tmp_path: Path) -> None:
        Manifest(id="x", base_model_hash="sha256:x").dump(tmp_path)
        (tmp_path / DEFAULT_ADAPTER_FILENAME).write_bytes(b"")
        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "metrics.json").write_text("{}")
        paths = BundlePaths.resolve(tmp_path)
        assert paths.readme is not None and paths.readme.name == "README.md"
        assert paths.metrics is not None and paths.metrics.name == "metrics.json"

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="not found"):
            BundlePaths.resolve(tmp_path / "missing")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match=MANIFEST_FILENAME):
            BundlePaths.resolve(tmp_path)

    def test_missing_adapter_file(self, tmp_path: Path) -> None:
        Manifest(id="x", base_model_hash="sha256:x").dump(tmp_path)
        with pytest.raises(BundleError, match="missing"):
            BundlePaths.resolve(tmp_path)

    def test_custom_adapter_filename(self, tmp_path: Path) -> None:
        m = Manifest(
            id="x", base_model_hash="sha256:x",
            adapter_filename="custom.safetensors",
        )
        m.dump(tmp_path)
        (tmp_path / "custom.safetensors").write_bytes(b"")
        paths = BundlePaths.resolve(tmp_path)
        assert paths.adapter_safetensors.name == "custom.safetensors"


# ---------------------------------------------------------------------------
# validate_bundle
# ---------------------------------------------------------------------------


class TestValidateBundle:
    def test_ok_without_base_model(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        report = validate_bundle(bundle)
        assert report.ok, report.errors
        assert report.manifest is not None
        assert report.manifest.id == "demo"

    def test_kind_mismatch_warns(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        # Mutate manifest to declare a kind not present in the transfer_config
        manifest_path = bundle / MANIFEST_FILENAME
        m = json.loads(manifest_path.read_text())
        m["adapter_summary"]["kinds"] = ["houlsby"]
        manifest_path.write_text(json.dumps(m))
        report = validate_bundle(bundle)
        assert report.ok  # still ok; mismatch is a warning
        assert any("houlsby" in w for w in report.warnings)

    def test_missing_adapter_file_errors(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        (bundle / DEFAULT_ADAPTER_FILENAME).unlink()
        report = validate_bundle(bundle)
        assert not report.ok


def _export_via_cli(tmp_path: Path, out: Path, *, base_model_hash: str) -> Path:
    """Helper: build a delta.pth then run the export CLI to produce a bundle."""
    src = _write_delta_pth(tmp_path, base_hash=base_model_hash)
    args = _make_export_args(
        checkpoint=str(src),
        out=str(out),
        bundle_id="demo",
        base_model_hash=base_model_hash,
    )
    rc = adapters_cli.run(args)
    assert rc == 0
    return out


# ---------------------------------------------------------------------------
# CLI: export
# ---------------------------------------------------------------------------


class TestExportCli:
    def test_export_from_delta_pth(self, tmp_path: Path) -> None:
        src = _write_delta_pth(tmp_path, base_hash="sha256:from-ckpt")
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src),
            out=str(out),
            bundle_id="wtc11-atac-lora",
            label="WTC11 ATAC LoRA",
            base_model_id="org/alphagenome",
            organism="human",
            modality="atac",
            biosample="WTC11",
        ))
        assert rc == 0

        paths = BundlePaths.resolve(out)
        assert paths.adapter_safetensors.is_file()
        assert paths.readme is not None
        manifest = Manifest.load(paths.manifest)
        assert manifest.id == "wtc11-atac-lora"
        assert manifest.label == "WTC11 ATAC LoRA"
        # base hash is taken from the source delta checkpoint
        assert manifest.base_model_hash == "sha256:from-ckpt"
        assert manifest.adapter_summary.get("kinds") == ["lora"]
        assert manifest.adapter_summary.get("lora_rank") == 4
        assert "locon_rank" not in manifest.adapter_summary
        assert manifest.organism == "human"
        # provenance pulled from the source checkpoint metadata
        assert manifest.provenance.get("epoch") == 3
        assert manifest.provenance.get("val_loss") == pytest.approx(0.2)

    def test_export_from_safetensors_requires_hash_source(
        self, tmp_path: Path
    ) -> None:
        src = _write_delta_safetensors(tmp_path)
        with pytest.raises(ValueError, match="base_model_hash"):
            adapters_cli.run(_make_export_args(
                checkpoint=str(src),
                out=str(tmp_path / "bundle"),
                bundle_id="x",
            ))

    def test_export_from_safetensors_with_explicit_hash(
        self, tmp_path: Path
    ) -> None:
        src = _write_delta_safetensors(tmp_path)
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src),
            out=str(out),
            bundle_id="x",
            base_model_hash="sha256:explicit",
        ))
        assert rc == 0
        manifest = Manifest.load(out)
        assert manifest.base_model_hash == "sha256:explicit"
        # safetensors source preserves byte-for-byte → adapter file matches
        assert (out / DEFAULT_ADAPTER_FILENAME).read_bytes() == src.read_bytes()

    def test_export_refuses_nonempty_dir_without_force(
        self, tmp_path: Path
    ) -> None:
        src = _write_delta_pth(tmp_path, base_hash="sha256:abc")
        out = tmp_path / "bundle"
        out.mkdir()
        (out / "leftover").write_text("x")
        with pytest.raises(FileExistsError):
            adapters_cli.run(_make_export_args(
                checkpoint=str(src),
                out=str(out),
                bundle_id="x",
                base_model_hash="sha256:abc",
            ))

    def test_export_force_overwrites(self, tmp_path: Path) -> None:
        src = _write_delta_pth(tmp_path, base_hash="sha256:abc")
        out = tmp_path / "bundle"
        out.mkdir()
        (out / "leftover").write_text("x")
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src),
            out=str(out),
            bundle_id="x",
            base_model_hash="sha256:abc",
            force=True,
        ))
        assert rc == 0
        assert not (out / "leftover").exists()

    def test_export_no_readme_skips_card(self, tmp_path: Path) -> None:
        src = _write_delta_pth(tmp_path, base_hash="sha256:abc")
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src),
            out=str(out),
            bundle_id="x",
            base_model_hash="sha256:abc",
            no_readme=True,
        ))
        assert rc == 0
        assert not (out / "README.md").exists()

    def test_export_metrics_file_is_copied(self, tmp_path: Path) -> None:
        src = _write_delta_pth(tmp_path, base_hash="sha256:abc")
        metrics = tmp_path / "metrics.json"
        metrics.write_text('{"pearson_r": 0.42}')
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src),
            out=str(out),
            bundle_id="x",
            base_model_hash="sha256:abc",
            metrics=str(metrics),
        ))
        assert rc == 0
        assert (out / "metrics.json").read_text() == '{"pearson_r": 0.42}'
        manifest = Manifest.load(out)
        assert manifest.metrics_path == "metrics.json"

    def test_export_unknown_source_format_raises(self, tmp_path: Path) -> None:
        weird = tmp_path / "random.pth"
        torch.save({"hello": "world"}, weird)
        with pytest.raises(ValueError, match="neither"):
            adapters_cli.run(_make_export_args(
                checkpoint=str(weird),
                out=str(tmp_path / "bundle"),
                bundle_id="x",
                base_model_hash="sha256:abc",
            ))


# ---------------------------------------------------------------------------
# CLI: inspect
# ---------------------------------------------------------------------------


class TestInspectCli:
    def test_inspect_text(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.cli._output import emit_text as orig_emit_text

        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        buf = io.StringIO()
        with mock.patch.object(
            adapters_cli, "emit_text",
            side_effect=lambda text, **kw: orig_emit_text(text, file=buf),
        ):
            rc = adapters_cli.run(argparse.Namespace(
                adapters_command="inspect",
                bundle_dir=str(bundle),
                json_output=False,
            ))
        assert rc == 0
        out = buf.getvalue()
        assert "demo" in out
        assert "sha256:demo" in out
        assert "lora" in out

    def test_inspect_json(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.cli._output import emit_json as orig_emit_json

        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        buf = io.StringIO()
        with mock.patch.object(
            adapters_cli, "emit_json",
            side_effect=lambda data, **kw: orig_emit_json(data, file=buf),
        ):
            rc = adapters_cli.run(argparse.Namespace(
                adapters_command="inspect",
                bundle_dir=str(bundle),
                json_output=True,
            ))
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert payload["manifest"]["id"] == "demo"
        assert payload["manifest"]["base_model_hash"] == "sha256:demo"
        assert payload["files"]["adapter"] == DEFAULT_ADAPTER_FILENAME


# ---------------------------------------------------------------------------
# CLI: validate
# ---------------------------------------------------------------------------


class TestValidateCli:
    def test_validate_clean_bundle(self, tmp_path: Path) -> None:
        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        rc = adapters_cli.run(argparse.Namespace(
            adapters_command="validate",
            bundle_dir=str(bundle),
            base_weights=None,
            json_output=False,
        ))
        assert rc == 0

    def test_validate_broken_bundle_returns_nonzero(
        self, tmp_path: Path
    ) -> None:
        bundle = tmp_path / "b"
        _export_via_cli(tmp_path, bundle, base_model_hash="sha256:demo")
        (bundle / DEFAULT_ADAPTER_FILENAME).unlink()
        rc = adapters_cli.run(argparse.Namespace(
            adapters_command="validate",
            bundle_dir=str(bundle),
            base_weights=None,
            json_output=False,
        ))
        assert rc == 1


# ---------------------------------------------------------------------------
# Model card rendering
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_render_includes_key_fields(self) -> None:
        m = Manifest(
            id="wtc11-atac-lora",
            base_model_hash="sha256:abc",
            label="WTC11 ATAC LoRA",
            base_model_id="org/alphagenome",
            adapter_summary={"kind": "lora"},
            organism="human",
            modality="atac",
            biosample="WTC11",
            license="apache-2.0",
        )
        card = render_model_card(m)
        assert "library_name: alphagenome-pytorch" in card
        assert "base_model: org/alphagenome" in card
        assert "base_model_relation: adapter" in card
        assert "WTC11 ATAC LoRA" in card
        assert "sha256:abc" in card
        assert "atac" in card


# ---------------------------------------------------------------------------
# Organism provenance survives export (regression: mouse must not fall back to
# human), and catalog adapters land on the base device.
# ---------------------------------------------------------------------------


class TestExportOrganismProvenance:
    """`agt adapters export` must carry organism provenance through so a served
    mouse fine-tune resolves to mouse, not human."""

    def _write_mouse_delta(self, tmp_path: Path) -> Path:
        model = _make_lora_model()
        cfg = _lora_config()
        p = tmp_path / "mouse.delta.pth"
        save_delta_checkpoint(
            p, model, cfg,
            base_model_hash="sha256:mouse-base",
            epoch=1, val_loss=0.1,
            organism="mouse",
            organism_indices=[1],
            track_metadata=[
                {"head": "atac", "track_name": "m", "organism": 1, "is_padding": False}
            ],
        )
        return p

    def test_export_reload_keeps_default_organism_mouse(self, tmp_path: Path) -> None:
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            _read_delta_export_header,
            resolve_finetuned_organism,
        )

        src = self._write_mouse_delta(tmp_path)
        out = tmp_path / "bundle"
        # Export WITHOUT --organism: provenance must come from the checkpoint.
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src), out=str(out), bundle_id="mouse-atac",
        ))
        assert rc == 0

        paths = BundlePaths.resolve(out)
        header = _read_delta_export_header(paths.adapter_safetensors)
        assert header.get("organism") == "mouse"
        assert header.get("organism_indices") == [1]

        # Reload → resolved default organism is still mouse (index 1), not human.
        ctx = resolve_finetuned_organism(
            organism_indices=header.get("organism_indices"),
            checkpoint_organism=header.get("organism"),
            track_metadata=header.get("track_metadata"),
            num_organisms=2,
        )
        assert ctx.default_organism_index == 1

    def test_organism_flag_writes_embedded_metadata(self, tmp_path: Path) -> None:
        """--organism updates the bundle's EMBEDDED metadata (not just the
        manifest) so serving honors it even for checkpoints lacking organism."""
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            _read_delta_export_header,
            resolve_finetuned_organism,
        )

        src = _write_delta_pth(tmp_path)  # no organism embedded
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src), out=str(out), bundle_id="x", organism="mouse",
        ))
        assert rc == 0

        header = _read_delta_export_header(BundlePaths.resolve(out).adapter_safetensors)
        assert header.get("organism") == "mouse"
        assert header.get("organism_indices") == [1]
        ctx = resolve_finetuned_organism(
            organism_indices=header.get("organism_indices"),
            checkpoint_organism=header.get("organism"),
            track_metadata=header.get("track_metadata"),
            num_organisms=2,
        )
        assert ctx.default_organism_index == 1  # serves mouse, not human

    def test_organism_flag_overrides_embedded_organism(self, tmp_path: Path) -> None:
        """--organism overrides an organism the source checkpoint already embeds."""
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            _read_delta_export_header,
        )

        src = self._write_mouse_delta(tmp_path)  # embeds organism=mouse, [1]
        out = tmp_path / "bundle"
        rc = adapters_cli.run(_make_export_args(
            checkpoint=str(src), out=str(out), bundle_id="x", organism="human",
        ))
        assert rc == 0

        header = _read_delta_export_header(BundlePaths.resolve(out).adapter_safetensors)
        assert header.get("organism") == "human"
        assert header.get("organism_indices") == [0]


class TestCatalogAdapterDevicePlacement:
    """build_adapter_entry must leave the captured adapter/head modules on the
    base model's device — prepare_for_transfer can create some on CPU, and
    detached entries never ride along in a later base_model.to(device)."""

    def _base(self, device: str) -> nn.Module:
        base = nn.Module()
        base.q_proj = nn.Linear(16, 16)
        base.heads = nn.ModuleDict()
        return base.to(device)

    def _build_entry(self, tmp_path: Path, device: str):
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            compute_base_model_hash,
        )
        from alphagenome_pytorch.extensions.serving.router import (
            build_adapter_entry,
        )

        base = self._base(device)
        delta = tmp_path / "src.delta.pth"
        save_delta_checkpoint(
            delta, _make_lora_model(), _lora_config(),
            base_model_hash=compute_base_model_hash(base),
            epoch=1, val_loss=0.1,
        )
        out = tmp_path / "bundle"
        assert adapters_cli.run(_make_export_args(
            checkpoint=str(delta), out=str(out), bundle_id="d",
        )) == 0
        paths = BundlePaths.resolve(out)
        return build_adapter_entry(
            base_model=base, bundle_paths=paths, manifest=Manifest.load(paths.manifest),
        )

    def _assert_all_on(self, entry, dev_type: str) -> None:
        moved = 0
        for att in entry.adapter_attachments:
            for p in att.wrapper.parameters():
                assert p.device.type == dev_type
                moved += 1
        for hmod in entry.head_modules.values():
            for p in hmod.parameters():
                assert p.device.type == dev_type
        assert moved > 0  # the LoRA wrapper contributed params

    def test_cpu_placement(self, tmp_path: Path) -> None:
        entry = self._build_entry(tmp_path, "cpu")
        self._assert_all_on(entry, "cpu")

    def test_carries_per_entry_serving_fields(self, tmp_path: Path) -> None:
        """The catalog builder relies on build_adapter_entry storing each
        bundle's own scorer / metadata catalog / track names / organism."""
        from alphagenome_pytorch.extensions.finetuning.checkpointing import (
            compute_base_model_hash,
        )
        from alphagenome_pytorch.extensions.serving.router import (
            build_adapter_entry,
        )

        base = self._base("cpu")
        delta = tmp_path / "src.delta.pth"
        save_delta_checkpoint(
            delta, _make_lora_model(), _lora_config(),
            base_model_hash=compute_base_model_hash(base),
        )
        out = tmp_path / "bundle"
        assert adapters_cli.run(_make_export_args(
            checkpoint=str(delta), out=str(out), bundle_id="d",
        )) == 0
        paths = BundlePaths.resolve(out)

        sentinel_scorer = object()
        sentinel_catalog = object()
        entry = build_adapter_entry(
            base_model=base, bundle_paths=paths,
            manifest=Manifest.load(paths.manifest),
            metadata_catalog=sentinel_catalog,
            track_names={"atac": ["t0"]},
            scorer=sentinel_scorer,
            default_organism=1,
        )
        assert entry.scorer is sentinel_scorer
        assert entry.metadata_catalog is sentinel_catalog
        assert entry.track_names == {"atac": ["t0"]}
        assert entry.default_organism == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_placement(self, tmp_path: Path) -> None:
        entry = self._build_entry(tmp_path, "cuda")
        self._assert_all_on(entry, "cuda")


# ---------------------------------------------------------------------------
# adapter_summary generation and kind resolution
# ---------------------------------------------------------------------------


def _summary(config: TransferConfig) -> dict:
    from dataclasses import asdict

    return adapters_cli._summary_from_header(
        {"transfer_config": asdict(config)}
    )["adapter_summary"]


class TestSummaryFromHeader:
    def test_pure_locon_reports_locon_not_lora(self) -> None:
        # transfer_config is a full asdict, so lora_rank/lora_alpha are always
        # present as defaults; the summary must not leak them for a Locon run.
        summary = _summary(
            TransferConfig(mode="locon", locon_rank=4, locon_alpha=1)
        )
        assert summary["kinds"] == ["locon"]
        assert summary["locon_rank"] == 4
        assert summary["locon_alpha"] == 1
        assert "lora_rank" not in summary
        assert "lora_alpha" not in summary

    def test_pure_lora(self) -> None:
        summary = _summary(TransferConfig(mode="lora", lora_rank=8, lora_alpha=16))
        assert summary["kinds"] == ["lora"]
        assert summary["lora_rank"] == 8
        assert summary["lora_alpha"] == 16
        assert "locon_rank" not in summary

    def test_combined_lora_locon_reports_both(self) -> None:
        summary = _summary(
            TransferConfig(
                mode=["lora", "locon"],
                lora_rank=8,
                lora_alpha=16,
                locon_rank=4,
                locon_alpha=1,
            )
        )
        assert summary["kinds"] == ["lora", "locon"]
        assert summary["lora_rank"] == 8
        assert summary["lora_alpha"] == 16
        assert summary["locon_rank"] == 4
        assert summary["locon_alpha"] == 1
        # No misleading scalar kind that would collapse to just "lora".
        assert "kind" not in summary

    def test_linear_mode_has_no_adapter_hyperparams(self) -> None:
        summary = _summary(TransferConfig(mode="linear"))
        assert summary["kinds"] == ["linear"]
        assert "lora_rank" not in summary
        assert "locon_rank" not in summary


class TestAdapterSummaryKinds:
    def test_prefers_kinds_list(self) -> None:
        assert adapter_summary_kinds({"kinds": ["lora", "locon"]}) == ["lora", "locon"]

    def test_falls_back_to_legacy_scalar_kind(self) -> None:
        assert adapter_summary_kinds({"kind": "lora"}) == ["lora"]

    def test_kinds_wins_over_legacy_kind(self) -> None:
        assert adapter_summary_kinds({"kind": "lora", "kinds": ["locon"]}) == ["locon"]

    def test_empty_or_missing(self) -> None:
        assert adapter_summary_kinds(None) == []
        assert adapter_summary_kinds({}) == []
        assert adapter_summary_kinds({"kinds": []}) == []
