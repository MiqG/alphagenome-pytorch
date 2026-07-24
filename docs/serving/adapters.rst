Adapter Bundles and Catalog Serving
====================================

Finetuned adapters can be packaged as **bundles** — a directory containing the
exported delta-weights file plus a sidecar manifest — and served either alone
(singleton mode) or alongside other adapters over a shared base trunk
(catalog mode).

Bundle Layout
-------------

A bundle is a directory:

.. code-block:: text

   my-bundle/
     adapter.safetensors          # output of export_delta_weights
     alphagenome_adapter.json     # sidecar manifest (this format)
     README.md                    # generated model card
     metrics.json                 # optional, copy of evaluation output

The ``adapter.safetensors`` file embeds the ``transfer_config`` (and optional
``track_names`` / ``track_metadata``) used to reconstruct the finetuned model
at load time. The manifest also gates base-model compatibility:

- ``base_model_hash`` hashes trunk key names, shapes, and dtypes. It detects an
  incompatible architecture but intentionally does not distinguish folds.
- ``base_model_weights_hash`` hashes the canonical trunk tensor contents. It
  identifies the exact base checkpoint/fold and is independent of whether the
  same tensors were serialized as ``.pth`` or safetensors.

Both are checked when exact provenance is available. Older bundles without
``base_model_weights_hash`` remain loadable with a warning and receive only the
structural compatibility check.

Manifest Schema (v1)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "schema_version": 1,
     "id": "wtc11-atac-lora",
     "label": "WTC11 ATAC LoRA",
     "base_model_id": "your-org/alphagenome",
     "base_model_hash": "sha256:abc...",
     "base_model_weights_hash": "sha256-tensors-v1:def...",
     "alphagenome_pytorch_version": "x.y.z",
     "adapter_summary": {"kinds": ["lora"], "lora_rank": 8, "lora_alpha": 16, "lora_targets": ["q_proj", "v_proj"]},
     "genome": "hg38",
     "organism": "human",
     "modalities": ["atac"],
     "biosample": "WTC11",
     "heads": ["atac_wtc11"],
     "num_tracks": 4,
     "metrics_path": "metrics.json",
     "license": "apache-2.0",
     "provenance": {"created_at": "...", "git_commit": "..."},
     "adapter_filename": "adapter.safetensors"
   }

CLI Workflow
------------

Build a bundle from an existing delta checkpoint:

.. code-block:: bash

   agt adapters export \
     --checkpoint runs/wtc11-atac/best.delta.pth \
     --base-model your-org/alphagenome \
     --id wtc11-atac-lora \
     --label "WTC11 ATAC LoRA" \
     --genome hg38 \
     --organism human \
     --modality atac \
     --biosample WTC11 \
     --out dist/wtc11-atac-lora

New fine-tuning runs record the exact base weights hash automatically. When
exporting an older checkpoint, pass ``--base-weights`` to add it:

.. code-block:: bash

   agt adapters export \
     --checkpoint old-run/best.delta.pth \
     --base-weights fold_3.safetensors \
     --base-model your-org/alphagenome \
     --id wtc11-atac-lora \
     --out dist/wtc11-atac-lora

``--organism`` is an **override**: it is written into the bundle's embedded
delta metadata (``organism`` / ``organism_indices``), not just the manifest, so
the served model honors it (catalog mode resolves organism from the embedded
metadata). For a checkpoint that already embeds organism provenance, omit
``--organism`` and the trained organism carries through unchanged.

Inspect or validate before sharing:

.. code-block:: bash

   agt adapters inspect dist/wtc11-atac-lora
   agt adapters validate dist/wtc11-atac-lora --base-weights model.pth

Pull a bundle (local or Hugging Face) and print its resolved local path:

.. code-block:: bash

   agt adapters pull hf://your-org/alphagenome-wtc11-atac-lora

Publish a bundle to the Hugging Face Hub (requires the ``hf`` extra,
``pip install 'alphagenome-pytorch[hf]'``):

.. code-block:: bash

   agt adapters publish dist/wtc11-atac-lora hf://your-org/alphagenome-wtc11-atac-lora

URI Forms
~~~~~~~~~

Anywhere ``--checkpoint`` or ``--source`` accepts a bundle reference,
the following are recognized:

- bare path or ``local:/abs/path`` — local directory or local file.
- ``file:///abs/path`` — local URL.
- ``hf://org/repo[/subdir][@revision]`` — Hugging Face Hub.

Singleton Serving
-----------------

The existing ``--checkpoint`` flag accepts a delta checkpoint, a delta-weights
``.safetensors`` file, *or* a bundle URI:

.. code-block:: bash

   agt serve \
     --weights base.safetensors \
     --checkpoint hf://your-org/alphagenome-wtc11-atac-lora \
     --fasta hg38.fa \
     --rest-port 8080

Singleton mode is fully compatible with the official ``alphagenome`` gRPC
client because the server has exactly one model and the client does not need
to select one.

Catalog Serving
---------------

Catalog mode hosts multiple finetunes over a shared base trunk and is
**REST-first**. gRPC is supported only for custom clients that pass an
explicit ``alphagenome-model-id`` metadata header.

Catalog file (``adapters.yaml``):

.. code-block:: yaml

   base:
     id: alphagenome-base               # optional; if present, base is also served
     label: AlphaGenome (base)
   adapters:
     - id: wtc11-atac-lora
       source: hf://your-org/alphagenome-wtc11-atac-lora
     - id: k562-rna-locon
       source: local:/srv/bundles/k562-rna-locon

Start the server:

.. code-block:: bash

   agt serve \
     --weights base.safetensors \
     --adapter-catalog adapters.yaml \
     --fasta hg38.fa \
     --rest-port 8080

REST Endpoints
~~~~~~~~~~~~~~

Catalog mode adds two GET routes alongside the existing prediction/scoring
routes:

- ``GET /v1/models`` — list all served models.
- ``GET /v1/models/{id}/metadata`` — output metadata for a specific model.

All POST routes accept an optional top-level ``model_id`` field to route the
request to a specific entry. Scoped variants
(``POST /v1/models/{id}/predict_interval``) are also available.

Model selection rules:

- Singleton mode: ``model_id`` is ignored if absent; rejected with HTTP 400
  if present and does not match.
- Catalog mode with a single model registered: ``model_id`` is optional.
- Catalog mode with multiple models registered: missing ``model_id`` →
  HTTP 400 (ambiguous); unknown ``model_id`` → HTTP 404.

gRPC Behavior
~~~~~~~~~~~~~

Catalog-mode gRPC requires every RPC to carry an ``alphagenome-model-id``
metadata header. Missing → ``FAILED_PRECONDITION``; unknown id →
``NOT_FOUND``. There is no default-model fallback: this is intentional, so
the official ``alphagenome`` client cannot silently bind to the wrong model.
Users who want official-client compatibility should run a singleton process.

How Catalog Mode Works
----------------------

The router holds one base model resident on the device and a list of
``ServedModelEntry`` objects. Each entry captures the finetune's adapter
wrapper modules (``LoRA`` / ``Locon`` / ``IA3`` / ``Houlsby``) and any new
heads. Wrappers store references to the trunk's ``Linear`` / ``Conv1d``
parameters, so all entries share the same base weights.

A request for ``model_id`` acquires the router lock, swaps the entry in (the
active entry's wrappers are detached — ``setattr`` back to ``original_layer`` —
the requested entry's wrappers are reattached, and any new heads are placed onto
``base_model.heads``), and runs the model call. **One catalog model operation
runs at a time:** the lock spans the swap *and* the inference/scoring call, so a
concurrent request can never swap the shared trunk mid-forward. Response
serialization and the network write happen after the lock is released. Swap
latency is bounded by adapter + head state-dict size, not by reloading base
weights.

Each entry is **self-describing**: it carries its own bundle's embedded track
metadata, track names, variant scorer, and default organism. An entry's service
adapter and its scorer share one runtime, so predictions and variant scores
resolve tracks and organism identically. An explicit ``--track-metadata`` at
serve time overrides the embedded metadata for every entry.

Organism defaulting:

- A request that omits ``organism`` uses the bundle's **trained organism**
  (e.g. a mouse finetune serves mouse), not human.
- An explicit ``organism`` in the request overrides the bundle default.
- A **mixed-organism** bundle (trained on more than one organism) is rejected at
  catalog startup — multi-organism serving is not supported in v1.

Constraints (v1)
~~~~~~~~~~~~~~~~

- Adapters must remain unmerged — pass ``--no-merge-adapters`` semantics
  permanently in catalog mode.
- Do not run with ``torch.compile``; swap mutates the live module tree.
- Bundles whose ``transfer_config`` uses ``keep_heads`` or ``remove_heads``
  cannot be served in catalog mode (the router cannot reversibly remove
  base heads). Use singleton mode for such adapters.
- Mixed-organism bundles (trained on more than one organism) cannot be served
  in catalog mode — there is no single default organism. Serve a
  single-organism bundle.
