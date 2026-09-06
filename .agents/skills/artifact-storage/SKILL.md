# Artifact Storage

Use this skill whenever an experiment creates, republishes, restores, or hands off generated artifacts such as Gaussian Splat PLY files.

## Canonical interface

Publish a successful reconstruction through the repository CLI:

```bash
python main.py publish-splat --run-manifest output/<dataset>/manifest.json
```

The CLI requires the configured artifact bucket and `hf-cache-hub` publisher (`HF_ARTIFACT_BUCKET` and `HF_CACHE_HUB_ROOT`, or their explicit CLI equivalents). Agents do not call storage-provider upload/download APIs directly and do not choose an alternate publisher.

## Mandatory rules

1. Never commit generated `*.ply` files to Git.
2. Never use a GitHub branch, Git blob, raw GitHub URL, or committed binary as an artifact fallback.
3. Keep only lightweight identity and provenance in Git: SHA-256, exact byte size, durable locator, source/run identity, producing revision, and evaluation result.
4. If durable publishing is unavailable, record the artifact as blocked/unavailable. Do not invent a locator.
5. A regenerated file with a different SHA-256 is a new artifact. Never present it as recovery of older bytes.
6. Repository CI owns the tracked-PLY prohibition; do not weaken or bypass it.

## Completion condition

Artifact handoff is complete only when the canonical publisher reports successful remote read-back and the declared SHA-256, byte size, and durable locator resolve to the exact same bytes outside Git.
