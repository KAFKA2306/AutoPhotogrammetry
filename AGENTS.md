# AGENTS.md

## Evidence branch freeze — mandatory

This branch contains historical evidence and existing large binary debt. It is not an artifact store.

- Do not add, modify, republish, or force-add any generated `*.ply` file on this branch.
- Do not use `git add -f`, Git plumbing, the GitHub Contents API, or any other mechanism to bypass ignore rules for generated artifacts.
- Do not use GitHub branches, Git blobs, raw GitHub URLs, or commits as fallback storage for generated PLY bytes.
- Existing tracked PLY files are historical debt only; their presence is not precedent for adding more.
- Publish generated PLY bytes through the repository artifact publishing/cache path. Keep only SHA-256, exact byte size, durable external locator, provenance, run identity, and evaluation metadata in Git.
- If durable artifact storage is unavailable, record materialization as blocked. Do not commit the binary as a fallback.
- Treat `evidence/360-render-8view-20260826` as frozen for new binary evidence. Any further experiment must not increase the set of tracked PLY blobs.

Use `.agents/skills/artifact-storage/SKILL.md` for artifact handling.

## Official Hugging Face upload method

For authenticated Storage Bucket publishing, use the official `huggingface_hub`
`batch_bucket_files()` API and verify exact read-back with
`download_bucket_files()`. The canonical remote layout is
`autophotogrammetry/gaussian-splats/<dataset>/<sha256>.ply`. Do not diagnose a
stalled `hf buckets cp` as an authentication failure when `hf auth whoami` and
bucket listing succeed; use the Python batch API instead.
