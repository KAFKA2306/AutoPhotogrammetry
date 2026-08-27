---
name: artifact-storage
description: Mandatory artifact handling rules for generated reconstruction binaries such as Gaussian Splat PLY files.
---

# Artifact Storage

## Mandatory rules

1. Never commit generated `*.ply` files to Git, regardless of branch name, evidence purpose, fallback need, retrospective use, or temporary status.
2. Never use `git add -f`, Git plumbing, GitHub branch storage, Git blobs, raw GitHub URLs, or committed binaries to bypass artifact storage policy.
3. Store generated PLY bytes through the repository artifact publishing/cache path.
4. Keep only lightweight artifact identity and provenance in Git: SHA-256, exact byte size, durable artifact locator, source/run identity, producing revision, and evaluation result.
5. If durable artifact storage cannot be used, record the artifact as blocked or unavailable. Do not invent a locator and do not commit the binary.
6. Existing tracked PLY files on historical evidence branches are debt, not precedent. Do not add or modify them.
7. A regenerated file with a different SHA-256 is a new artifact. Never present it as recovery of an older artifact.

## Completion condition

Artifact handoff is complete only when SHA-256, byte size, and durable external locator resolve to the exact same bytes outside Git.
