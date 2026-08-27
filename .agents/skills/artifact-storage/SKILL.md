# Artifact Storage

Use this skill whenever an experiment creates, republishes, restores, or hands off generated artifacts such as Gaussian Splat PLY files.

## Mandatory rules

1. Never commit generated `*.ply` files to Git, regardless of branch name, evidence purpose, fallback need, retrospective use, or temporary status.
2. Never use GitHub branch storage, Git blobs, raw GitHub URLs, or committed binaries as an artifact fallback.
3. Store generated PLY bytes through the repository's artifact publishing/cache path.
4. Keep only lightweight artifact identity and provenance in Git:
   - SHA-256
   - exact byte size
   - durable artifact locator
   - source/run identity
   - producing revision
   - evaluation result
5. If the durable artifact store cannot be used, record the artifact as blocked or unavailable. Do not invent a locator and do not bypass storage policy by committing the file.
6. Before any commit or PR, run `git ls-files '*.ply'`. The result must be empty.
7. A regenerated file with a different SHA-256 is a new artifact. Never present it as recovery of an older artifact.

## Completion condition

Artifact handoff is complete only when the declared SHA-256, byte size, and durable locator all resolve to the exact same bytes outside Git.
