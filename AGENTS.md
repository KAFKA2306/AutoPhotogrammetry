# AGENTS.md

## Photogrammetry / 3D reconstruction

- Do not optimize image count or COLMAP registered count/ratio as a proxy for reconstruction quality.
- Before tuning training parameters, validate the input shot, timestamps, perspective directions, crop, COLMAP pose connectivity, and registered image identities.
- Prefer controlled A/B comparisons. Change one experimental responsibility at a time and keep source identity, evaluation views, and unrelated settings fixed.
- Treat additional images as useful only when they add usable geometric constraints. Explicitly test whether removing weak, sky/sea-dominated, redundant, or poorly connected views improves reconstruction.
- Evaluate registration evidence and downstream reconstruction quality separately. At minimum inspect pose/component connectivity, major-structure retention on fixed hold-out views, floaters/smear, and NaN/Inf.
- Freeze the selected dataset identity before camera-optimizer, iteration-budget, regularization, or other training-side A/B experiments.
- Preserve experiment-specific numbers, renders, hashes, and conclusions in Issues or machine-readable artifacts. Keep this file limited to reusable decision rules.

## Artifact storage — mandatory

- Never commit generated PLY files to Git, including evidence, fallback, retrospective, temporary, or diagnostic branches.
- Never use a GitHub branch, Git blob, raw GitHub URL, or release commit as fallback storage for generated PLY bytes.
- Materialize generated PLYs through the repository artifact publishing/cache path. Git may contain only lightweight metadata such as SHA-256, byte size, durable artifact locator, provenance, run identity, and evaluation result.
- If durable artifact storage is unavailable, leave materialization explicitly blocked. Do not bypass the block by committing the binary to Git.
- Before committing, verify that no `*.ply` is tracked by Git. The repository CI enforces this rule independently of agent behavior.

Use `.agents/skills/photogrammetry-reconstruction/SKILL.md` for the repeatable experiment workflow.
Use `.agents/skills/artifact-storage/SKILL.md` whenever an experiment produces or republishes generated artifacts.
