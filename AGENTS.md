# AGENTS.md

## Photogrammetry / 3D reconstruction

- Do not optimize image count or COLMAP registered count/ratio as a proxy for reconstruction quality.
- Before tuning training parameters, validate the input shot, timestamps, perspective directions, crop, COLMAP pose connectivity, and registered image identities.
- Prefer controlled A/B comparisons. Change one experimental responsibility at a time and keep source identity, evaluation views, and unrelated settings fixed.
- Treat additional images as useful only when they add usable geometric constraints. Explicitly test whether removing weak, sky/sea-dominated, redundant, or poorly connected views improves reconstruction.
- Evaluate registration evidence and downstream reconstruction quality separately. At minimum inspect pose/component connectivity, major-structure retention on fixed hold-out views, floaters/smear, and NaN/Inf.
- Freeze the selected dataset identity before camera-optimizer, iteration-budget, regularization, or other training-side A/B experiments.
- Preserve experiment-specific numbers, renders, hashes, and conclusions in Issues or machine-readable artifacts. Keep this file limited to reusable decision rules.

Use `.agents/skills/photogrammetry-reconstruction/SKILL.md` for the repeatable experiment workflow.
