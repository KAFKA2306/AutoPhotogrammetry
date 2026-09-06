# AGENTS.md

## 公開文書

- README、Issue Form、利用者向け文書は平易な日本語で書く。公式名称・標準用語・API名は正式名称を使い、独自略語を作らない。
- 公開サービスの入口と実装を分離しない。READMEから現在の相談・利用入口へ直接到達できる状態を保つ。
- synthetic / fixture / conceptは、その用途を明示し、顧客実績・production asset・品質証拠として扱わない。

## Photogrammetry / 3D reconstruction

- Do not optimize image count or COLMAP registered count/ratio as a proxy for reconstruction quality.
- Before tuning training parameters, validate the input shot, timestamps, perspective directions, crop, COLMAP pose connectivity, and registered image identities.
- Prefer controlled A/B comparisons. Change one experimental responsibility at a time and keep source identity, evaluation views, and unrelated settings fixed.
- Treat additional images as useful only when they add usable geometric constraints. Explicitly test whether removing weak, sky/sea-dominated, redundant, or poorly connected views improves reconstruction.
- Evaluate registration evidence and downstream reconstruction quality separately. At minimum inspect pose/component connectivity, major-structure retention on fixed hold-out views, floaters/smear, and NaN/Inf.
- Freeze the selected dataset identity before camera-optimizer, iteration-budget, regularization, or other training-side A/B experiments.
- Preserve experiment-specific numbers, renders, hashes, and conclusions in Issues or machine-readable artifacts. Keep this file limited to reusable decision rules.

## Artifact storage

- Generated PLY files are never committed to Git. Repository CI owns enforcement.
- Use `.agents/skills/artifact-storage/SKILL.md` whenever a run creates, republishes, restores, or hands off generated artifacts.
- The canonical publish interface is `python main.py publish-splat --run-manifest <path>`. Agent instructions must not choose a storage-provider API or alternate publisher implementation.
- The publish implementation delegates durable storage, remote read-back, SHA-256, and byte-size verification to the repository's canonical artifact publisher.
- If durable storage is unavailable, record the artifact as blocked/unavailable. Do not invent a locator or use Git as fallback storage.

Use `.agents/skills/photogrammetry-reconstruction/SKILL.md` for the repeatable experiment workflow.
