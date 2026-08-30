---
name: splat-review-loop
description: "Run iterative Gaussian Splat reconstruction experiments, compare rendered RGB/GT evidence, and leave reproducible issue reviews. Use when improving or auditing Splat quality from video-derived datasets."
---

# Splat Review Loop

Use this skill when a user asks to generate, improve, compare, or review Gaussian Splats made from video frames.

## Required outcome

For every candidate that is rendered, produce a reproducible review containing:

- the exact source-video/dataset identity and training configuration;
- the evaluation view IDs and metric definition;
- RGB and ground-truth contact sheets in WEBP format;
- numerical comparison (at minimum MSE and PSNR when GT exists);
- concrete visual findings tied to view IDs and image regions;
- a keep/reject decision and the next experiment;
- a GitHub Issue comment with working raw WEBP links.

Do not report a candidate as improved from metrics alone. Inspect both RGB and GT images before deciding.

## Establish scope before training

Count source videos separately from Splat candidates. One source video can produce many training variants. Read the canonical video registry and identify whether the task covers one video, all registry videos, or an additional retrospective source. Do not claim that the registry has been processed because one video's parameter sweep was processed.

For each source video, establish a baseline first. Keep its evaluation split, view IDs, image resolution, and metric code fixed for the whole comparison. Do not compare a late temporal cluster, crop, mask-derived dataset, or different view set directly against a full-scene global best; label it as conditional evidence.

Before a batch run, validate that the source frames and camera metadata exist and that durable artifact storage is authenticated and writable. Run one end-to-end pilot (train → export → render → publish/resolve → review) before scaling to many videos.

## Experiment loop

1. Inspect the current baseline, existing experiment outputs, and recent Issue comments. Choose a hypothesis that addresses an observed failure; avoid blind parameter sweeps.
2. Train a candidate with an explicit run identity. Preserve the command/config and seed.
3. Export the PLY and calculate SHA-256 and exact byte size. Treat a changed SHA as a new artifact.
4. Render the fixed evaluation views with both RGB and GT outputs. Convert the selected views to RGB and GT contact-sheet WEBPs.
5. View both WEBPs. Record view-specific artifacts such as floating splats, water/sky ghosts, streaks, missing geometry, color drift, or loss of structural detail.
6. Compute the same metrics used for the baseline. Select a new best only when it improves the fixed metric and does not introduce a material visual regression.
7. Publish PLY artifacts with the official Hugging Face `batch_bucket_files()` API, verify exact read-back with `download_bucket_files()`, then upload WEBPs to the designated evidence branch or artifact location, verify each raw URL with HTTP 200 and non-zero size, and comment on the relevant GitHub Issue. Include immutable commit/revision links when available.
8. Decide the next hypothesis from the evidence. Stop a parameter family when multiple neighboring values are worse; switch to data, masks, camera poses, frame selection, or reconstruction strategy instead of endlessly repeating the same sweep.

## Artifact and repository safety

Read and obey `.agents/skills/artifact-storage/SKILL.md` for PLY handling. Never commit generated PLY files, use GitHub raw as a PLY fallback, force-add ignored PLYs, or invent an external locator. If durable artifact storage is unavailable, mark the artifact blocked/unavailable and continue with lightweight review evidence only when appropriate.

WEBPs and lightweight metadata may be committed to the designated evidence location, but never use evidence uploads to bypass the PLY policy. Keep the PLY SHA-256, byte size, source/run identity, producing revision, and evaluation result together.

## Issue review format

Each rendered candidate comment should state:

- configuration and training step count;
- fixed view IDs and metric values;
- comparison against the correct same-scope baseline;
- at least two concrete visual observations, including weak views/regions;
- keep/reject status and why;
- PLY SHA/size without embedding the PLY;
- RGB WEBP and GT WEBP raw links, each verified before posting.

If the render failed, do not fabricate a visual review or links. Report the failure and repair the render pipeline before treating the candidate as evaluated.

## Stopping and handoff

Stop the current brute-force loop when the best score plateaus across a meaningful neighborhood, when the same artifact persists across unrelated parameter changes, or when storage/authentication is not ready for the requested batch. Handoff should identify the verified best per source video, unresolved visual defects, unevaluated videos, and the next data-level intervention.
