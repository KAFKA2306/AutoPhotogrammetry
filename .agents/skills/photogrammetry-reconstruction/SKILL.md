---
name: photogrammetry-reconstruction
description: Use when running, reviewing, or planning photogrammetry or 3DGS reconstruction experiments involving source-shot selection, temporal sampling, perspective-view selection, COLMAP registration/connectivity, dataset freezing, camera optimization, or reconstruction-quality A/B tests. Do not use raw image count or registered ratio alone to select a winner.
---

# Photogrammetry Reconstruction

Use this workflow to turn reconstruction experiments into controlled, reviewable evidence rather than parameter search driven by proxy metrics.

## Core rule

More observations are not automatically more useful geometric constraints.

Do not select a dataset or training configuration from image count, COLMAP registered count, or registered ratio alone. Compare pose evidence and downstream reconstruction quality separately.

## Workflow

### 1. Fix the experimental responsibility

State exactly what changes in the current experiment and what remains fixed.

Prefer one responsibility per comparison, for example:

- shot / time interval
- temporal sampling density
- perspective direction set
- crop
- COLMAP camera or matching configuration
- connected-component choice
- camera optimizer
- iteration budget
- regularization or artifact-control parameter

Do not mix input-selection experiments with training-parameter experiments when the effects cannot be separated.

### 2. Fix source identity

Before comparing variants, record enough information to reproduce the source:

- source URL or local identity
- checksum when materialized
- shot or time range
- source-frame timestamps or identities
- projection settings and crop

For a controlled A/B, keep these fixed unless one of them is the explicit variable under test.

### 3. Evaluate input geometry before training tuning

For each input candidate, inspect at least:

- generated image identities/count
- registered image identities/count
- COLMAP connected components / pose connectivity
- reprojection evidence when available
- whether registered views cover the target structure rather than only easy background regions

Treat weakly textured, sky/sea-dominated, redundant, or poorly connected views as candidates for removal. Do not assume adding directions or timestamps is beneficial.

### 4. Use a fixed downstream evaluator

When comparing input candidates, keep the reconstruction/training baseline fixed and render the same hold-out viewpoints.

Compare against ground truth or the source frame and inspect at least:

- major-structure retention
- geometry or silhouette consistency
- floaters
- smear / streaking / duplicated surfaces
- obvious transparency or banding artifacts
- NaN/Inf or other numerical failures

Do not promote a candidate solely because its COLMAP registration ratio improved.

### 5. Select and freeze the dataset

Select the dataset only when the combined pose evidence and fixed-view reconstruction evidence justify it.

A valid outcome is `winner: none` when no candidate produces stable reconstruction.

Once selected, freeze and record the dataset identity before changing:

- camera optimizer
- iteration budget
- Scale Regularization / MCMC
- antialiasing, anisotropy, culling, densification, or related training controls

Do not silently change the input dataset during downstream training A/B tests.

### 6. Run training-side A/B tests only on the frozen dataset

Change one training responsibility at a time where practical. Keep the dataset identity and evaluation views fixed.

Training longer is not evidence of improvement by itself. Compare the same reconstruction-quality evidence after the run.

### 7. Preserve evidence

Keep experiment-specific numbers and outputs out of this skill. Store them in repository Issues or artifacts so later agents can audit the actual run.

For each comparison, preserve when available:

- changed variable
- fixed variables
- source/dataset identity or hash
- generated and registered image identities/counts
- pose/component evidence
- evaluation-view identities
- render/GT evidence
- NaN/Inf or numerical-failure counts
- decision: PASS / PARTIAL PASS / FAIL, with the reason

Prefer an existing machine-readable evidence format in the repository. Do not create a parallel schema when an established artifact already represents the same information.

## Decision rules

Use these rules when conclusions conflict:

1. Actual hold-out reconstruction evidence outranks registration ratio alone.
2. Major target-structure retention outranks a higher count of easy/background registered views.
3. A controlled comparison outranks an uncontrolled run with more images or more iterations.
4. Input quality and pose connectivity are resolved before attributing failure to training parameters.
5. If the evidence does not distinguish candidates, record that result instead of manufacturing a winner.

## AutoPhotogrammetry issue routing

Use the existing Issue authority instead of creating a new overlapping experiment thread:

- #140 — parent / 360-degree source and end-to-end benchmark program
- #146 — exact same-source, same-shot, same-source-frame `8-view` vs `14-view` controlled comparison only
- #148 — shot, temporal density, direction selection, crop, COLMAP camera/matching, pose connectivity, frozen dataset selection
- #149 — camera optimizer A/B on the frozen dataset
- #150 — Splatfacto artifact-control parameter A/B on the frozen dataset
- #41 — Scale Regularization A/B
- #43 — default / Scale Regularization / MCMC production-strategy comparison
- #151 — representative real PLY/WebP evidence-package curation and VRMine handoff

Historical comments may contain experiments that now belong to another Issue. Preserve them as evidence, but do not continue that responsibility in the historical thread.

Do not create a new Issue when one of the authorities above already owns the changed variable. If a new experiment changes multiple responsibility classes, split it before execution.

## Evidence boundary

Use this skill for rules and decision procedure only.

Do not copy run-specific values, hashes, winner names, view counts, or current best parameters into the skill. Those belong to Issues or machine-readable artifacts and may change independently.

AI-generated summary images may be explanatory material, but they are never reconstruction evidence. Evidence must trace to actual source/run artifacts such as real WebP renders, GT images, PLY files, manifests, logs, metrics, and hashes.

## Repository evidence

The workflow is grounded in controlled evidence and responsibility separation from:

- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/140
- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/146
- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/148
- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/149
- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/150
- https://github.com/KAFKA2306/AutoPhotogrammetry/issues/151

Treat those Issues as evidence and routing authority, not as universal numeric thresholds or a requirement to reuse a specific number of views.
