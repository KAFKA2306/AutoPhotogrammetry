# Nordic Wikimedia discovery

The Nordic source path is deliberately separate from the fixed final-exhibition registry.

## Authorities

- Seed categories: `sources/discovery/nordic-seeds.json`
- MediaWiki Action API `categorymembers`: https://www.mediawiki.org/wiki/API:Categorymembers
- TimedMediaHandler `videoinfo`: https://www.mediawiki.org/wiki/Extension:TimedMediaHandler/API
- MediaWiki REST file information: https://www.mediawiki.org/wiki/API:REST_API/Reference
- Discovery pool: `sources/discovery/nordic-wikimedia.json`
- Eight-region coverage: `sources/discovery/nordic-coverage.json`
- Explicit Stage A selection: `sources/discovery/nordic-preflight-queue.json`

`categorymembers` continuation is followed to exhaustion and subcategories are traversed with cycle protection by `processing/wikimedia_discovery.py`. Video-specific metadata comes from `videoinfo`; the REST file endpoint is only a duration fallback when the Action API response does not contain duration.

The checked-in initial pool is a conservative bootstrap from issue #84 category-page observations. Missing media URLs, Commons SHA-1 values, byte sizes, authors, and verified licenses are stored as missing instead of inferred, so those records cannot pass Stage A until an API refresh supplies the evidence.

## Refresh and validation

```bash
uv run python -m processing.nordic_pool refresh
uv run python -m processing.nordic_pool validate
```

A refresh fails before replacing the previous snapshots when category or metadata retrieval is partial. Timestamps are excluded from content comparison, so a successful refresh with unchanged content does not create a timestamp-only Git diff.

The scheduled workflow `.github/workflows/nordic-wikimedia-discovery.yml` runs weekly and also supports `workflow_dispatch`. It commits only the pool and coverage JSON files to a review branch and opens a pull request when their content changes. It does not commit source videos, checkpoints, or PLY files.

## Stage A selection

A candidate passes Stage A only when the stored evidence confirms all required source, rights, media, identity, duration, resolution, and downloadability fields. Unknown or review-needed licenses fail closed. The 120-second long-form criterion is a separate Boolean gate; no aggregate score, rank, or expected-success field is generated.

Selection is explicit:

```bash
uv run python -m processing.nordic_pool queue <candidate-id>
```

Only Stage A pass records can enter `sources/discovery/nordic-preflight-queue.json`. Re-running the same selection is idempotent. If the Commons SHA-1 or media URL changes, the queued record is treated as stale instead of silently changing identity.

## Existing Stage B preflight

Run one or more explicitly queued candidates with:

```bash
uv run python -m processing.nordic_preflight <candidate-id> [<candidate-id> ...]
```

The selected media is downloaded outside Git, verified against the Commons SHA-1 and byte size, and assigned a downloaded-byte SHA-256. The existing `processing/video_preflight.py` implementation then measures exactly the current Stage B fields:

- `scene_cut_count`
- `sharp_frame_ratio`
- `adjacent_view_overlap`
- `camera_translation_proxy`
- `dynamic_pixel_ratio`
- `exposure_variation`

Each candidate gets an evaluation registry under `output/nordic-preflight/<candidate-id>/evaluation-registry.json`. That registry uses the same evaluation schema and policy as `sources/videos.json`, but it is separate from the canonical final-exhibition registry. One failed candidate does not erase other preflight results, and no candidate is automatically advanced to COLMAP.

## Final-20 boundary

`processing/exhibition_manifest.py` requires exactly 20 entries in canonical `sources/videos.json`. Nordic discovery, Stage A selection, and Stage B evaluation therefore do not append candidates to that file. This preserves the #33 final-exhibition contract while allowing an independent, variable-size discovery and evaluation path.
