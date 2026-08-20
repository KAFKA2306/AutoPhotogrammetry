# Nordic Wikimedia video discovery

This repository keeps broad Nordic source discovery separate from the fixed evaluation and exhibition registries.

## Scope

The discovery configuration covers Norway, Sweden, Finland, Denmark, Iceland, Greenland, the Faroe Islands, and Åland. A region with no confirmed Wikimedia Commons video category is recorded as `missing`; the code does not invent a category name.

Primary upstream interfaces:

- MediaWiki Action API `categorymembers`: https://www.mediawiki.org/wiki/API:Categorymembers
- TimedMediaHandler `videoinfo`: https://www.mediawiki.org/wiki/Extension:TimedMediaHandler/API
- MediaWiki REST API file information: https://www.mediawiki.org/wiki/API:REST_API/Reference

## Data flow

```text
sources/discovery/nordic-seeds.json
  -> categorymembers traversal
  -> videoinfo + file categories
  -> optional REST file metadata fallback
  -> sources/discovery/nordic-wikimedia.json
  -> Stage A metadata gate
  -> explicit promotion
  -> existing video_preflight.py
  -> existing #23 COLMAP / Splatfacto evaluation
```

`sources/videos.json` remains the evaluation registry. The Nordic discovery pool can contain any number of candidates and does not change the exactly-20 exhibition contract.

## Commands

Refresh the discovery pool and coverage report:

```bash
python -m processing.nordic_discovery discover
```

Validate the checked-in pool without network access:

```bash
python -m processing.nordic_discovery validate
```

Promote one Stage A candidate into the evaluation registry:

```bash
python -m processing.nordic_discovery promote <candidate-id>
```

Download, verify SHA-1/size, record the downloaded SHA-256, and run the existing CPU preflight:

```bash
python -m processing.nordic_discovery preflight <candidate-id>
```

Promotion and preflight fail when Stage A is incomplete. `rank`, `score`, and `expected_success` are not part of the discovery schema.

## Bootstrap snapshot

The checked-in initial pool is intentionally incomplete metadata. It records candidates observed on Wikimedia Commons category pages, but leaves original media URLs, authors, licenses, MIME types, Commons SHA-1 values, and downloadability unset. Those records therefore fail Stage A until the API-backed discovery job refreshes them.

This prevents category-page observations from being represented as verified file provenance.

## Scheduled refresh

`.github/workflows/nordic-wikimedia-discovery.yml` runs weekly and can also be triggered manually. It regenerates the pool and coverage report, validates them, runs the Nordic unit tests, and opens a pull request only when the checked-in data changed.

If the MediaWiki traversal or metadata retrieval reports failures, the default discovery command exits before replacing the previous checked-in snapshot.
