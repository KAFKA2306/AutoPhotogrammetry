# Photogrammetry input audit

AutoPhotogrammetry can review a photo set before a reconstruction run and produce auditable JSON and HTML reports without modifying the original images.

This is intended for museums, archives, product teams, manufacturers, and 3D production teams that already control the rights to use the submitted photographs.

## What the audit produces

For one dataset, the existing `audit` command produces:

- `readiness-report.json`
- `readiness-report.html`
- `selected-manifest.json`
- a separate `selected/` directory containing the non-destructively selected images

The report records the input and selected image counts, low-sharpness and near-duplicate counts, duplicate manifest hashes, provenance coverage, image dimensions, content types, and `generated_views_used=false`.

It does **not** claim reconstruction quality. Registration rate, reprojection error, mesh completeness, and other reconstruction measurements remain unset until they are actually measured by a reconstruction run.

## Run a local audit

The input dataset must already contain supported images and the collection `manifest.json` used by the repository's existing provenance path.

```bash
python main.py audit --dataset <dataset-id>
```

The default output is written under:

```text
output/<dataset-id>/
```

The input photographs are not renamed or deleted by the audit.

## When this is useful

An input audit can help answer practical questions before spending GPU time on reconstruction:

- Are duplicate or near-duplicate photographs dominating the set?
- Are some photographs too soft for the current selection threshold?
- Is provenance complete for every submitted image?
- What image sizes and content types are present?
- Which selected files should be handed to the next reconstruction step?

COLMAP's current capture guidance recommends textured scenes, similar illumination, high visual overlap, and different viewpoints for image-based reconstruction. The audit reports evidence available in this repository; it does not replace those capture requirements or guarantee that Structure-from-Motion will succeed.

Reference: https://colmap.github.io/tutorial.html

## Free sample and paid PoC

A first evaluation can use a small, redistribution-safe sample to demonstrate the report format and the non-destructive selection output.

A paid proof of concept can use one customer-controlled object or asset and an agreed photo set. The initial deliverable can include:

- the input audit JSON and HTML reports
- the selected image manifest and selected image set
- provenance and SHA-256 evidence available from the submitted manifest
- one agreed reconstruction backend run when the required environment is available
- the backend run manifest and generated artifact list
- explicit notes about measurements that were not performed

Pricing, turnaround time, image count, reconstruction backend, and any confidentiality requirements must be agreed before customer data is transferred. Do not upload customer photographs or confidential material to a public GitHub issue.

## Start a PoC inquiry

[Open a pre-filled PoC inquiry](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?title=Photogrammetry+input+audit+PoC+inquiry&body=Organization+%2F+role%3A%0AObject+or+asset+type%3A%0AApproximate+image+count%3A%0ADo+you+control+the+rights+to+use+these+images%3F+%28yes%2Fno%2Funsure%29%3A%0ADesired+timing%3A%0AWhat+would+you+like+to+verify%3F%0A%0ADo+not+include+private+images%2C+personal+data%2C+credentials%2C+unpublished+contract+terms%2C+or+other+confidential+information+in+this+public+GitHub+issue.)

GitHub Issues are public in this repository. Use the issue only to provide non-confidential qualification information. Arrange a private transfer method separately before sharing customer images, personal data, credentials, or unpublished commercial information.
