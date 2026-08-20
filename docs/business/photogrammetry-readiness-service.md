# Photogrammetry readiness audit service

AutoPhotogrammetry can turn a rights-cleared photo set into an auditable pre-reconstruction package: a JSON/HTML readiness report, a non-destructively selected image set, provenance evidence, and—when explicitly included—an external reconstruction run manifest.

This service is for museums and archives, product/EC teams and manufacturers, and 3D production teams that already control the submitted photographs or have confirmed permission to use them.

## What is being sold

The initial service is a **photo-set audit and reconstruction-preparation service**, not a guarantee that a complete or metrically accurate 3D model will be produced.

The audit can report evidence such as:

- input and selected image counts
- exact and near-duplicate evidence
- low-sharpness warnings
- provenance coverage and SHA-256
- image dimensions/content types
- a non-destructive selected set and manifest
- `generated_views_used=false`
- an explicitly supplied backend run status/manifest when a reconstruction is part of the engagement
- measurements that remain unavailable rather than replacing them with inferred scores

Registration rate, reprojection error, geometry completeness and texture quality are not promised when they have not actually been measured.

## Required material and rights

For a customer dataset:

1. The customer must control the submitted photographs or provide a confirmed license/permission basis for the intended processing.
2. Do not submit private photographs, personal information, credentials, unpublished contract terms or other confidential material through a public GitHub issue.
3. A private transfer method must be agreed before customer image bytes are transferred.
4. Customer images are not converted into public repository fixtures without separate explicit permission.

## Free sample

The free sample is intended to demonstrate the report format and evidence contract using a small redistribution-safe or synthetic dataset. It does not include a commitment to a full reconstruction or a private customer-data transfer.

Typical sample output:

- readiness JSON/HTML
- selected-manifest JSON
- reason-code summary
- provenance and hash fields
- explicit unmeasured-quality fields

See `sample-readiness-report.json` in this directory for an illustrative, non-customer sample record.

## Paid one-object PoC

A paid PoC can cover one customer-controlled object/asset and an agreed photo set, typically tens to hundreds of images. Scope is agreed before transfer and can include:

- readiness audit
- selected backend-input set
- provenance/SHA-256 evidence available from the submitted manifest
- one agreed reconstruction backend run when the required environment is available
- backend run manifest and artifact inventory
- explicit notes about failed or unmeasured criteria

Pricing, turnaround time, image count, reconstruction backend, confidentiality requirements and the rights basis are agreed per PoC. No automatic payment or SaaS subscription is required for the first validation stage.

## Batch / deployment discussion

For 10+ objects, the discussion can cover repeatable batch processing, capture guidance, recapture rounds and/or private deployment of the CLI. A batch request is not treated as proven demand until an actual qualified inquiry is recorded.

## Calls to action

These public issue links collect **non-confidential qualification information only**.

### 撮影セットを監査する

[Open an input-audit inquiry](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?title=Photogrammetry+input+audit+inquiry&body=Organization+type+%28museum%2Farchive%2FEC%2Fmanufacturer%2F3D+production%2Fother%29%3A%0AObject+or+asset+type%3A%0AApproximate+image+count%3A%0ADo+you+control+the+rights+to+use+the+images%3F+%28yes%2Fno%2Funsure%29%3A%0AWhat+would+you+like+the+audit+to+check%3F%3A%0A%0ADo+not+attach+private+images%2C+personal+data%2C+credentials%2C+or+confidential+information.)

### 1対象物のPoCを相談する

[Open a one-object PoC inquiry](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?title=Photogrammetry+one-object+PoC+inquiry&body=Organization+type%3A%0AObject+or+asset+type%3A%0AApproximate+image+count%3A%0AIntended+3D+use%3A%0ADo+you+control+the+rights+to+use+the+images%3F+%28yes%2Fno%2Funsure%29%3A%0ADesired+timing%3A%0AWhat+decision+should+the+PoC+support%3F%3A%0A%0ADo+not+attach+private+images%2C+personal+data%2C+credentials%2C+or+confidential+information.)

### 大量3D化を相談する

[Open a batch/deployment inquiry](https://github.com/KAFKA2306/AutoPhotogrammetry/issues/new?title=Photogrammetry+batch+or+deployment+inquiry&body=Organization+type%3A%0AApproximate+object+count%3A%0AApproximate+images+per+object%3A%0ACapture+workflow%3A%0AIntended+3D+use%3A%0APreferred+operation+%28managed+batch%2Fprivate+CLI%2Fprivate+deployment%2Funsure%29%3A%0ADo+you+control+the+rights+to+use+the+images%3F+%28yes%2Fno%2Funsure%29%3A%0A%0ADo+not+attach+private+images%2C+personal+data%2C+credentials%2C+or+confidential+information.)

## Measurement contract

Only the following funnel events are counted for the initial validation:

- `service_page_viewed`
- `sample_report_opened`
- `audit_inquiry_started`
- `qualified_inquiry`
- `pilot_booked`
- `paid_pilot`

These events must not contain customer image bytes, names, email addresses, phone numbers, issue-body text or other personal/confidential payloads. `readiness-service-kpi-template.json` defines the 60-day targets and the empty evidence slots. Empty/null evidence means **not yet measured**, not zero success.

The external success criteria in Issue #3 remain empirical. Publishing this page or adding tracking infrastructure does not count as a proposal, qualified inquiry, booked pilot or paid pilot.
