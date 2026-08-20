from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

EXPECTED_REGION_IDS = (
    "norway",
    "sweden",
    "finland",
    "denmark",
    "iceland",
    "greenland",
    "faroe-islands",
    "aland",
)
SEED_TYPES = {"drone", "aerial", "general-video"}


def load_nordic_seeds(path: str | Path = "sources/discovery/nordic-seeds.json") -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_nordic_seeds(payload)
    return payload


def validate_nordic_seeds(payload: Mapping) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Nordic seed schema_version must be 1")
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Nordic seed regions must be a list")

    region_ids = [region.get("id") for region in regions if isinstance(region, Mapping)]
    if tuple(region_ids) != EXPECTED_REGION_IDS:
        raise ValueError(
            f"Nordic seed coverage/order must be exactly {EXPECTED_REGION_IDS}; got {region_ids}"
        )

    categories: set[str] = set()
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("Every Nordic region must be an object")
        status = region.get("status")
        seeds = region.get("seeds")
        if status not in {"configured", "missing"} or not isinstance(seeds, list):
            raise ValueError(f"Invalid Nordic seed region: {region.get('id')}")

        if status == "missing":
            if seeds:
                raise ValueError(f"Missing region cannot have active seeds: {region.get('id')}")
            if not region.get("missing_reason"):
                raise ValueError(f"Missing region requires missing_reason: {region.get('id')}")
            continue

        if not seeds:
            raise ValueError(f"Configured region requires at least one seed: {region.get('id')}")
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise ValueError(f"Invalid seed object in {region.get('id')}")
            title = seed.get("category_title")
            source_url = seed.get("source_url")
            seed_type = seed.get("seed_type")
            if not isinstance(title, str) or not title.startswith("Category:"):
                raise ValueError(f"Seed must use an exact Commons Category title: {title!r}")
            if title in categories:
                raise ValueError(f"Duplicate Commons seed category: {title}")
            categories.add(title)
            if seed.get("enabled") is not True:
                raise ValueError(f"Configured seed must be explicitly enabled: {title}")
            if seed_type not in SEED_TYPES:
                raise ValueError(f"Unsupported seed_type {seed_type!r}: {title}")
            if not isinstance(source_url, str) or not source_url.startswith(
                "https://commons.wikimedia.org/wiki/Category:"
            ):
                raise ValueError(f"Seed requires a confirmed Commons category URL: {title}")


def enabled_seed_categories(payload: Mapping) -> list[dict[str, str]]:
    validate_nordic_seeds(payload)
    result: list[dict[str, str]] = []
    for region in payload["regions"]:
        for seed in region["seeds"]:
            result.append(
                {
                    "region_id": region["id"],
                    "region_name": region["display_name"],
                    "category_title": seed["category_title"],
                    "seed_type": seed["seed_type"],
                    "source_url": seed["source_url"],
                }
            )
    return result
