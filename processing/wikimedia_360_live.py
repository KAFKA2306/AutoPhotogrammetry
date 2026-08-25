from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from processing.wikimedia_360 import DEFAULT_POOL, build_pool, validate_pool
from processing.wikimedia_discovery import _request_file_json, _request_json

MAX_ATTEMPTS = 5
DEFAULT_RETRY_SECONDS = 5.0


def _retry_after_seconds(error: HTTPError, attempt: int) -> float:
    header = error.headers.get("Retry-After") if error.headers is not None else None
    if header is not None:
        try:
            value = float(header)
        except (TypeError, ValueError):
            value = -1.0
        if value >= 0:
            return value
    return DEFAULT_RETRY_SECONDS * (2**attempt)


def _with_backoff(
    operation: Callable[[], dict[str, Any]],
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(max_attempts):
        try:
            return operation()
        except HTTPError as exc:
            if exc.code not in {429, 503} or attempt == max_attempts - 1:
                raise
            sleep(_retry_after_seconds(exc, attempt))
        except RuntimeError as exc:
            if "maxlag" not in str(exc).casefold() or attempt == max_attempts - 1:
                raise
            sleep(DEFAULT_RETRY_SECONDS * (2**attempt))
    raise AssertionError("retry loop exhausted without returning or raising")


def request_json(params: Mapping[str, str]) -> dict[str, Any]:
    request_params = {**params, "maxlag": "5"}
    return _with_backoff(lambda: _request_json(request_params))


def request_file_json(canonical_title: str) -> dict[str, Any]:
    return _with_backoff(lambda: _request_file_json(canonical_title))


def refresh(output_path: str | Path = DEFAULT_POOL) -> dict[str, Any]:
    pool = build_pool(request_json=request_json, request_file_json=request_file_json)
    validate_pool(pool)

    if pool["discovery_failures"] or pool["metadata_failures"]:
        raise RuntimeError(
            "Wikimedia 360 discovery is incomplete: "
            f"discovery_failures={len(pool['discovery_failures'])}, "
            f"metadata_failures={len(pool['metadata_failures'])}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output": path.as_posix(),
        "discovered_file_count": pool["discovered_file_count"],
        "candidate_count": pool["candidate_count"],
        "stage_a_projection_review_count": sum(
            1
            for candidate in pool["candidates"]
            if candidate["stage_a"].get("eligible_for_projection_review") is True
        ),
        "eac_count": sum(
            1 for candidate in pool["candidates"] if candidate["projection_type"] == "eac"
        ),
        "metadata_failure_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the Wikimedia 360 pool with Wikimedia rate-limit backoff."
    )
    parser.add_argument("--output", default=str(DEFAULT_POOL))
    args = parser.parse_args()
    print(json.dumps(refresh(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
