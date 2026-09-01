from json import load
from pathlib import Path
BENCHMARK = Path(__file__).parents[1] / "docs" / "benchmarks" / "4dgs-services.json"
STATUS_VALUES = {"PASS", "PARTIAL", "FAIL", "BLOCKED", "UNVERIFIED"}


def _load() -> dict:
    with BENCHMARK.open(encoding="utf-8") as stream:
        return load(stream)


def test_4dgs_service_benchmark_has_single_machine_readable_authority() -> None:
    data = _load()

    assert data["schema_version"] == 1
    assert set(data["status_values"]) == STATUS_VALUES
    assert data["rules"]["vendor_claims_are_measurements"] is False
    assert data["rules"]["paper_metrics_are_local_measurements"] is False
    assert data["rules"]["unknown_values_are_null"] is True

    entries = data["entries"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert {"nexia-video", "gracia"}.issubset(ids)


def test_runtime_measurements_do_not_promote_unmeasured_values() -> None:
    data = _load()

    for entry in data["entries"]:
        assert entry["status"] in STATUS_VALUES
        runtime = entry["runtime"]
        assert runtime["status"] in STATUS_VALUES
        assert isinstance(runtime["local_playback_verified"], bool)

        for metric in (
            "transfer_bytes",
            "time_to_first_visible_frame_ms",
            "steady_state_fps",
            "seek_latency_ms",
            "viewpoint_change_latency_ms",
        ):
            assert metric in runtime

        if all(
            runtime[metric] is None
            for metric in (
                "transfer_bytes",
                "time_to_first_visible_frame_ms",
                "steady_state_fps",
                "seek_latency_ms",
                "viewpoint_change_latency_ms",
            )
        ):
            assert runtime["status"] != "PASS"

        if runtime["local_playback_verified"] is False:
            assert runtime["status"] != "PASS"


def test_nexia_compression_is_explicitly_vendor_claim_only() -> None:
    data = _load()
    nexia = next(entry for entry in data["entries"] if entry["id"] == "nexia-video")

    compression = nexia["public_claims"]["compression"]
    assert compression["basis"] == "provider_test_environment"
    assert compression["locally_reproduced"] is False
    assert nexia["runtime"]["status"] == "BLOCKED"
    assert nexia["runtime"]["public_runtime_asset_reference_confirmed"] is False


def test_gracia_public_asset_reference_and_sdk_revision_are_traceable() -> None:
    data = _load()
    gracia = next(entry for entry in data["entries"] if entry["id"] == "gracia")

    repo_source = next(
        source for source in gracia["sources"] if source["type"] == "official_repository"
    )
    assert repo_source["revision"] == "7d649d3c362f6911f0e1fbad51bb25cf56dee23f"
    assert gracia["runtime"]["public_runtime_asset_reference_confirmed"] is True
    assert gracia["runtime"]["local_playback_verified"] is False
    assert gracia["runtime"]["public_asset_examples"]
    assert gracia["integration"]["locally_verified"] is False
    assert gracia["integration"]["status"] == "PARTIAL"
    assert gracia["licensing"]["web_sdk"] == "proprietary"
    assert gracia["licensing"]["examples_and_documentation"] == "MIT"
