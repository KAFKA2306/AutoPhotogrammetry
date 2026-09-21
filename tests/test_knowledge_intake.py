from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from processing.knowledge_intake import build_experiment_candidates, write_experiment_candidates


class KnowledgeIntakeTests(unittest.TestCase):
    def test_routes_only_autophotogrammetry_events(self) -> None:
        events = [
            {
                "id": "ke_a",
                "claim": {"statement": "Gaussian Splat candidate", "tags": ["reconstruction"]},
                "applies_to": {"repositories": ["KAFKA2306/AutoPhotogrammetry"]},
                "evidence": {"status": "CLAIMED", "numeric_claim_present": False},
            },
            {
                "id": "ke_b",
                "claim": {"statement": "Avatar PhysBone", "tags": ["rigging_physics"]},
                "applies_to": {"repositories": ["KAFKA2306/avatars2509"]},
                "evidence": {"status": "CLAIMED", "numeric_claim_present": False},
            },
        ]
        candidates = build_experiment_candidates(events)
        self.assertEqual([item["knowledge_event_id"] for item in candidates], ["ke_a"])

    def test_source_numeric_claim_never_populates_actual_metric(self) -> None:
        [candidate] = build_experiment_candidates(
            [
                {
                    "id": "ke_metric",
                    "claim": {
                        "statement": "VRAMを40%削減",
                        "tags": ["performance", "reconstruction"],
                    },
                    "applies_to": {"repositories": ["KAFKA2306/AutoPhotogrammetry"]},
                    "evidence": {"status": "CLAIMED", "numeric_claim_present": True},
                }
            ]
        )
        self.assertTrue(candidate["source_numeric_claim"]["present"])
        self.assertFalse(candidate["source_numeric_claim"]["may_populate_actual_metrics"])
        self.assertTrue(candidate["verification_contract"]["one_variable_only"])

    def test_writes_machine_readable_candidate_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            output = root / "candidates.json"
            events.write_text(
                json.dumps(
                    {
                        "id": "ke_x",
                        "claim": {"statement": "COLMAP pose candidate", "tags": ["reconstruction"]},
                        "applies_to": {"repositories": ["KAFKA2306/AutoPhotogrammetry"]},
                        "evidence": {"status": "CLAIMED", "numeric_claim_present": False},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = write_experiment_candidates(events, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(payload["candidates"][0]["status"], "CANDIDATE")


if __name__ == "__main__":
    unittest.main()
