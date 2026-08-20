import subprocess
import unittest
from unittest.mock import patch

from processing.gpu_memory import ComputeMemoryMonitor, MIB, query_compute_memory


class GpuMemoryTests(unittest.TestCase):
    def test_query_compute_memory_sums_process_rows(self):
        completed = subprocess.CompletedProcess(
            ["nvidia-smi"], 0, "101, 256\n202, 1024\n", ""
        )
        with patch("processing.gpu_memory.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "processing.gpu_memory.subprocess.run", return_value=completed
        ):
            result = query_compute_memory()
        self.assertEqual(result, {"process_count": 2, "total_bytes": 1280 * MIB})

    def test_query_compute_memory_returns_none_when_tool_is_missing(self):
        with patch("processing.gpu_memory.shutil.which", return_value=None):
            self.assertIsNone(query_compute_memory())

    def test_monitor_reports_baseline_delta(self):
        samples = [
            {"process_count": 1, "total_bytes": 100 * MIB},
            {"process_count": 2, "total_bytes": 900 * MIB},
        ]
        with patch("processing.gpu_memory.query_compute_memory", side_effect=samples):
            with ComputeMemoryMonitor(interval_seconds=60) as monitor:
                pass
        measurement = monitor.measurement()
        self.assertEqual(measurement.status, "measured")
        self.assertEqual(measurement.baseline_bytes, 100 * MIB)
        self.assertEqual(measurement.peak_delta_bytes, 800 * MIB)
        self.assertEqual(measurement.max_process_count, 2)

    def test_monitor_keeps_peak_null_when_query_is_unavailable(self):
        with patch("processing.gpu_memory.query_compute_memory", return_value=None):
            with ComputeMemoryMonitor(interval_seconds=60) as monitor:
                pass
        measurement = monitor.measurement()
        self.assertEqual(measurement.status, "unavailable")
        self.assertIsNone(measurement.peak_delta_bytes)


if __name__ == "__main__":
    unittest.main()
