from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass

MIB = 1024 * 1024


def query_compute_memory() -> dict | None:
    """Return aggregate NVIDIA compute-process memory without inventing a value.

    The quality runner executes in one GPU container, but the CUDA context used by the
    parent process can already consume memory before ns-train starts.  We therefore
    measure aggregate compute-process memory and report a delta from the immediately
    preceding baseline rather than total device memory.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None

    total_mib = 0
    process_count = 0
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            return None
        try:
            int(parts[0])
            used_mib = int(parts[1])
        except ValueError:
            return None
        if used_mib < 0:
            return None
        process_count += 1
        total_mib += used_mib
    return {
        "process_count": process_count,
        "total_bytes": total_mib * MIB,
    }


@dataclass
class ComputeMemoryMeasurement:
    status: str
    method: str
    baseline_bytes: int | None
    baseline_process_count: int | None
    peak_total_bytes: int | None
    peak_delta_bytes: int | None
    max_process_count: int | None
    samples: int

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "baseline_bytes": self.baseline_bytes,
            "baseline_process_count": self.baseline_process_count,
            "peak_total_bytes": self.peak_total_bytes,
            "peak_delta_bytes": self.peak_delta_bytes,
            "max_process_count": self.max_process_count,
            "samples": self.samples,
        }


class ComputeMemoryMonitor:
    """Sample aggregate NVIDIA compute VRAM while an external GPU command runs."""

    def __init__(self, *, interval_seconds: float = 0.2):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline: dict | None = None
        self._peak_total: int | None = None
        self._max_process_count: int | None = None
        self._samples = 0
        self._query_failed = False

    def _sample(self) -> None:
        sample = query_compute_memory()
        if sample is None:
            self._query_failed = True
            return
        self._samples += 1
        total = int(sample["total_bytes"])
        count = int(sample["process_count"])
        if self._peak_total is None or total > self._peak_total:
            self._peak_total = total
        if self._max_process_count is None or count > self._max_process_count:
            self._max_process_count = count

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "ComputeMemoryMonitor":
        self._baseline = query_compute_memory()
        if self._baseline is not None:
            self._peak_total = int(self._baseline["total_bytes"])
            self._max_process_count = int(self._baseline["process_count"])
            self._samples = 1
            self._thread = threading.Thread(target=self._loop, name="gpu-memory-monitor", daemon=True)
            self._thread.start()
        else:
            self._query_failed = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._baseline is not None:
            self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))

    def measurement(self) -> ComputeMemoryMeasurement:
        if self._baseline is None or self._query_failed or self._peak_total is None:
            return ComputeMemoryMeasurement(
                status="unavailable",
                method="nvidia-smi-compute-process-baseline-delta",
                baseline_bytes=(None if self._baseline is None else int(self._baseline["total_bytes"])),
                baseline_process_count=(
                    None if self._baseline is None else int(self._baseline["process_count"])
                ),
                peak_total_bytes=self._peak_total,
                peak_delta_bytes=None,
                max_process_count=self._max_process_count,
                samples=self._samples,
            )
        baseline = int(self._baseline["total_bytes"])
        return ComputeMemoryMeasurement(
            status="measured",
            method="nvidia-smi-compute-process-baseline-delta",
            baseline_bytes=baseline,
            baseline_process_count=int(self._baseline["process_count"]),
            peak_total_bytes=self._peak_total,
            peak_delta_bytes=max(0, self._peak_total - baseline),
            max_process_count=self._max_process_count,
            samples=self._samples,
        )
