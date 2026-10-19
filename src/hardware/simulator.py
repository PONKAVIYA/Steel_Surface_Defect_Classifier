"""Simulated camera for integration tests and demos without the ECE hardware.

Every image it produces is labelled source="simulated_hardware" so it can never be mistaken for a
real device reading. It replays image files from a folder (the bundled demo samples by default) and
can inject faults (timeout, corrupted frame, disconnected device) to exercise error handling.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path

from src.data.preprocessing import SUPPORTED_EXTENSIONS
from src.hardware.interface import (
    CapturedImage,
    HardwareError,
    HardwareNotConnected,
    HardwareTimeout,
    ImageSourceAdapter,
)

FAULTS = ("timeout", "corrupt_frame", "disconnect")


class SimulatedCameraAdapter(ImageSourceAdapter):
    source_type = "simulated_hardware"

    def __init__(self, image_folder: str | Path, device_id: str = "SIM-CAM-01", timeout_s: float = 5.0):
        super().__init__(device_id, timeout_s)
        self.folder = Path(image_folder)
        self._files: list[Path] = []
        self._cycle = None
        self._fault_queue: list[str] = []
        self.sent_results: list[dict] = []
        self.frame_counter = 0

    def connect(self) -> None:
        if not self.folder.is_dir():
            raise HardwareError(f"Simulator image folder not found: {self.folder}")
        self._files = sorted(p for p in self.folder.rglob("*")
                             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not self._files:
            raise HardwareError(f"Simulator image folder {self.folder} contains no images")
        self._cycle = itertools.cycle(self._files)
        self.connected = True

    def inject_fault(self, fault: str) -> None:
        if fault not in FAULTS:
            raise ValueError(f"Unknown fault {fault}; options {FAULTS}")
        self._fault_queue.append(fault)

    def acquire(self, timeout_s: float | None = None) -> CapturedImage:
        if not self.connected:
            raise HardwareNotConnected("Simulated camera is not connected; call connect() first.")
        if self._fault_queue:
            fault = self._fault_queue.pop(0)
            if fault == "timeout":
                raise HardwareTimeout(f"Simulated timeout: no frame within {timeout_s or self.timeout_s:.1f}s")
            if fault == "disconnect":
                self.connected = False
                raise HardwareNotConnected("Simulated device disconnected")
            if fault == "corrupt_frame":
                self.frame_counter += 1
                return CapturedImage(b"\x89PNG\r\n\x1a\nCORRUPTED", f"sim_frame_{self.frame_counter:05d}.png",
                                     self.source_type, self.device_id, datetime.now(timezone.utc).isoformat(),
                                     {"simulated": True, "fault": "corrupt_frame"})
        path = next(self._cycle)
        self.frame_counter += 1
        return CapturedImage(path.read_bytes(), f"sim_frame_{self.frame_counter:05d}{path.suffix.lower()}",
                             self.source_type, self.device_id, datetime.now(timezone.utc).isoformat(),
                             {"simulated": True, "replayed_file": path.name})

    def send_result(self, capture: CapturedImage, message: dict) -> None:
        self.sent_results.append(message)

    def status(self) -> dict:
        s = super().status()
        s.update({"image_folder": str(self.folder), "n_images": len(self._files), "simulated": True})
        return s
