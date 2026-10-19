"""Hardware integration boundary between the ECE acquisition system and this software.

The physical camera, lighting, controller and transport are NOT known yet; nothing here assumes a
particular microcontroller, camera or protocol. The contract is deliberately small:

  ECE system  --(image + metadata)-->  ImageSourceAdapter.acquire()  -->  InspectionService
  ECE system  <--(result message)---   ImageSourceAdapter.send_result()

Two adapters ship today:
  * SimulatedCameraAdapter (src/hardware/simulator.py) - for tests/demos; marked "simulated_hardware".
  * FolderDropAdapter (below) - the device (or its host PC) writes image files into a drop folder and
    reads a JSON result written next to each processed file. Works with USB cameras on a host PC,
    network shares, or a Raspberry Pi syncing files, without any protocol decision.
The HTTP API in src/api/service.py is the alternative push-style transport (POST an image, get JSON).
A serial/USB adapter can be added by subclassing ImageSourceAdapter once the ECE team fixes the protocol.
See docs/HARDWARE_INTERFACE.md for message formats, timeouts and error codes.
"""
from __future__ import annotations

import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.data.preprocessing import SUPPORTED_EXTENSIONS

RESULT_MESSAGE_VERSION = "1.0"


class HardwareError(RuntimeError):
    code = "HARDWARE_ERROR"


class HardwareTimeout(HardwareError):
    code = "ACQUISITION_TIMEOUT"


class HardwareNotConnected(HardwareError):
    code = "DEVICE_NOT_CONNECTED"


@dataclass
class CapturedImage:
    image_bytes: bytes
    filename: str
    source: str            # "hardware" (real device) or "simulated_hardware"
    device_id: str
    captured_at: str       # ISO-8601 UTC, from the device clock when provided
    metadata: dict = field(default_factory=dict)


def build_result_message(summary: dict, capture: CapturedImage | None = None, error: dict | None = None) -> dict:
    """JSON message returned to the device/operator interface (see docs/HARDWARE_INTERFACE.md)."""
    msg = {
        "message_version": RESULT_MESSAGE_VERSION,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "device_id": capture.device_id if capture else summary.get("device_id"),
        "capture_filename": capture.filename if capture else None,
        "status": "error" if error else "ok",
    }
    if error:
        msg["error"] = error
        return msg
    msg.update({
        "inspection_id": summary["inspection_id"],
        "is_model_prediction": True,
        "predicted_class": summary["predicted_class"],
        "predicted_display_name": summary["predicted_display_name"],
        "confidence": round(float(summary["confidence"]), 6),
        "review_needed": bool(summary["review_needed"]),
        "image_quality_ok": bool(summary["quality"]["suitable"]),
        "quality_warnings": [i["code"] for i in summary["quality"]["issues"] if i["severity"] == "warning"],
        "model_version": summary["model_version"],
        "simulated": capture.source == "simulated_hardware" if capture else summary.get("source") == "simulated_hardware",
    })
    return msg


class ImageSourceAdapter(ABC):
    source_type = "hardware"

    def __init__(self, device_id: str, timeout_s: float = 5.0):
        self.device_id = device_id
        self.timeout_s = float(timeout_s)
        self.connected = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def acquire(self, timeout_s: float | None = None) -> CapturedImage:
        """Block until one image is available or raise HardwareTimeout."""

    def send_result(self, capture: CapturedImage, message: dict) -> None:
        """Return the result to the device. Default: nothing to send (operator reads the UI)."""

    def close(self) -> None:
        self.connected = False

    def status(self) -> dict:
        return {"adapter": type(self).__name__, "device_id": self.device_id, "connected": self.connected,
                "source_type": self.source_type, "timeout_s": self.timeout_s}


class FolderDropAdapter(ImageSourceAdapter):
    """Picks up image files the acquisition system writes into `drop_folder`.

    Protocol: the device writes `<name>.tmp` and renames it to `<name>.png|.jpg|.bmp` when complete
    (atomic rename avoids reading half-written files). Optional `<name>.json` metadata may accompany it.
    After processing, the image moves to `processed/` and `<name>.result.json` is written to `results/`.
    """
    source_type = "hardware"

    def __init__(self, drop_folder: str | Path, device_id: str, timeout_s: float = 5.0, poll_s: float = 0.2):
        super().__init__(device_id, timeout_s)
        self.drop = Path(drop_folder)
        self.poll_s = poll_s

    def connect(self) -> None:
        for sub in ("", "processed", "results"):
            (self.drop / sub).mkdir(parents=True, exist_ok=True)
        self.connected = True

    def _pending(self) -> list[Path]:
        return sorted((p for p in self.drop.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                      key=lambda p: p.stat().st_mtime)

    def acquire(self, timeout_s: float | None = None) -> CapturedImage:
        if not self.connected:
            raise HardwareNotConnected("Drop-folder adapter is not connected; call connect() first.")
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        while True:
            files = self._pending()
            if files:
                path = files[0]
                meta_path = path.with_suffix(".json")
                meta = {}
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        meta = {"metadata_error": "unreadable metadata file"}
                data = path.read_bytes()
                dest = self.drop / "processed" / path.name
                shutil.move(str(path), dest)
                if meta_path.is_file():
                    shutil.move(str(meta_path), self.drop / "processed" / meta_path.name)
                captured = meta.get("captured_at") or datetime.fromtimestamp(
                    dest.stat().st_mtime, timezone.utc).isoformat()
                return CapturedImage(data, path.name, self.source_type, meta.get("device_id", self.device_id),
                                     captured, meta)
            if time.monotonic() >= deadline:
                raise HardwareTimeout(f"No image arrived in {self.drop} within the timeout.")
            time.sleep(self.poll_s)

    def send_result(self, capture: CapturedImage, message: dict) -> None:
        out = self.drop / "results" / (Path(capture.filename).stem + ".result.json")
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(message, indent=2), encoding="utf-8")
        tmp.replace(out)

    def status(self) -> dict:
        s = super().status()
        s.update({"drop_folder": str(self.drop),
                  "pending_files": len(self._pending()) if self.connected and self.drop.is_dir() else None})
        return s


def build_adapter(cfg: dict) -> ImageSourceAdapter | None:
    """Create the adapter selected in config `hardware.adapter` (None in standalone mode)."""
    from src.config import resolve_path
    from src.hardware.simulator import SimulatedCameraAdapter

    h = cfg["hardware"]
    kind = h.get("adapter", "standalone")
    if kind == "standalone":
        return None
    if kind == "simulated":
        return SimulatedCameraAdapter(resolve_path("samples"), h.get("device_id", "SIM-CAM-01"), h.get("timeout_s", 5))
    if kind == "folder":
        return FolderDropAdapter(resolve_path(h["drop_folder"]), h.get("device_id", "CAM-01"), h.get("timeout_s", 5))
    raise ValueError(f"Unknown hardware adapter '{kind}' (standalone | simulated | folder)")
