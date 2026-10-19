"""Acquire images from the configured hardware adapter, classify them and record each inspection.

Usage:
    python scripts/hardware_loop.py --adapter simulated --count 5      # no device needed
    python scripts/hardware_loop.py --adapter folder --count 0         # run until Ctrl+C, watching the drop folder

Each result is also returned to the device through the adapter (folder adapter: results/<name>.result.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.preprocessing import ImageLoadError  # noqa: E402
from src.database.repository import InspectionRepository  # noqa: E402
from src.hardware.interface import (HardwareError, HardwareNotConnected, HardwareTimeout,  # noqa: E402
                                    build_adapter, build_result_message)
from src.inference.predictor import ModelUnavailableError, Predictor  # noqa: E402
from src.inference.service import InspectionService  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("hardware_loop")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", choices=["simulated", "folder"], default=None)
    ap.add_argument("--count", type=int, default=5, help="images to process (0 = until interrupted)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, {"hardware": {"adapter": args.adapter}} if args.adapter else None)
    try:
        predictor = Predictor.from_checkpoint(resolve_path(cfg["inference"]["model_path"]),
                                              cfg["inference"].get("review_threshold"))
    except ModelUnavailableError as exc:
        log.error("Cannot classify: %s", exc)
        return 2
    service = InspectionService(cfg, predictor, InspectionRepository(resolve_path(cfg["paths"]["inspections_dir"])))
    adapter = build_adapter(cfg)
    if adapter is None:
        log.error("hardware.adapter is 'standalone'; choose --adapter simulated or folder")
        return 2
    adapter.connect()
    log.info("Adapter: %s", json.dumps(adapter.status()))
    done = 0
    try:
        while args.count == 0 or done < args.count:
            try:
                cap = adapter.acquire()
            except HardwareTimeout as exc:
                log.info("%s (waiting)", exc)
                continue
            except HardwareNotConnected as exc:
                log.error("%s", exc)
                return 1
            try:
                outcome = service.inspect(cap.image_bytes, cap.source, cap.filename, cap.device_id, explain=False)
            except ImageLoadError as exc:
                adapter.send_result(cap, build_result_message({}, cap, {"code": "IMAGE_DECODE_ERROR",
                                                                         "message": str(exc)}))
                log.warning("Frame %s rejected: %s", cap.filename, exc)
                done += 1
                continue
            if not args.no_save:
                service.save(outcome, f"captured_at={cap.captured_at}")
            msg = build_result_message(outcome.summary(), cap)
            adapter.send_result(cap, msg)
            log.info("%s %s -> %s (%.3f)%s%s", outcome.inspection_id, cap.filename, msg["predicted_display_name"],
                     msg["confidence"], " REVIEW" if msg["review_needed"] else "",
                     " [simulated]" if msg["simulated"] else "")
            done += 1
    except KeyboardInterrupt:
        log.info("Stopped by user")
    except HardwareError as exc:
        log.error("Hardware error: %s", exc)
        return 1
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
