# Hardware interface (IT ↔ ECE)

This document defines how images get from the ECE acquisition setup (camera, lighting, controller) into the classifier and how results go back. **No camera, microcontroller or protocol has been confirmed yet**, so the software defines a small contract and ships two working transports plus a simulator. The ECE team picks a transport; nothing else in the software changes.

## Status

| Item | Status |
|---|---|
| Standalone mode (uploads, demo samples) | Implemented and tested |
| Simulated camera (`SimulatedCameraAdapter`) | Implemented and tested; all frames recorded as `simulated_hardware` |
| Drop-folder transport (`FolderDropAdapter`) | Implemented and tested with files written by the test suite |
| HTTP transport (`POST /v1/inspect`) | Implemented and tested with an in-process client |
| Real camera / device | **Not connected. Not tested.** Waiting for ECE hardware decisions |
| Serial / USB-CDC transport | Not implemented (add a subclass of `ImageSourceAdapter` once the protocol is fixed) |

## 1. Image requirements

| Property | Required | Why |
|---|---|---|
| Format | PNG or BMP preferred (lossless); JPEG accepted | The provided training images are JPEG; avoid adding more compression artefacts than they have |
| Channels | 8-bit grayscale preferred; RGB accepted and converted with ITU-R 601 luma | The model is grayscale-only |
| Size | Square patch, ideally **200 × 200 px** | NEU training patches are 200 × 200; the model resizes to 64 × 64 (or the deployed input size) |
| Field of view | Similar magnification to NEU patches (a defect occupies a noticeable part of the patch) | Very large frames shrink defects below what 64 px can resolve. Crop or tile large frames on the device |
| Exposure | Mean brightness within the training range (about 38–240 / 255); avoid large saturated areas | The quality checks warn outside the training range |
| Focus | Sharp; the variance-of-Laplacian check warns on blur | Fine crazing and thin scratches disappear when blurred |
| Max upload | 10 MB (`quality.max_upload_mb`) | Protects the service |

Lighting should be diffuse and stable. Strong colour casts, specular glare or moving shadows are outside the training distribution; the model has no way to compensate for them.

## 2. Transport options

### A. Drop folder (simplest; no protocol decision needed)

Works when the camera is attached to a PC, a Raspberry Pi syncs files over the network, or a shared folder is available.

1. Writer (ECE side) saves `frame_0001.tmp`, then **renames** it to `frame_0001.png` (rename is atomic, so half-written files are never read).
2. Optional metadata file `frame_0001.json` written *before* the rename:
   ```json
   {"device_id": "CAM-01", "captured_at": "2026-10-19T08:15:30Z", "line_speed_mpm": 12.5}
   ```
3. The software (`scripts/hardware_loop.py --adapter folder`) picks up the oldest image, moves it to `processed/`, and writes the result to `results/frame_0001.result.json` (format in section 3).

Config (`configs/default.yaml`):
```yaml
hardware:
  adapter: folder
  drop_folder: data/hardware_inbox
  device_id: CAM-01
  timeout_s: 5.0
```

### B. HTTP on the local network

Start the service: `uvicorn src.api.service:app --host 0.0.0.0 --port 8000` (use `127.0.0.1` unless the device is on another machine; there is no authentication, so only expose it on a trusted lab network).

```
POST /v1/inspect   (multipart/form-data)
  file          image file (required)
  device_id     string (optional)
  source        "hardware" | "simulated_hardware" | "upload"   (default "hardware")
  operator_note string (optional)
  save          true | false  (default true: store the inspection record)
```

Example (any language that can send multipart HTTP works, including Python `requests` on a Raspberry Pi):
```bash
curl -F file=@frame_0001.png -F device_id=CAM-01 http://127.0.0.1:8000/v1/inspect
```
`GET /v1/health` and `GET /v1/model` support connection checks. Interactive schema: `http://127.0.0.1:8000/docs`.

### C. Serial / USB (not implemented)

Microcontrollers without a file system or network stack would stream frames over serial. This needs decisions only the ECE team can make: baud rate, framing (length-prefixed binary is recommended), checksum (CRC-32), image size and pixel format, and flow control. Implement it as a subclass of `ImageSourceAdapter` in `src/hardware/` with `connect()`, `acquire(timeout_s)` and `send_result()`; the rest of the pipeline is unchanged.

## 3. Result message (returned to the device / operator display)

```json
{
  "message_version": "1.0",
  "sent_at": "2026-10-19T08:15:31.204Z",
  "device_id": "CAM-01",
  "capture_filename": "frame_0001.png",
  "status": "ok",
  "inspection_id": "INS-20261019-081531-7F3A2C",
  "is_model_prediction": true,
  "predicted_class": "scratches",
  "predicted_display_name": "Scratches",
  "confidence": 0.973,
  "review_needed": false,
  "image_quality_ok": true,
  "quality_warnings": [],
  "model_version": "steelcnn-...",
  "simulated": false
}
```
The HTTP response additionally contains `probabilities` (all six classes), `review_threshold`, `inference_ms` and `saved`.

`confidence` is the top softmax score, **not** a calibrated probability. `review_needed: true` means a person should look at the surface. Any device action (alarm, marking, sorting) must treat the result as advisory.

## 4. Errors

| Code | HTTP | Meaning | Device action |
|---|---|---|---|
| `IMAGE_DECODE_ERROR` | 400 | Corrupted or truncated image | Re-capture |
| `EMPTY_FILE` | 400 | Zero-byte upload | Re-capture |
| `UNSUPPORTED_FORMAT` | 415 | Not PNG/JPEG/BMP | Fix encoder setting |
| `FILE_TOO_LARGE` | 413 | Exceeds `max_upload_mb` | Crop / compress |
| `INVALID_SOURCE` | 422 | Unknown `source` value | Fix client |
| `MODEL_UNAVAILABLE` | 503 | No trained model loaded; **no prediction made** | Alert operator |
| `DATABASE_ERROR` | 500 | Prediction made but not stored | Alert operator; retry |
| `ACQUISITION_TIMEOUT` | – | No frame within `timeout_s` (adapter) | Check trigger / camera |
| `DEVICE_NOT_CONNECTED` | – | Adapter not connected / device lost | Reconnect |

Error body: `{"error": {"code": "...", "message": "..."}}`. In drop-folder mode, decode errors are written as a result file with `"status": "error"`.

## 5. Timing

* `hardware.timeout_s` (default 5 s) bounds each acquisition wait.
* Measured classification time on the development CPU is a few milliseconds per image (see `TEST_REPORT.md`); the camera trigger and transfer will dominate. Re-measure on the actual host.
* HTTP clients should use a 10 s request timeout and retry once on connection errors.

## 6. Commands back to the device

None are sent today. If the line needs actuation (stop, mark, divert), define it explicitly with the ECE team, keep it advisory with a human in the loop, and add it to `send_result()` of the chosen adapter.

## 7. Questions for the ECE team

1. Camera model, resolution, bit depth, and whether it outputs grayscale.
2. Host: PC, Raspberry Pi, or a microcontroller? (decides drop folder vs HTTP vs serial)
3. Field of view per frame in mm, so we can match NEU patch scale (crop/tile on the device if larger).
4. Lighting: type, angle, and whether it is stable between frames.
5. Trigger: free-running, encoder, or manual button?
6. Should results drive any output (LED, buzzer, display), and what latency is acceptable?
7. Will labelled images from the real setup be collected? They are needed to measure accuracy on your hardware; NEU accuracy does not transfer automatically.

## 8. Testing without hardware

```bash
python scripts/hardware_loop.py --adapter simulated --count 5      # replays samples/, records "simulated_hardware"
python -m pytest tests/test_api_or_hardware.py -q                  # API, simulator faults, drop folder
```
In the app, set `hardware.adapter: simulated` and use *Inspect surface → Capture from camera*.
