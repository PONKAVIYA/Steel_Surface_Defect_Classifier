"""One inspection = decode -> quality checks -> prediction -> (optional) Grad-CAM -> (optional) save.

Shared by the Streamlit app, the FastAPI service and the hardware adapters so that every entry point
uses exactly the same preprocessing and produces the same traceability record.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image

from src.database.repository import InspectionRecord, InspectionRepository, new_inspection_id
from src.data.preprocessing import load_image
from src.inference.explainability import ExplanationResult
from src.inference.image_quality import QualityReport, assess_image_quality, load_quality_thresholds
from src.inference.predictor import ModelUnavailableError, PredictionResult, Predictor

_EXT = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp"}


@dataclass
class InspectionOutcome:
    inspection_id: str
    created_at: str
    source: str
    device_id: str | None
    original_filename: str | None
    image_bytes: bytes
    image_ext: str
    image_size: tuple[int, int]
    prediction: PredictionResult
    quality: QualityReport
    explanation: ExplanationResult | None
    saved: bool = False

    def summary(self) -> dict:
        p = self.prediction
        return {
            "inspection_id": self.inspection_id,
            "created_at": self.created_at,
            "source": self.source,
            "device_id": self.device_id,
            "is_model_prediction": True,
            "predicted_class": p.class_name,
            "predicted_display_name": p.display_name,
            "confidence": p.confidence,
            "probabilities": p.probabilities,
            "review_needed": p.review_needed,
            "review_threshold": p.review_threshold,
            "review_reason": p.review_reason,
            "quality": self.quality.to_dict(),
            "model_version": p.model_version,
            "preprocessing": p.preprocessing,
            "inference_ms": p.inference_ms,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "saved": self.saved,
        }


class InspectionService:
    def __init__(self, cfg: dict, predictor: Predictor | None, repository: InspectionRepository | None):
        self.cfg = cfg
        self.predictor = predictor
        self.repository = repository
        self.quality_thresholds, self.quality_source = load_quality_thresholds(cfg)

    @property
    def model_available(self) -> bool:
        return self.predictor is not None

    def inspect(self, image_bytes: bytes, source: str, original_filename: str | None = None,
                device_id: str | None = None, explain: bool = True, layer: str | None = None) -> InspectionOutcome:
        if self.predictor is None:
            raise ModelUnavailableError("No trained model is loaded; real classification is unavailable.")
        img: Image.Image = load_image(image_bytes)  # raises ImageLoadError for corrupt/unsupported files
        quality = assess_image_quality(img, self.cfg, self.predictor.input_size, self.quality_thresholds)
        quality.thresholds_source = self.quality_source
        prediction = self.predictor.predict(img)
        explanation = None
        if explain:
            explanation = self.predictor.explain(img, prediction.class_index,
                                                 layer or self.cfg["inference"].get("gradcam_layer", "block4"))
        now = datetime.now(timezone.utc)
        return InspectionOutcome(
            inspection_id=new_inspection_id(now), created_at=now.isoformat(), source=source,
            device_id=device_id, original_filename=original_filename, image_bytes=image_bytes,
            image_ext=_EXT.get(img.format or "", ".png"), image_size=img.size, prediction=prediction,
            quality=quality, explanation=explanation,
        )

    def save(self, outcome: InspectionOutcome, operator_note: str = "") -> InspectionRecord:
        if self.repository is None:
            raise RuntimeError("No inspection repository configured.")
        p = outcome.prediction
        record = InspectionRecord(
            inspection_id=outcome.inspection_id, created_at=outcome.created_at, source=outcome.source,
            predicted_class=p.class_name, predicted_index=p.class_index, confidence=p.confidence,
            probabilities=p.probabilities, review_needed=p.review_needed, review_threshold=p.review_threshold,
            model_version=p.model_version, preprocessing=p.preprocessing,
            quality=outcome.quality.to_dict(), quality_suitable=outcome.quality.suitable,
            inference_ms=p.inference_ms, operator_note=(operator_note or "").strip(),
            device_id=outcome.device_id, original_filename=outcome.original_filename,
            image_width=outcome.image_size[0], image_height=outcome.image_size[1],
        )
        saved = self.repository.save(record, outcome.image_bytes, outcome.image_ext)
        outcome.saved = True
        return saved
