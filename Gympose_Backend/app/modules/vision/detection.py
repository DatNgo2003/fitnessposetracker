from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

from app.core.config import get_settings


_logger = logging.getLogger(__name__)


class PersonDetector:
    _initialized: bool = False

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        # Import here to avoid slow startup
        from ultralytics import YOLO

        device = "cuda" if self._cuda_available() else "cpu"
        _logger.info(f"Initializing YOLOv8 Person Detector on {device}...")

        # Path to YOLOv8 model weights
        # detection.py is at /app/app/modules/vision/detection.py
        # Model is at /app/models/yolov8n.pt
        model_path = os.path.join(
            os.path.dirname(__file__), "../../../models/yolov8m.pt"
        )

        if not os.path.exists(model_path):
            _logger.warning(
                f"Model not found at {model_path}, using default 'yolov8m.pt'"
            )
            model_path = "yolov8m.pt"  # Will auto-download

        self._model = YOLO(model_path)
        self._model.to(device)
        _logger.info(f"YOLOv8 loaded successfully from {model_path}")
        self._initialized = True

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def detect_persons(
        self, image_bgr: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect persons in image using YOLOv8.

        Args:
            image_bgr: Input image in BGR format (OpenCV)

        Returns:
            List of (x1, y1, x2, y2, score) for 'person' class only.
        """
        self._lazy_init()
        assert self._model is not None

        threshold = float(self.settings.detection_score_threshold)

        # Run YOLOv8 inference
        # verbose=False to reduce logging
        results = self._model(image_bgr, verbose=False, conf=threshold)

        detections: List[Tuple[float, float, float, float, float]] = []

        if not results or len(results) == 0:
            return detections

        # Get first result (single image)
        result = results[0]

        # Filter for person class (class_id=0 in COCO)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            # Check if class is person (0 in COCO)
            class_id = int(box.cls[0])
            if class_id != 0:  # 0 = person in COCO
                continue

            # Get bbox coordinates and confidence
            xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            conf = float(box.conf[0])

            if conf >= threshold:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                detections.append((x1, y1, x2, y2, conf))

        return detections
