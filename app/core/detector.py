# app/core/detector.py
import logging
import numpy as np
from pathlib import Path
import inspect

from insightface.model_zoo.scrfd import SCRFD
from app import config

log = logging.getLogger(__name__)

class FaceDetector:
    """SCRFD detector (handles different insightface signatures)."""

    def __init__(self):
        model_path = Path(config.SCRFD_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SCRFD model not found at {model_path.resolve()}. "
                "Place 'scrfd_2.5g_bnkps.onnx' in app/models/ or set SCRFD_MODEL env."
            )
        self.detector = SCRFD(str(model_path))
        self.detector.prepare(ctx_id=-1, input_size=(640, 640))

        # Set thresholds where supported
        # Some versions expose .det_thresh / .nms_thresh attributes.
        if hasattr(self.detector, "det_thresh"):
            self.detector.det_thresh = config.DET_SCORE_THRESH
        if hasattr(self.detector, "nms_thresh"):
            self.detector.nms_thresh = config.DET_NMS_THRESH

        # Cache detect() signature to decide which kwargs are allowed
        self._detect_sig = inspect.signature(self.detector.detect)
        log.info("SCRFD initialized", extra={"model": str(model_path.resolve())})

    def _detect_kwargs(self):
        """Return a dict of kwargs compatible with this SCRFD.detect()."""
        params = self._detect_sig.parameters
        kw = {}
        if "max_num" in params:
            kw["max_num"] = config.MAX_FACES
        # Newer insightface accepts 'thresh' and sometimes 'metric'
        if "thresh" in params:
            kw["thresh"] = config.DET_SCORE_THRESH
        if "metric" in params:
            kw["metric"] = "default"
        return kw

    def detect(self, img_bgr: np.ndarray):
        try:
            bboxes, kps = self.detector.detect(img_bgr, **self._detect_kwargs())
        except TypeError:
            # Fallback: call with no kwargs
            bboxes, kps = self.detector.detect(img_bgr)

        if bboxes is None or len(bboxes) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 5, 2), dtype=np.float32))
        return bboxes.astype(np.float32), kps.astype(np.float32)
