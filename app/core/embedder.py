import logging
import numpy as np
import cv2
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from app import config

log = logging.getLogger(__name__)

class FaceEmbedder:
    """MobileFaceNet ONNX wrapper."""

    def __init__(self):
        model = config.MOBILEFACENET_MODEL
        self.arc = ArcFaceONNX(str(model))
        self.arc.prepare(ctx_id=-1)
        log.info("MobileFaceNet initialized", extra={"model": str(model)})

    def embed(self, aligned_faces):
        feats = []
        for img in aligned_faces:
            if img.shape[:2] != (config.ALIGNED_SIZE, config.ALIGNED_SIZE):
                img = cv2.resize(img, (config.ALIGNED_SIZE, config.ALIGNED_SIZE))
            feat = self.arc.get_feat(img)
            feat = feat / (np.linalg.norm(feat) + 1e-12)
            feats.append(feat.astype(np.float32))
        if not feats:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)
        return np.vstack(feats)
