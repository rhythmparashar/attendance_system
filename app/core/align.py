import numpy as np
import cv2
from app import config

REF_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_5pts(img_bgr: np.ndarray, kps: np.ndarray, out_size=config.ALIGNED_SIZE):
    assert kps.shape == (5, 2)
    dst = REF_5PTS.copy()
    if out_size != 112:
        dst *= out_size / 112.0
    M = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img_bgr, M, (out_size, out_size), borderValue=0.0)
