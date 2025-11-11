# app/core/video.py
from __future__ import annotations
import io
import math
import cv2
import numpy as np
from typing import List, Tuple

def iter_video_frames_from_bytes(video_bytes: bytes,
                                 sample_every: int = 2,
                                 max_frames: int = 180) -> List[np.ndarray]:
    """
    Decode video bytes with OpenCV and yield BGR frames.
    sample_every=2 -> keep every 2nd frame
    max_frames is the cap before sampling (for safety)
    """
    # Write bytes to a memory buffer (OpenCV prefers filenames; we use cv2.imdecode for images,
    # but for video we use a temp memory file via cv2.VideoCapture with a named pipe-like trick)
    # Simplest cross-platform: write to a tmp file-like buffer on disk.
    # However, to avoid deps, we use a simple approach: save to /tmp then delete.
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    idx = 0
    kept = 0
    try:
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                frames.append(frame)
                kept += 1
            idx += 1
    finally:
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return frames

def pick_diverse_indices(embs: np.ndarray,
                         max_keep: int = 12,
                         dedupe_cosine: float = 0.95) -> List[int]:
    """
    Greedy selection: start with the sharpest/best order externally,
    drop near-duplicates (> dedupe_cosine), keep up to max_keep.
    Expects embs L2-normalized.
    """
    keep: List[int] = []
    for i in range(embs.shape[0]):
        if len(keep) == 0:
            keep.append(i); continue
        sim = (embs[i][None, :] @ embs[keep].T).max()
        if sim < dedupe_cosine:
            keep.append(i)
        if len(keep) >= max_keep:
            break
    return keep
