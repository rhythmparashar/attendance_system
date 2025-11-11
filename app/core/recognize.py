import logging
from app.core.detector import FaceDetector
from app.core.align import align_5pts
from app.core.embedder import FaceEmbedder
from app.core.storage import LocalStore
from app import config
from app.core.video import iter_video_frames_from_bytes, pick_diverse_indices
from app.core.quality import face_quality_scores, passes_quality
import numpy as np


log = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.det = FaceDetector()
        self.emb = FaceEmbedder()
        self.db = LocalStore()

    def enroll_image(self, img, person_id, source):
        bboxes, kps = self.det.detect(img)
        aligned = [align_5pts(img, k) for k in kps]
        for a in aligned:
            self.db.save_aligned(person_id, a)
        embs = self.emb.embed(aligned)
        added = self.db.add_embeddings(person_id, embs, source)
        return {"faces": len(aligned), "embeddings_added": added}

    def recognize_image(self, img):
        bboxes, kps = self.det.detect(img)
        aligned = [align_5pts(img, k) for k in kps]
        embs = self.emb.embed(aligned)
        sims = self.db.search_cosine(embs, top_k=config.TOP_K)
        out = []
        for i in range(len(bboxes)):
            best = sims[i][0] if sims[i] else {"person_id": "unknown", "score": 0.0}
            label = best["person_id"] if best["score"] >= config.SIMILARITY_THRESHOLD else "unknown"
            out.append({
                "bbox": bboxes[i][:4].tolist(),
                "score": float(bboxes[i][4]),
                "prediction": {"person_id": label, "similarity": best["score"]},
                "top_k": sims[i],
            })
        return out
    
    def enroll_video(self, video_bytes: bytes, person_id: str, source: str,
                     sample_every: int = 2,
                     min_face_size_px: int = 140,
                     max_frames_considered: int = 180,
                     max_templates: int = 12) -> dict:
        # 1) decode & sample frames
        frames = iter_video_frames_from_bytes(video_bytes, sample_every=sample_every,
                                              max_frames=max_frames_considered)
        if not frames:
            return {"faces": 0, "embeddings_added": 0, "note": "no frames decoded"}

        aligned_faces = []
        qinfo = []  # quality info for ranking
        total_detected = 0

        # 2) detect & align
        for fr in frames:
            bboxes, kps = self.det.detect(fr)
            if bboxes.shape[0] == 0:
                continue
            # pick largest face only (assuming single subject during enroll)
            areas = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
            idx = int(np.argmax(areas))
            x1,y1,x2,y2,score = bboxes[idx]
            if min(x2-x1, y2-y1) < min_face_size_px:
                continue
            a = align_5pts(fr, kps[idx])
            q = face_quality_scores(a)
            if not passes_quality(q):
                continue
            aligned_faces.append(a)
            qinfo.append(q)
            total_detected += 1

        if not aligned_faces:
            return {"faces": 0, "embeddings_added": 0, "note": "no usable faces passed quality"}

        # 3) embed all kept faces
        embs = self.emb.embed(aligned_faces)   # (N, D), L2-normalized already
        # Rank by simple composite: sharpness primary, brightness closeness secondary
        ranked = sorted(range(len(aligned_faces)),
                        key=lambda i: (qinfo[i]["sharpness"], -abs(qinfo[i]["brightness"]-128)),
                        reverse=True)

        embs_ranked = embs[ranked]
        idx_keep = pick_diverse_indices(embs_ranked, max_keep=max_templates, dedupe_cosine=0.95)

        final_embs = embs_ranked[idx_keep]
        final_faces = [aligned_faces[ranked[i]] for i in idx_keep]

        # 4) persist crops (optional) + embeddings
        for a in final_faces:
            self.db.save_aligned(person_id, a)
        added = self.db.add_embeddings(person_id, final_embs, source)

        return {
            "frames_total": len(frames),
            "frames_detected": total_detected,
            "faces_kept": len(final_faces),
            "embeddings_added": int(added),
        }
