import logging, pickle, time, uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np, cv2
from app import config

log = logging.getLogger(__name__)

@dataclass
class EmbRecord:
    person_id: str
    embedding: np.ndarray
    ts: float
    source: str

class LocalStore:
    def __init__(self):
        config.FACES_DIR.mkdir(parents=True, exist_ok=True)
        config.EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.index_file = config.EMBED_INDEX_FILE
        self._records = []
        self._matrix = None
        self._person = []
        self._load()

    def _load(self):
        if self.index_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self._records = [
                EmbRecord(r["person_id"], np.asarray(r["embedding"], np.float32), r["ts"], r["source"])
                for r in data
            ]
            log.info("Embeddings loaded", extra={"count": len(self._records)})
        self._rebuild_cache()

    def _rebuild_cache(self):
        if self._records:
            self._matrix = np.vstack([r.embedding for r in self._records])
            self._person = [r.person_id for r in self._records]
        else:
            self._matrix = np.zeros((0, config.EMB_DIM), np.float32)
            self._person = []

    def persist(self):
        with open(self.index_file, "wb") as f:
            pickle.dump([r.__dict__ for r in self._records], f)
        log.info("Embeddings persisted", extra={"count": len(self._records)})

    def save_aligned(self, person_id, aligned_bgr):
        pid_dir = config.FACES_DIR / person_id
        pid_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.jpg"
        out = pid_dir / name
        cv2.imwrite(str(out), aligned_bgr)
        return out

    def add_embeddings(self, person_id, embs, source):
        now = time.time()
        for e in embs:
            self._records.append(EmbRecord(person_id, e, now, source))
        self.persist()
        self._rebuild_cache()
        return embs.shape[0]

    def search_cosine(self, q, top_k):
        """
        Returns, for each query vector, a ranked list of unique persons:
        [{person_id, score}, ...] with score = max cosine among that person's templates.
        """
        if self._matrix is None or len(self._records) == 0:
            return []
        sims = q @ self._matrix.T  # (B, N) with L2-normalized vectors
        results = []
        for i in range(q.shape[0]):
            # collapse by person_id (max similarity)
            agg = {}
            for j, pid in enumerate(self._person):
                s = float(sims[i, j])
                if (pid not in agg) or (s > agg[pid]):
                    agg[pid] = s
            # sort by score desc and take top_k
            ranked = sorted(({"person_id": pid, "score": sc} for pid, sc in agg.items()),
                            key=lambda x: -x["score"])[:top_k]
            results.append(ranked)
        return results

    def people(self):
        counts = {}
        for r in self._records:
            counts[r.person_id] = counts.get(r.person_id, 0) + 1
        return counts

    def delete_person(self, person_id):
        before = len(self._records)
        self._records = [r for r in self._records if r.person_id != person_id]
        pid_dir = config.FACES_DIR / person_id
        if pid_dir.exists():
            for p in pid_dir.glob("*.jpg"):
                p.unlink(missing_ok=True)
            pid_dir.rmdir()
        self.persist()
        self._rebuild_cache()
        return before - len(self._records)
