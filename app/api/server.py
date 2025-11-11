import io, logging, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from app.logging_config import setup_logging
from app.core.recognize import Pipeline
from fastapi import UploadFile

setup_logging()
app = FastAPI(title="Attendance Face API")
PIPE = Pipeline()
log = logging.getLogger(__name__)

def _read_img(file: UploadFile):
    arr = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image upload")
    return img

@app.get("/healthz")
def healthz():
    return {"status": "ok", "people": PIPE.db.people()}

@app.post("/enroll")
async def enroll(person_id: str = Query(...), files: list[UploadFile] = File(...)):
    total_faces = total_emb = 0
    for f in files:
        try:
            img = _read_img(f)
            res = PIPE.enroll_image(img, person_id, f.filename)
            total_faces += res["faces"]
            total_emb += res["embeddings_added"]
        except Exception as e:
            log.exception("enroll failed", extra={"file": f.filename})
            return JSONResponse(status_code=400, content={"error": str(e)})
    return {"person_id": person_id, "faces": total_faces, "embeddings": total_emb}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        img = _read_img(file)
        preds = PIPE.recognize_image(img)
        return {"faces": preds}
    except Exception as e:
        log.exception("recognize failed")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/enroll/video")
async def enroll_video(person_id: str = Query(..., description="Employee/Person ID"),
                       video: UploadFile = File(...)):
    """
    Accepts a short video (mp4/webm). Samples frames, selects best faces,
    embeds, dedupes and stores up to ~12 templates.
    """
    try:
        raw = video.file.read()
        if not raw:
            return JSONResponse(status_code=400, content={"error": "Empty video upload"})
        res = PIPE.enroll_video(raw, person_id=person_id, source=video.filename or "video")
        return {"person_id": person_id, **res}
    except Exception as e:
        log.exception("enroll_video failed", extra={"file": video.filename})
        return JSONResponse(status_code=400, content={"error": str(e)})