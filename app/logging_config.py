import logging
import sys

class KVFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        return " | ".join(f"{k}={v}" for k, v in base.items())

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(KVFormatter())

def setup_logging(level=logging.INFO):
    logging.root.handlers.clear()
    logging.root.setLevel(level)
    logging.root.addHandler(_handler)
    for noisy in ["uvicorn", "uvicorn.error", "uvicorn.access", "onnxruntime"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
