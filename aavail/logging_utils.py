"""Structured, non-PII JSONL audit logging to an injectable path."""
from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_event_logger(log_path: str | Path) -> logging.Logger:
    """Create a file-only logger. Tests pass a tmp_path; no production files touched."""
    log_path=Path(log_path).resolve()
    log_path.parent.mkdir(parents=True,exist_ok=True)
    name=f'aavail.events.{log_path}'
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate=False
    if not logger.handlers:
        handler=RotatingFileHandler(log_path,maxBytes=1_000_000,backupCount=2,encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """Do not pass customer names, raw payloads, or unredacted IDs to fields."""
    from datetime import datetime,timezone
    record={'ts':datetime.now(timezone.utc).isoformat(),'event':str(event), **fields}
    logger.info(json.dumps(record,ensure_ascii=True,allow_nan=False,default=str))


def read_jsonl(log_path: str | Path) -> list[dict]:
    path=Path(log_path)
    if not path.exists():
        return []
    out=[]
    for raw in path.read_text(encoding='utf-8').splitlines():
        if raw.strip():
            out.append(json.loads(raw))
    return out
