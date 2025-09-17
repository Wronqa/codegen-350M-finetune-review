from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeVar

import yaml

try:
    import numpy as np  
except Exception:
    np = None  

try:
    import torch  
except Exception:
    torch = None  

from .schema import PreparedRecord

T = TypeVar("T")

def setup_logger(name: str = "dataprep") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def load_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.getLogger("dataprep").warning("Bad JSON at %s:%d (%s)", p, i, e)
    return out

def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> int:
    return write_jsonl(path, records)

def sample_cap(items: List[T], cap: Optional[int], *, seed: Optional[int] = None) -> List[T]:
    if cap is None or cap <= 0 or len(items) <= cap:
        return items
    rnd = random.Random(seed)
    return rnd.sample(items, cap)

def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed) 
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(seed)  
            if torch.cuda.is_available():  
                torch.cuda.manual_seed_all(seed)  
        except Exception:
            pass

def clean_one_line(text: str, limit: int = 300) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t[:limit].rstrip()

def looks_like_python(code: str) -> bool:
    if not code.strip():
        return False
    
    patterns = [
        r"^\s*def\s+\w+\(.*\):",        
        r"^\s*class\s+\w+\(?.*?\)?:",     
        r"^\s*import\s+\w+",              
        r"^\s*from\s+\w+(\.\w+)*\s+import",
        r"^\s*if\s+.*:\s*$",               
        r"^\s*for\s+\w+\s+in\s+.*:\s*$",   
        r"^\s*while\s+.*:\s*$",          
        r"self",                          
        r"f\".*{.*}.*\"",                 
    ]
    
    score = sum(bool(re.search(p, code, re.MULTILINE)) for p in patterns)
    
    return score >= 1

def to_prepared(code: str, comment: str, max_resp_chars: int) -> Optional[PreparedRecord]:
    if not code or not comment:
        return None
    if not looks_like_python(code):
        return None
    response = clean_one_line(comment, limit=max_resp_chars)
    if not response:
        return None
    return PreparedRecord(
        prompt=PreparedRecord.build_prompt(code),
        response=response,
    )