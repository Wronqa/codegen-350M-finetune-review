from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Iterable

import pandas as pd

from ..schema import RawRecord

SOURCE_NAME = "kaggle_code_review_v2"



_CODE_CANDIDATES = [
    "code", "code_snippet", "source_code", "existing_code_snippet",
    "original_code", "code_block", "snippet", "old_code", "before",
    "patch"  
]

_COMMENT_CANDIDATES = [
    "comment", "comments", "review", "review_comment", "review_comments",
    "content", "instruction", "message", "suggestion", "feedback",
    "response", "responce" 
]


def _pick_column(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    return None



def _download_kaggle_dataset(dataset: str, download_dir: Path) -> Path:
  
    try:
        from kaggle import api  
    except Exception as e:
        raise RuntimeError(
        "Missing 'kaggle' package or import error. "
        "Please install it with: pip install kaggle"
        ) from e

    download_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(download_dir), quiet=False, unzip=False)

    for zf in download_dir.glob("*.zip"):
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(download_dir)
        zf.unlink(missing_ok=True)

    return download_dir



def fetch_kaggle_code_review_v2(
    dataset: str = "bulivington/code-review-data-v2",
    filename: str = "code_review_data.csv",
    local_dir: Optional[str] = None,
    encoding: str = "utf-8",
    code_col: Optional[str] = None,
    comment_col: Optional[str] = None,
) -> List[RawRecord]:
    if local_dir:
        data_dir = Path(local_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"local_dir does not exist: {data_dir}")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="kaggle_code_review_v2_"))
        data_dir = _download_kaggle_dataset(dataset, tmp)

    csv_path = data_dir / filename
    if not csv_path.exists():
        candidates = list(data_dir.glob("*.csv"))
        if len(candidates) == 1:
            csv_path = candidates[0]
        elif len(candidates) > 1:
            csv_path = max(candidates, key=lambda p: p.stat().st_size)
        else:
            raise FileNotFoundError(
                f"File '{filename}' not found in {data_dir}. "
                f"Available: {[p.name for p in data_dir.glob('*')]}"
            )

    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if df.empty:
        return []

    ccol = code_col or _pick_column(df.columns, _CODE_CANDIDATES)
    tcol = comment_col or _pick_column(df.columns, _COMMENT_CANDIDATES)

    if not ccol or not tcol:
        raise ValueError(
            "Failed to recognize code/comment columns.\n"
            f"Columns in CSV: {list(df.columns)}\n"
            f"Try specifying explicitly: code_col='...', comment_col='...'"
        )

    series_code = df[ccol].fillna("").astype(str).str.strip()
    series_comment = df[tcol].fillna("").astype(str).str.strip()

    out: List[RawRecord] = []
    dropped = 0

    for code, comment in zip(series_code, series_comment):
        if not code or not comment:
            dropped += 1
            continue
        out.append(RawRecord(source=SOURCE_NAME, code=code, comment=comment))

    if dropped:
        print(f"[{SOURCE_NAME}] dropped {dropped} empty records")

    return out
