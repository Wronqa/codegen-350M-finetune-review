from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import (
    DEFAULT_DATA_CONFIG,
    RAW_DATA_DIR,
    PREPARED_DATA_DIR,
    SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_MAX_EXAMPLES_FINAL,
    DEFAULT_MAX_RESPONSE_CHARS,
)

from .utils import (
    set_seed,
    to_prepared,
    setup_logger,
    load_yaml,
    ensure_dir,
    load_jsonl,
    save_jsonl,
)

def _validate_prepare_params(train_ratio: float, max_final: int, max_resp_chars: int) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"prepare.train_ratio must be in (0,1), got {train_ratio}")
    if max_final <= 0:
        raise ValueError(f"prepare.max_examples_final must be > 0, got {max_final}")
    if max_resp_chars <= 0:
        raise ValueError(f"prepare.max_response_chars must be > 0, got {max_resp_chars}")

def _dedup_pairs(rows: List[Dict], *, log, by_keys: Tuple[str, str] = ("prompt", "response")) -> List[Dict]:
    """Usuń duplikaty par (prompt,response)."""
    seen = set()
    out = []
    for r in rows:
        key = tuple((r.get(k) or "").strip() for k in by_keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    removed = len(rows) - len(out)
    if removed:
        log.info("[prepare] dedup removed %d duplicates", removed)
    return out

def load_config() -> dict:
    cfg_path = os.getenv("DATA_CONFIG", str(DEFAULT_DATA_CONFIG))
    log = setup_logger("dataprep.prepare")
    log.info("Loading config: %s", cfg_path)
    cfg = load_yaml(cfg_path)

    if "sources" not in cfg or not isinstance(cfg["sources"], list):
        raise ValueError("Config error: 'sources' must be a list (can be empty).")
    return cfg

def load_all_raw_sources(cfg: dict, *, log) -> List[Dict]:
    merged: List[Dict] = []
    missing = 0
    for src in cfg.get("sources", []):
        name = src.get("name")
        if not name:
            log.warning("[prepare] source without 'name' in config, skipping: %s", src)
            continue
        p: Path = RAW_DATA_DIR / name / "data.jsonl"
        if not p.exists():
            log.warning("[prepare] missing raw file for %s: %s", name, p)
            missing += 1
            continue
        rows = load_jsonl(p)
        log.info("[prepare] loaded %d raw rows from %s", len(rows), p)
        merged.extend(rows)
    log.info("[prepare] total raw records loaded: %d (missing sources: %d)", len(merged), missing)
    return merged

def normalize_records(merged: List[Dict], *, max_resp_chars: int, log) -> List[Dict]:
    prepared: List[Dict] = []
    for r in merged:
        code = (r.get("code") or "").strip()
        comment = (r.get("comment") or "").strip()
        pr = to_prepared(code, comment, max_resp_chars=max_resp_chars)
        if pr:
            prepared.append({"prompt": pr.prompt, "response": pr.response})
    log.info("[prepare] prepared usable records: %d / %d", len(prepared), len(merged))
    return prepared

def cap_and_shuffle(prepared: List[Dict], *, max_final: int, rng: random.Random, log) -> List[Dict]:
    if len(prepared) > max_final:
        prepared = rng.sample(prepared, max_final)
        log.info("[prepare] capped: %d -> %d", len(prepared), max_final)
    rng.shuffle(prepared)
    return prepared

def split_dataset(prepared: List[Dict], *, train_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    cut = int(len(prepared) * train_ratio)
    train = prepared[:cut]
    val = prepared[cut:]
    return train, val

def save_splits(train: List[Dict], val: List[Dict], *, log) -> None:
    ensure_dir(PREPARED_DATA_DIR)
    save_jsonl(train, PREPARED_DATA_DIR / "train.jsonl")
    save_jsonl(val, PREPARED_DATA_DIR / "val.jsonl")

    manifest = {
        "train": len(train),
        "val": len(val),
        "dir": str(PREPARED_DATA_DIR),
    }
    import json
    with open(PREPARED_DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("[prepare] saved: %d train, %d val → %s", len(train), len(val), PREPARED_DATA_DIR)

def main() -> None:
    log = setup_logger("dataprep.prepare")

    cfg = load_config()

    seed = int(cfg.get("seed", SEED))
    set_seed(seed)
    rng = random.Random(seed)
    log.info("Seed: %d", seed)

    prep_cfg = (cfg.get("prepare") or {})
    train_ratio = float(prep_cfg.get("train_ratio", DEFAULT_TRAIN_RATIO))
    max_final = int(prep_cfg.get("max_examples_final", DEFAULT_MAX_EXAMPLES_FINAL))
    max_resp_chars = int(prep_cfg.get("max_response_chars", DEFAULT_MAX_RESPONSE_CHARS))
    _validate_prepare_params(train_ratio, max_final, max_resp_chars)
    log.info("train_ratio=%.3f | max_final=%d | max_response_chars=%d",
             train_ratio, max_final, max_resp_chars)

    merged = load_all_raw_sources(cfg, log=log)
    if not merged:
        log.warning("[prepare] no raw data found — nothing to do")
        return

    prepared = normalize_records(merged, max_resp_chars=max_resp_chars, log=log)
    if not prepared:
        log.warning("[prepare] no usable records after normalization — nothing to save")
        return

    prepared = _dedup_pairs(prepared, log=log)

    prepared = cap_and_shuffle(prepared, max_final=max_final, rng=rng, log=log)
    train, val = split_dataset(prepared, train_ratio=train_ratio)
    save_splits(train, val, log=log)

if __name__ == "__main__":
    main()
