import json
from typing import List, Dict, Callable

from .schema import RawRecord
from .utils import set_seed, setup_logger, load_yaml, ensure_dir, write_jsonl, sample_cap
from src.config import SEED, MAX_EXAMPLES_PER_SOURCE, RAW_DATA_DIR, DEFAULT_DATA_CONFIG


from .sources.nutanix import fetch_nutanix_code_suggestions
from .sources.kaggle_code_review_v2 import fetch_kaggle_code_review_v2 


FETCHERS: Dict[str, Callable[..., List[RawRecord]]] = {
    "nutanix": fetch_nutanix_code_suggestions,
     "kaggle_code_review_v2": fetch_kaggle_code_review_v2,
}


def _validate_config(cfg: dict) -> None:
    if "sources" not in cfg or not isinstance(cfg["sources"], list) or not cfg["sources"]:
        raise ValueError("Config error: 'sources' must be a non-empty list.")
    for i, s in enumerate(cfg["sources"]):
        if "name" not in s or "type" not in s:
            raise ValueError(f"Config error: sources[{i}] must contain 'name' and 'type'.")


def main() -> None:
    log = setup_logger("dataprep.fetch")


    log.info("Loading config: %s", DEFAULT_DATA_CONFIG)
    cfg = load_yaml(DEFAULT_DATA_CONFIG)
    _validate_config(cfg)

    set_seed(SEED)
    log.info("Seed: %d", SEED)

    ensure_dir(RAW_DATA_DIR)

    cap = int(cfg.get("max_examples_per_source", MAX_EXAMPLES_PER_SOURCE))
    log.info("Max examples per source: %d", cap)

    counts: Dict[str, int] = {}

    for src in cfg["sources"]:
        name = src["name"]
        typ = src["type"]
        args = src.get("args", {})

        if typ not in FETCHERS:
            raise ValueError(f"Unknown source type: {typ}. Available: {', '.join(sorted(FETCHERS))}")

        log.info("[fetch] %s (%s) args=%s", name, typ, json.dumps(args, ensure_ascii=False))

        fetcher = FETCHERS[typ]
        records: List[RawRecord] = fetcher(**args)
        original_n = len(records)

        if original_n > cap:
            records = sample_cap(records, cap, seed=SEED)
            log.info("[cap] %s: %d -> %d", name, original_n, len(records))
        else:
            log.info("[cap] %s: no capping needed (%d <= %d)", name, original_n, cap)

        out_dir = RAW_DATA_DIR / name
        ensure_dir(out_dir)
        out_file = out_dir / "data.jsonl"

        n_written = write_jsonl(out_file, (r.__dict__ for r in records))
        counts[name] = n_written

        log.info("[save] %s: wrote %d records -> %s", name, n_written, out_file)

    total = sum(counts.values())
    log.info("[summary] sources: %d | total examples: %d", len(counts), total)
    for k, v in counts.items():
        log.info("  - %-28s %6d", k, v)

    print("[fetch] summary:", counts)


if __name__ == "__main__":
    main()
