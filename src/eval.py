import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import csv
import json
import random
import logging
from typing import Dict, List, Optional, Tuple
from threading import Thread
from queue import Queue, Empty

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import logging as hf_logging
from difflib import SequenceMatcher
from peft import PeftModel

from config import (
    MODEL_ID,
    VAL_PATH,
    ADAPTERS_DIR,
    EVAL_REPORT_DIR,
    EVAL_MAX_NEW_TOKENS,
    BERT_LANG,
    MLFLOW_EVAL_EXPERIMENT_NAME,
    SEED,
    DEFAULT_MAX_RESPONSE_CHARS,
    MLFLOW_TRACKING_URI,
    EVAL_LOG_EVERY,
    EVAL_GEN_TIMEOUT_S,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  

try:
    import evaluate
except Exception:
    evaluate = None

try:
    from bert_score import score as bertscore_score
except Exception:
    bertscore_score = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")
hf_logging.set_verbosity_error()

SINGLE_SENTENCE: bool = True
DETERMINISTIC: bool = True

LOG_EVERY: int = int(EVAL_LOG_EVERY)
GEN_TIMEOUT_S: float = float(EVAL_GEN_TIMEOUT_S)


def one_sentence(text: str, max_chars: int = 300) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    s = (parts[0] if parts and parts[0] else t).strip()
    return s[:max_chars].rstrip()

def normalize(text: str, max_chars: Optional[int] = None) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t[:max_chars].rstrip() if max_chars else t

def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def device_str() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
    return "cpu"


def now_stamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H-%M-%S")


def get_eval_limit(ds_len: int) -> Optional[int]:
    val = os.getenv("EVAL_LIMIT")
    if not val:
        return None
    try:
        n = int(val)
        return max(0, min(n, ds_len))
    except Exception:
        return None


def setup_torch(deterministic: bool) -> None:
    torch.set_grad_enabled(False)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def load_model_and_tokenizer(model_id: str, adapters_dir, dtype) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    log.info("[init] Loading base model: %s", model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    log.info("[init] Loading LoRA adapter: %s", adapters_dir)
    model = PeftModel.from_pretrained(base, adapters_dir.as_posix())
    model.eval()
    model.config.use_cache = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model, tok, dev


def resolve_context_window(model) -> int:
    raw_ctx = getattr(model.config, "max_position_embeddings", None)
    try:
        ctx = int(raw_ctx) if raw_ctx else 2048
    except Exception:
        ctx = 2048
    return ctx

def load_validation_dataset(val_path) -> Dataset:
    log.info("[data] Loading validation jsonl: %s", val_path)
    ds = load_dataset("json", data_files={"val": str(val_path)})["val"]
    
    required = {"prompt", "response"}
    if not required.issubset(set(ds.column_names)):
        missing = required - set(ds.column_names)
        raise ValueError(f"VAL_PATH missing required columns: {missing}")
    return ds

def _gen_worker(q: Queue, model, inputs, gen_kwargs):
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        q.put(out)
    except Exception as e:
        q.put(e)

def generate_one(
    model,
    tokenizer,
    prompt: str,
    dev,
    max_input_len: int,
    max_new_tokens: int,
    single_sentence: bool,
    max_chars: int,
    timeout_s: float,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(dev)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    q: Queue = Queue(maxsize=1)
    t = Thread(target=_gen_worker, args=(q, model, inputs, gen_kwargs), daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        log.warning("[eval] Generation timeout after %.1fs — returning empty output", timeout_s)
        return ""

    try:
        res = q.get_nowait()
    except Empty:
        log.warning("[eval] Generation queue empty after thread join — returning empty output")
        return ""

    if isinstance(res, Exception):
        log.warning("[eval] Generation failed: %s", res)
        return ""

    out = res
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    if gen_ids.numel() == 0:
        return ""

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = one_sentence(gen_text, max_chars=max_chars) if single_sentence else normalize(gen_text, max_chars=max_chars)

    text = text.strip()
    if not text or text in {".", "..."}:
        return ""
    return text


def eval_loop(
    ds: Dataset,
    model,
    tokenizer,
    dev,
    ctx_window: int,
    max_new_tokens: int,
    single_sentence: bool,
    max_chars: int,
    log_every: int,
    timeout_s: float,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], float]:
    max_input_len = max(8, int(ctx_window) - int(max_new_tokens) - 8)

    n_total = len(ds) if limit is None else min(limit, len(ds))
    log.info("[eval] Evaluating %d sample(s) (max_new_tokens=%s)", n_total, max_new_tokens)

    rows: List[Dict[str, str]] = []
    empty_gens = 0
    sims_running_sum = 0.0
    len_running_sum = 0

    iterator = tqdm(ds.select(range(n_total)), total=n_total, desc="eval")
    for i, ex in enumerate(iterator):
        prompt_raw = ex["prompt"]
        expected_raw = ex["response"]

        expected = one_sentence(expected_raw, max_chars=max_chars) if single_sentence \
                   else normalize(expected_raw, max_chars=max_chars)

        generated = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_raw,
            dev=dev,
            max_input_len=max_input_len,
            max_new_tokens=int(max_new_tokens),
            single_sentence=bool(single_sentence),
            max_chars=int(max_chars),
            timeout_s=float(timeout_s),
        )

        if len(generated.strip()) == 0:
            empty_gens += 1

        sim = seq_sim(expected, generated)
        sims_running_sum += sim
        len_running_sum += len(generated)

        rows.append({"prompt": prompt_raw, "expected": expected, "generated": generated})

        if (i + 1) % log_every == 0:
            avg_sim_so_far = sims_running_sum / (i + 1)
            avg_len_so_far = len_running_sum / (i + 1)
            iterator.set_postfix({
                "avg_sim": f"{avg_sim_so_far:.3f}",
                "avg_len": f"{avg_len_so_far:.1f}",
                "empty%": f"{(empty_gens / (i + 1) * 100):.1f}",
            })
            log.info(
                "[eval][%d/%d] avg_sim=%.3f avg_len=%.1f empty=%.1f%%",
                i + 1, n_total, avg_sim_so_far, avg_len_so_far, (empty_gens / (i + 1) * 100.0),
            )

    return rows, float(max_input_len)


def compute_metrics(rows: List[Dict[str, str]], dtype, device_desc: str, max_input_len: float, ctx_window: int) -> Dict[str, float | int | str]:
    sims = [seq_sim(r["expected"], r["generated"]) for r in rows] or [0.0]
    avg_sim = float(sum(sims) / len(sims))
    lengths = [len(r["generated"]) for r in rows] if rows else []
    avg_len_chars = float(sum(lengths) / len(lengths)) if lengths else 0.0
    pct_punct = float(sum(1 for r in rows if r["generated"].endswith((".", "!", "?"))) / len(rows) * 100) if rows else 0.0
    pct_empty = float(sum(1 for r in rows if not r["generated"].strip()) / len(rows) * 100) if rows else 0.0

    if lengths:
        p50 = float(np.median(lengths))
        p05 = float(np.percentile(lengths, 5))
        p95 = float(np.percentile(lengths, 95))
    else:
        p50 = p05 = p95 = 0.0

    metrics: Dict[str, float | int | str] = {
        "device": device_desc,
        "dtype": str(dtype),
        "samples": len(rows),
        "avg_similarity": round(avg_sim, 3),
        "avg_generated_len_chars": round(avg_len_chars, 1),
        "len_chars_p50": round(p50, 1),
        "len_chars_p05": round(p05, 1),
        "len_chars_p95": round(p95, 1),
        "pct_ends_with_punctuation": round(pct_punct, 1),
        "pct_empty_generations": round(pct_empty, 1),
        "single_sentence_mode": bool(SINGLE_SENTENCE),
        "max_input_len": int(max_input_len),
        "context_window": int(ctx_window),
        "seed": SEED,
        "deterministic": bool(DETERMINISTIC),
    }
    return metrics


def add_optional_text_metrics(metrics: Dict[str, float | int | str], rows: List[Dict[str, str]]) -> None:
    try:
        if evaluate is not None and rows:
            rouge = evaluate.load("rouge")
            preds = [r["generated"] for r in rows]
            refs  = [r["expected"]  for r in rows]
            rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
            metrics["rouge1"]    = round(float(rouge_res.get("rouge1", 0.0)), 4)
            metrics["rouge2"]    = round(float(rouge_res.get("rouge2", 0.0)), 4)
            metrics["rougeL"]    = round(float(rouge_res.get("rougeL", 0.0)), 4)
            metrics["rougeLsum"] = round(float(rouge_res.get("rougeLsum", 0.0)), 4)
    except Exception as e:
        log.warning("[eval][warn] ROUGE failed: %s", e)

    try:
        if bertscore_score is not None and rows:
            preds = [r["generated"] for r in rows]
            refs  = [r["expected"]  for r in rows]
            P, R, F1 = bertscore_score(preds, refs, lang=BERT_LANG, verbose=False)
            metrics["bertscore_P"]  = round(float(P.mean()), 4)
            metrics["bertscore_R"]  = round(float(R.mean()), 4)
            metrics["bertscore_F1"] = round(float(F1.mean()), 4)
    except Exception as e:
        log.warning("[eval][warn] BERTScore failed: %s", e)



def save_reports(rows: List[Dict[str, str]], metrics: Dict[str, float | int | str], run_name: str) -> Tuple[str, str]:
    EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = EVAL_REPORT_DIR / f"{run_name}_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "expected", "generated"])
        w.writeheader()
        w.writerows(rows)

    json_path = EVAL_REPORT_DIR / f"{run_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            **metrics,
            "samples_csv": str(csv_path),
        }, f, indent=2)

    return str(csv_path), str(json_path)


def mlflow_start(run_name: str):
    try:
        import mlflow  
    except Exception:
        return None, None

    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EVAL_EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=run_name)
        return mlflow, run
    except Exception as e:
        log.warning("MLflow disabled (reason: %s)", e)
        return None, None


def mlflow_log_and_end(mlflow, run, adapter_dir, metrics: Dict[str, float | int | str], csv_path: str, json_path: str) -> None:
    if not (mlflow and run):
        return
    try:
        mlflow.log_param("model_id", MODEL_ID)
        mlflow.log_param("adapter_dir", str(adapter_dir))
        mlflow.log_param("max_new_tokens", EVAL_MAX_NEW_TOKENS)
        mlflow.log_param("bert_lang", BERT_LANG)
        mlflow.log_param("single_sentence_mode", str(bool(SINGLE_SENTENCE)))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("deterministic", str(bool(DETERMINISTIC)))

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(json_path))
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


def main():
    random.seed(SEED); np.random.seed(SEED); set_seed(SEED)
    setup_torch(DETERMINISTIC)

    ts = now_stamp()
    run_name = f"{MLFLOW_EVAL_EXPERIMENT_NAME}_{ts}"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model, tokenizer, dev = load_model_and_tokenizer(MODEL_ID, ADAPTERS_DIR, dtype)
    if torch.cuda.is_available() and not DETERMINISTIC:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info("[init] Device: %s | dtype: %s", device_str(), dtype)

    ds = load_validation_dataset(VAL_PATH)

    limit = get_eval_limit(len(ds))

    ctx = resolve_context_window(model)
    rows, max_input_len = eval_loop(
        ds=ds,
        model=model,
        tokenizer=tokenizer,
        dev=dev,
        ctx_window=ctx,
        max_new_tokens=int(EVAL_MAX_NEW_TOKENS),
        single_sentence=bool(SINGLE_SENTENCE),
        max_chars=int(DEFAULT_MAX_RESPONSE_CHARS),
        log_every=int(LOG_EVERY),
        timeout_s=float(GEN_TIMEOUT_S),
        limit=limit,
    )

    metrics = compute_metrics(rows, dtype=dtype, device_desc=device_str(), max_input_len=max_input_len, ctx_window=ctx)
    add_optional_text_metrics(metrics, rows)

    csv_path, json_path = save_reports(rows, metrics, run_name)
    log.info("[eval] done. Metrics:\n%s", json.dumps(metrics, indent=2))

    mlflow, run = mlflow_start(run_name)
    mlflow_log_and_end(mlflow, run, ADAPTERS_DIR, metrics, csv_path, json_path)


if __name__ == "__main__":
    main()
