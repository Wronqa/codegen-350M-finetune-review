from __future__ import annotations

import os
import torch
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from utils.mlflow import start_run, log_params, log_metrics, MlflowStepLogger

from config import (
    TRAIN_PATH, VAL_PATH, ADAPTERS_DIR,
    MODEL_ID, SEED, MAX_SEQ_LEN, PER_DEV_BATCH, GRAD_ACCUM, LR, EPOCHS, FP16, WARMUP_RATIO,
    LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT, LR_SCHEDULER, PACKING,
    PADDING_SIDE, TRUNCATION_SIDE, MLFLOW_TRAIN_EXPERIMENT_NAME
)

def _ensure_dirs() -> None:
    os.makedirs(ADAPTERS_DIR, exist_ok=True)

def _load_dataset() -> DatasetDict:
    data_files = {"train": str(TRAIN_PATH), "validation": str(VAL_PATH)}
    ds = load_dataset("json", data_files=data_files)

    for split in ("train", "validation"):
        cols = set(ds[split].column_names)
        if not {"prompt", "response"}.issubset(cols):
            missing = {"prompt", "response"} - cols
            raise ValueError(f"{split} missing columns: {missing}")
    return ds

def _format_examples(ds: DatasetDict, sep: str = "\n\n### Response:\n") -> DatasetDict:
    def to_text(example):
        prompt = example["prompt"]
        resp = example["response"]
        return {"text": f"{prompt}{sep}{resp}"}
    return ds.map(to_text, remove_columns=ds["train"].column_names)

def _load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        padding_side=PADDING_SIDE,
        truncation_side=TRUNCATION_SIDE,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _get_last_ckpt_dir(output_dir: str) -> str | None:
    try:
        ckpt = get_last_checkpoint(output_dir)
        return ckpt
    except Exception:
        return None

def _load_base_model() -> AutoModelForCausalLM:
    dtype = torch.float16 if FP16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  
    return model

FIXED_LORA_TARGETS: List[str] = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"]

def _build_lora_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=FIXED_LORA_TARGETS,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model

def _build_sft_config() -> SFTConfig:
    return SFTConfig(
        output_dir=str(ADAPTERS_DIR),
        per_device_train_batch_size=PER_DEV_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        max_seq_length=MAX_SEQ_LEN,
        bf16=False,
        fp16=FP16,
        seed=SEED,
        report_to=[],     
        packing=PACKING,
        load_best_model_at_end=False,     
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
    )

def _build_trainer(model, tok, ds_fmt: DatasetDict) -> SFTTrainer:
    train_cfg = _build_sft_config()
    callbacks = [MlflowStepLogger()]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_fmt["train"],
        eval_dataset=ds_fmt["validation"],
        args=train_cfg,
        dataset_text_field="text",
        callbacks=callbacks,
    )
    return trainer

def _mlflow_hparams() -> Dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "seed": SEED,
        "max_seq_len": MAX_SEQ_LEN,
        "per_device_batch": PER_DEV_BATCH,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "epochs": EPOCHS,
        "fp16": FP16,
        "warmup_ratio": WARMUP_RATIO,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": ",".join(FIXED_LORA_TARGETS),
        "packing": PACKING,
        "padding_side": PADDING_SIDE,
        "truncation_side": TRUNCATION_SIDE,
        "scheduler": LR_SCHEDULER,
    }

def main() -> None:
    _ensure_dirs()

    ds = _load_dataset()
    ds_fmt = _format_examples(ds)

    tok = _load_tokenizer()
    base = _load_base_model()
    lora_model = _build_lora_model(base)

    trainer = _build_trainer(lora_model, tok, ds_fmt)

    last_ckpt = _get_last_ckpt_dir(str(ADAPTERS_DIR))
    if last_ckpt:
        print(f"[resume] Found checkpoint: {last_ckpt} -> resuming training from it.")
    else:
        print("[resume] No checkpoint found. Starting fresh training.")

    run_name = "codegen350m_lora_train"
    with start_run(run_name=run_name, experiment=MLFLOW_TRAIN_EXPERIMENT_NAME):
        log_params(_mlflow_hparams())

        trainer.train()

        eval_res = trainer.evaluate()
        final_metrics = {f"final_{k}": v for k, v in eval_res.items()}
        log_metrics(final_metrics)

    trainer.model.save_pretrained(str(ADAPTERS_DIR))
    tok.save_pretrained(str(ADAPTERS_DIR))
    print(f"LoRA adapter saved to: {ADAPTERS_DIR}")

if __name__ == "__main__":
    main()
