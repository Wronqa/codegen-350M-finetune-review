
import os
import json
from contextlib import contextmanager

from config import (
    MLFLOW_TRACKING_URI,
)

try:
    import mlflow
except Exception:
    mlflow = None 

try:
    import torch
except Exception:
    torch = None

def _has_active_run() -> bool:
    return (mlflow is not None) and (mlflow.active_run() is not None)

def _ensure_experiment_by_name(name: str) -> None:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        mlflow.create_experiment(name)
    mlflow.set_experiment(name)

@contextmanager
def start_run(run_name: str, experiment: str | None = None):
    if mlflow is None:
        yield None
        return

    uri = MLFLOW_TRACKING_URI
    if uri:
        mlflow.set_tracking_uri(uri)

    exp_name = experiment 
    _ensure_experiment_by_name(exp_name)

    with mlflow.start_run(run_name=run_name):
        yield mlflow.active_run()

def log_params(d: dict):
    if not _has_active_run():
        return
    for k, v in d.items():
        try:
            mlflow.log_param(k, json.dumps(v) if isinstance(v, (dict, list)) else v)
        except Exception:
            try:
                mlflow.log_param(k, str(v))
            except Exception:
                pass

def log_metrics(d: dict, *, step: int | None = None):
    if not _has_active_run():
        return
    for k, v in d.items():
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception:
            pass

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object 

class MlflowStepLogger(TrainerCallback):
    def __init__(self, every_n_steps: int = 25, log_gpu: bool = True):
        self.every_n_steps = every_n_steps
        self.log_gpu = log_gpu

    @staticmethod
    def _is_enabled() -> bool:
        return _has_active_run()

    @staticmethod
    def _log_dict(payload: dict, step: int):
        if not payload:
            return
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                try:
                    mlflow.log_metric(k, float(v), step=step)
                except Exception:
                    pass

    def _log_gpu_mem(self, step: int):
        if not self.log_gpu or torch is None:
            return
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            return
        try:
            for i in range(torch.cuda.device_count()):
                used = torch.cuda.memory_allocated(i) / 1024**2 
                reserved = torch.cuda.memory_reserved(i) / 1024**2  
                mlflow.log_metric(f"gpu{i}_mem_used_mb", used, step=step)
                mlflow.log_metric(f"gpu{i}_mem_reserved_mb", reserved, step=step)
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_enabled() or logs is None:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if self.every_n_steps and step % self.every_n_steps != 0:
            return
        self._log_dict(logs, step)
        self._log_gpu_mem(step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self._is_enabled() or metrics is None:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        self._log_dict(metrics, step)
        self._log_gpu_mem(step)
