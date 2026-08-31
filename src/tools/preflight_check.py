"""Preflight checker for the local Signal Forge veRL GRPO launch path."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - exercised only on minimal envs.
    yaml = None

OLD_AUTODL_DIR = "signal_forge_a0_autodl_20260730_145001"
IMPORTANT_KEYS = [
    "algorithm.adv_estimator",
    "algorithm.use_kl_in_reward",
    "algorithm.norm_adv_by_std_in_grpo",
    "data.train_files",
    "data.val_files",
    "data.train_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "data.filter_overlong_prompts",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "actor_rollout_ref.actor.ppo_epochs",
    "actor_rollout_ref.actor.use_dynamic_bsz",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.clip_ratio",
    "actor_rollout_ref.actor.clip_ratio_low",
    "actor_rollout_ref.actor.clip_ratio_high",
    "actor_rollout_ref.actor.entropy_coeff",
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.rollout.name",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.top_p",
    "actor_rollout_ref.rollout.top_k",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "actor_rollout_ref.rollout.gpu_memory_utilization",
    "actor_rollout_ref.rollout.max_model_len",
    "actor_rollout_ref.rollout.max_num_batched_tokens",
    "actor_rollout_ref.rollout.max_num_seqs",
    "reward.reward_manager.source",
    "reward.reward_manager.name",
    "reward.custom_reward_function.path",
    "reward.custom_reward_function.name",
    "reward.custom_reward_function.reward_kwargs",
    "trainer.logger",
    "trainer.project_name",
    "trainer.experiment_name",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
    "trainer.default_local_dir",
    "trainer.rollout_data_dir",
    "trainer.validation_data_dir",
    "trainer.total_epochs",
    "trainer.total_training_steps",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.use_v1",
]


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class Reporter:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.checks: list[Check] = []

    def add(self, status: str, name: str, message: str, **details: Any) -> None:
        if status == "WARN" and self.strict:
            status = "FAIL"
        self.checks.append(Check(name=name, status=status, message=message, details=details))

    def pass_(self, name: str, message: str, **details: Any) -> None:
        self.add("PASS", name, message, **details)

    def warn(self, name: str, message: str, **details: Any) -> None:
        self.add("WARN", name, message, **details)

    def fail(self, name: str, message: str, **details: Any) -> None:
        self.add("FAIL", name, message, **details)

    @property
    def fail_count(self) -> int:
        return sum(c.status == "FAIL" for c in self.checks)

    @property
    def warn_count(self) -> int:
        return sum(c.status == "WARN" for c in self.checks)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def set_deep(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def flatten_unresolved(value: Any, prefix: str = "") -> list[str]:
    hits = []
    if isinstance(value, dict):
        for key, sub in value.items():
            hits.extend(flatten_unresolved(sub, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, sub in enumerate(value):
            hits.extend(flatten_unresolved(sub, f"{prefix}[{index}]"))
    elif isinstance(value, str) and ("${" in value or value == "???"):
        hits.append(prefix)
    return hits


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def parse_scalar(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read resolved Hydra YAML")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Resolved config is not a mapping: {path}")
    return loaded


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    for item in overrides:
        if "=" not in item or item.startswith("--"):
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        set_deep(out, key, parse_scalar(value))
    return out


def run_text(args: list[str], cwd: Path | None = None, timeout: float = 5.0) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except Exception:
        return ""


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def source_fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        if path.exists():
            h.update(str(path).encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def path_value(value: Any, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value)
    if "://" in text or text.startswith("Qwen/"):
        return Path(text)
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = root / p
    return p


def is_nonempty_dir(path: Path | None) -> bool:
    return bool(path and path.exists() and path.is_dir() and any(path.iterdir()))


def detect_batch_semantics(root: Path) -> dict[str, Any]:
    trainer = root / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = trainer.read_text(encoding="utf-8") if trainer.exists() else ""
    actor_mult = "ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n" in text
    critic_mult = "ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n" in text
    return {
        "supported": bool(actor_mult),
        "actor_prompt_level_then_times_rollout_n": bool(actor_mult),
        "critic_prompt_level_then_times_rollout_n": bool(critic_mult),
        "source": str(trainer),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def dataset_rows(path: Path) -> tuple[int, list[str], Any | None]:
    import pandas as pd

    df = pd.read_parquet(path)
    return len(df), list(df.columns), df


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt)
    return str(prompt)


def source_counts(df: Any) -> dict[str, int]:
    if df is None or "data_source" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["data_source"].value_counts(dropna=False).to_dict().items()}


def maybe_filter_lengths(cfg: dict[str, Any], df: Any, tokenizer: Any) -> tuple[int, list[int], list[str]]:
    prompt_key = get(cfg, "data.prompt_key", "prompt")
    max_prompt = int(get(cfg, "data.max_prompt_length", 0) or 0)
    ids = []
    dropped = []
    for idx, row in df.iterrows():
        prompt = row[prompt_key]
        token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        ids.append(len(token_ids))
        if max_prompt and len(token_ids) > max_prompt:
            extra = row.get("extra_info", {}) if hasattr(row, "get") else {}
            prompt_id = extra.get("prompt_id", str(idx)) if isinstance(extra, dict) else str(idx)
            dropped.append(str(prompt_id))
    return len(df) - len(dropped), ids, dropped


def simulate_grpo(n: int, rewards: list[float]) -> dict[str, Any]:
    import torch

    if len(rewards) != n:
        raise ValueError("reward count must equal rollout.n")
    token_rewards = torch.zeros((n, 4), dtype=torch.float32)
    token_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    response_mask = torch.ones((n, 4), dtype=torch.float32)
    scores = token_rewards.sum(dim=-1)
    if n == 1:
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
    else:
        mean = torch.mean(scores)
        std = torch.std(scores)
    scalars = (scores - mean) / (std + 1e-6)
    advantages = scalars.unsqueeze(-1) * response_mask
    returns = advantages
    return {
        "advantages_finite": bool(torch.isfinite(advantages).all().item()),
        "returns_finite": bool(torch.isfinite(returns).all().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
    }


def check_provenance(rep: Reporter, cfg: dict[str, Any], root: Path, launch_script: Path, mode: str) -> dict[str, Any]:
    info = {
        "project_root": str(root),
        "launch_script": str(launch_script),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "git_branch": run_text(["git", "branch", "--show-current"], root),
        "git_head": run_text(["git", "rev-parse", "HEAD"], root),
        "git_dirty": bool(run_text(["git", "status", "--porcelain"], root)),
        "versions": {p: package_version(p) for p in ["torch", "vllm", "transformers", "verl", "rewardscope", "math-verify"]},
        "verl_fingerprint": source_fingerprint([
            root / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py",
            root / "verl" / "verl" / "experimental" / "reward_loop" / "reward_manager" / "naive.py",
            root / "verl" / "verl" / "utils" / "dataset" / "rl_dataset.py",
            root / "verl" / "verl" / "utils" / "config.py",
        ]),
        "important_parameters": {key: get(cfg, key) for key in IMPORTANT_KEYS},
    }
    unresolved = flatten_unresolved(cfg)
    if unresolved:
        rep.fail("A.config.unresolved", "resolved config still contains unresolved OmegaConf values", values=unresolved[:50])
    else:
        rep.pass_("A.config.unresolved", "no unresolved OmegaConf values detected")
    if info["git_dirty"]:
        rep.warn("A.git.dirty", "git worktree is dirty; report records this for reproducibility")
    else:
        rep.pass_("A.git.clean", "git worktree is clean")
    model_utils = root / "verl" / "verl" / "utils" / "model.py"
    models_pkg = root / "verl" / "verl" / "models"
    model_utils_text = model_utils.read_text(encoding="utf-8", errors="ignore") if model_utils.exists() else ""
    if "verl.models.registry" in model_utils_text and not models_pkg.exists():
        rep.fail(
            "A.verl.source_integrity",
            "local veRL source imports verl.models.registry but verl/verl/models is missing",
            importer=str(model_utils),
            missing_package=str(models_pkg),
        )
    else:
        rep.pass_("A.verl.source_integrity", "local veRL model package references are present or not required")
    py_path = os.environ.get("PYTHONPATH", "")
    old_index = py_path.find(OLD_AUTODL_DIR)
    current_index = py_path.find(str(root))
    if old_index >= 0 and (current_index < 0 or old_index < current_index):
        rep.fail("A.env.pythonpath_stale", "PYTHONPATH resolves old AutoDL bundle before current project", pythonpath=py_path)
    elif old_index >= 0:
        rep.warn("A.env.pythonpath_stale", "PYTHONPATH still contains old AutoDL bundle after current project", pythonpath=py_path)
    else:
        rep.pass_("A.env.pythonpath", "PYTHONPATH has no old AutoDL bundle reference")
    rep.pass_("A.config.provenance", f"captured provenance in {mode} mode", **info)
    return info


def check_paths(rep: Reporter, cfg: dict[str, Any], root: Path, allow_existing_output: bool) -> dict[str, Any]:
    paths = {
        "train": path_value(get(cfg, "data.train_files"), root),
        "val": path_value(get(cfg, "data.val_files"), root),
        "model": path_value(get(cfg, "actor_rollout_ref.model.path"), root),
        "checkpoint": path_value(get(cfg, "trainer.default_local_dir"), root),
        "rollout_output": path_value(get(cfg, "trainer.rollout_data_dir"), root),
        "validation_output": path_value(get(cfg, "trainer.validation_data_dir"), root),
        "wandb_dir": path_value(os.environ.get("WANDB_DIR"), root),
    }
    for label, value in paths.items():
        if value is None:
            rep.warn(f"B.path.{label}", f"{label} path is unset")
            continue
        if value.is_symlink() and not value.exists():
            rep.fail(f"B.path.{label}.symlink", f"{label} is a broken symlink", path=str(value))
        elif label in {"train", "val"} and not value.exists():
            rep.fail(f"B.path.{label}.exists", f"{label} file does not exist", path=str(value))
        elif label == "model" and str(value).startswith("Qwen/"):
            if os.environ.get("ALLOW_HF_MODEL_DOWNLOAD", "1").lower() in {"1", "true", "yes", "on"}:
                rep.pass_(
                    "B.path.model.hf_repo",
                    "model path is a Hugging Face repo id; training will download/cache it if not already cached",
                    repo_id=str(value),
                    hf_home=os.environ.get("HF_HOME"),
                    hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
                )
            else:
                rep.fail("B.path.model.local", "model path must be local when ALLOW_HF_MODEL_DOWNLOAD=0", path=str(value))
        elif label == "model" and not value.exists():
            rep.fail("B.path.model.local", "local model path does not exist", path=str(value))
        else:
            rep.pass_(f"B.path.{label}", f"{label} path is usable", path=str(value))
    model = paths["model"]
    if model and model.exists():
        required = ["config.json", "tokenizer_config.json"]
        missing = [name for name in required if not (model / name).exists()]
        if missing:
            rep.fail("B.model.required_files", "model config/tokenizer metadata missing", missing=missing, model=str(model))
        else:
            rep.pass_("B.model.required_files", "model config/tokenizer metadata exists", model=str(model))
    for label in ["checkpoint", "rollout_output", "validation_output"]:
        p = paths[label]
        if is_nonempty_dir(p) and not allow_existing_output:
            rep.fail(f"B.path.{label}.collision", "existing nonempty experiment output would be reused", path=str(p))
        elif p:
            parent = p if p.exists() and p.is_dir() else p.parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
                probe = parent / ".preflight_write_probe"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                rep.pass_(f"B.path.{label}.writable", "output parent is writable", path=str(parent))
            except Exception as exc:
                rep.fail(f"B.path.{label}.writable", "output parent is not writable", path=str(parent), error=str(exc))
    resume_mode = get(cfg, "trainer.resume_mode")
    resume_path = get(cfg, "trainer.resume_from_path")
    val_only = bool(get(cfg, "trainer.val_only", False))
    resume_dir = path_value(resume_path, root) if resume_path else None
    if resume_mode in {"disable", None}:
        rep.pass_("B.resume.mode", "resume_mode disables accidental checkpoint resume", resume_mode=resume_mode)
    elif resume_mode == "resume_path" and val_only and resume_dir and resume_dir.exists():
        rep.pass_("B.resume.mode", "explicit val-only checkpoint reload is configured", resume_mode=resume_mode, resume_from_path=str(resume_dir))
    elif resume_mode == "resume_path" and allow_existing_output and resume_dir and resume_dir.exists():
        rep.pass_("B.resume.mode", "explicit checkpoint resume for continued training is configured", resume_mode=resume_mode, resume_from_path=str(resume_dir), val_only=val_only)
    else:
        rep.fail("B.resume.mode", "launch should not accidentally resume", resume_mode=resume_mode, resume_from_path=resume_path, val_only=val_only)
    old_refs = []
    for candidate in [root / "config", root / "src" / "scripts_a0", root / "src" / "signal_forge"]:
        if not candidate.exists():
            continue
        for file in candidate.rglob("*"):
            if file.is_file() and OLD_AUTODL_DIR in file.read_text(encoding="utf-8", errors="ignore"):
                old_refs.append(str(file))
    if old_refs:
        rep.warn("B.paths.old_autodl_refs", "old AutoDL directory reference found in source/config", files=old_refs)
    else:
        rep.pass_("B.paths.old_autodl_refs", "no hard-coded old AutoDL directory reference found in checked files")
    return {k: str(v) if v is not None else None for k, v in paths.items()}


def check_dataset(rep: Reporter, cfg: dict[str, Any], paths: dict[str, str | None], deep: bool, tokenizer: Any | None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    prompt_key = get(cfg, "data.prompt_key", "prompt")
    batch_size = int(get(cfg, "data.gen_batch_size", get(cfg, "data.train_batch_size", 0)) or 0)
    for split, key in [("train", "train"), ("val", "val")]:
        p = Path(paths[key]) if paths.get(key) else None
        if not p or not p.exists():
            continue
        try:
            raw_size, columns, df = dataset_rows(p)
        except Exception as exc:
            rep.fail(f"C.dataset.{split}.read", "failed to read dataset", path=str(p), error=str(exc))
            continue
        required = {"data_source", prompt_key, "reward_model", "extra_info"}
        missing = sorted(required - set(columns))
        if missing:
            rep.fail(f"C.dataset.{split}.columns", "dataset missing required RLHFDataset columns", missing=missing)
        else:
            rep.pass_(f"C.dataset.{split}.columns", "dataset has required RLHFDataset columns", columns=columns)
        nulls = {}
        for col in ["data_source", prompt_key, "reward_model"]:
            if col in df.columns:
                nulls[col] = int(df[col].isna().sum())
        empty_prompt = 0
        if prompt_key in df.columns:
            empty_prompt = sum(not prompt_text(v).strip() for v in df[prompt_key])
        empty_gt = 0
        if "reward_model" in df.columns:
            for rm in df["reward_model"]:
                if not isinstance(rm, dict) or not str(rm.get("ground_truth", "")).strip():
                    empty_gt += 1
        if any(nulls.values()) or empty_prompt or empty_gt:
            rep.fail(f"C.dataset.{split}.nonempty", "dataset has null/empty prompt or ground_truth", nulls=nulls, empty_prompt=empty_prompt, empty_ground_truth=empty_gt)
        else:
            rep.pass_(f"C.dataset.{split}.nonempty", "prompt and ground truth fields are nonempty")
        effective = raw_size
        dropped_overlong = []
        prompt_lengths = []
        if deep and tokenizer is not None and prompt_key in df.columns:
            effective, prompt_lengths, dropped_overlong = maybe_filter_lengths(cfg, df, tokenizer)
        max_samples = int(get(cfg, f"data.{split}_max_samples", -1) or -1)
        if split == "train":
            max_samples = int(get(cfg, "data.train_max_samples", -1) or -1)
        elif split == "val":
            max_samples = int(get(cfg, "data.val_max_samples", -1) or -1)
        sampled = min(effective, max_samples) if max_samples > 0 else effective
        report[split] = {
            "path": str(p),
            "raw_rows": raw_size,
            "effective_rows": sampled,
            "columns": columns,
            "source_counts": source_counts(df),
            "dropped_overlong_count": len(dropped_overlong),
        }
        if prompt_lengths:
            report[split]["prompt_tokens"] = {
                "min": min(prompt_lengths),
                "mean": statistics.mean(prompt_lengths),
                "p90": percentile([float(x) for x in prompt_lengths], 0.90),
                "p95": percentile([float(x) for x in prompt_lengths], 0.95),
                "max": max(prompt_lengths),
            }
        if split == "train":
            steps_per_epoch = sampled // batch_size if batch_size > 0 else 0
            dropped_drop_last = sampled % batch_size if batch_size > 0 else sampled
            report["steps_per_epoch"] = steps_per_epoch
            report["drop_last_discarded_rows"] = dropped_drop_last
            if sampled <= 0:
                rep.fail("C.dataset.train.effective_rows", "effective train dataset has zero rows")
            elif sampled < batch_size:
                rep.fail("C.dataset.train.batch", "effective train dataset is smaller than one training batch", effective_rows=sampled, train_batch_size=batch_size)
            else:
                rep.pass_("C.dataset.train.batch", "effective train dataset can form at least one drop_last batch", effective_rows=sampled, train_batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            if dropped_drop_last:
                rep.warn("C.dataset.train.drop_last", "drop_last will discard train samples", discarded_rows=dropped_drop_last)
        rep.pass_(f"C.dataset.{split}.size", "dataset size inspected", **report[split])
    if "train" in report and "val" in report:
        try:
            train_df = dataset_rows(Path(report["train"]["path"]))[2]
            val_df = dataset_rows(Path(report["val"]["path"]))[2]
            train_ids = {str(x.get("prompt_id")) for x in train_df["extra_info"] if isinstance(x, dict)}
            val_ids = {str(x.get("prompt_id")) for x in val_df["extra_info"] if isinstance(x, dict)}
            overlap = sorted(train_ids & val_ids)
            if overlap:
                rep.warn("C.dataset.overlap", "train/val prompt_id overlap detected", overlap=overlap[:20], count=len(overlap))
            else:
                rep.pass_("C.dataset.overlap", "no train/val prompt_id overlap detected")
        except Exception as exc:
            rep.warn("C.dataset.overlap", "could not compute prompt_id overlap", error=str(exc))
    return report


def check_steps(rep: Reporter, cfg: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    steps_per_epoch = int(dataset.get("steps_per_epoch", 0) or 0)
    total_epochs = int(get(cfg, "trainer.total_epochs", 0) or 0)
    total_steps = int(get(cfg, "trainer.total_training_steps", 0) or 0)
    available = total_epochs * steps_per_epoch
    out = {"steps_per_epoch": steps_per_epoch, "total_epochs": total_epochs, "total_training_steps": total_steps, "available_epoch_loop_steps": available}
    val_only = bool(get(cfg, "trainer.val_only", False))
    if val_only:
        rep.pass_("D.steps.epoch_budget", "val-only run does not enter the training epoch loop", **out)
    elif available < total_steps:
        rep.fail("D.steps.epoch_budget", "epoch loop will exhaust before total_training_steps", **out)
    else:
        rep.pass_("D.steps.epoch_budget", "epoch loop can reach total_training_steps", **out)
    for key in ["save_freq", "test_freq"]:
        value = int(get(cfg, f"trainer.{key}", -1) or -1)
        if val_only and value <= 0:
            rep.pass_(f"D.steps.{key}", f"trainer.{key} is correctly disabled for val-only", value=value)
        elif value <= 0:
            rep.warn(f"D.steps.{key}", f"trainer.{key} is disabled", value=value)
        elif total_steps and value > total_steps:
            rep.warn(f"D.steps.{key}", f"trainer.{key} exceeds requested step count", value=value, total_training_steps=total_steps)
        else:
            rep.pass_(f"D.steps.{key}", f"trainer.{key} is reachable", value=value, total_training_steps=total_steps)
    actor_epochs = int(get(cfg, "actor_rollout_ref.actor.ppo_epochs", 1) or 1)
    rep.pass_("D.steps.epoch_names", "distinguished trainer.total_epochs from actor.ppo_epochs", trainer_total_epochs=total_epochs, actor_ppo_epochs=actor_epochs)
    return out


def check_batches(rep: Reporter, cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    semantics = detect_batch_semantics(root)
    train_prompts = int(get(cfg, "data.train_batch_size", 0) or 0)
    n = int(get(cfg, "actor_rollout_ref.rollout.n", 0) or 0)
    responses = train_prompts * n
    prompt_mini = int(get(cfg, "actor_rollout_ref.actor.ppo_mini_batch_size", 0) or 0)
    normalized_mini = prompt_mini * n if semantics["actor_prompt_level_then_times_rollout_n"] else prompt_mini
    world = int(get(cfg, "trainer.nnodes", 1) or 1) * int(get(cfg, "trainer.n_gpus_per_node", 1) or 1)
    tp = int(get(cfg, "actor_rollout_ref.rollout.tensor_model_parallel_size", 1) or 1)
    sp = int(get(cfg, "actor_rollout_ref.actor.ulysses_sequence_parallel_size", 1) or 1)
    denom = max(tp * sp, 1)
    dp = world // denom if world % denom == 0 else 0
    out = {
        "train_prompt_batch_prompts": train_prompts,
        "rollout_n_responses_per_prompt": n,
        "responses_per_step_responses": responses,
        "ppo_mini_batch_size_prompt_level_prompts": prompt_mini,
        "normalized_ppo_mini_batch_size_responses": normalized_mini,
        "world_size_gpus": world,
        "tensor_parallel_size": tp,
        "sequence_parallel_size": sp,
        "effective_data_parallel_size": dp,
        "batch_semantics": semantics,
    }
    if not semantics["supported"]:
        rep.fail("E.batch.semantics", "unsupported/unknown local batch semantics", **semantics)
        return out
    rep.pass_("E.batch.semantics", "local actor update multiplies prompt-level ppo_mini_batch_size by rollout.n", **semantics)
    if n <= 1:
        rep.fail("E.grpo.rollout_n", "GRPO requires rollout.n > 1", rollout_n=n)
    else:
        rep.pass_("E.grpo.rollout_n", "rollout.n is valid for GRPO", rollout_n=n)
    if normalized_mini <= 0 or responses <= 0:
        rep.fail("E.batch.positive", "batch sizes must be positive", **out)
    elif responses % normalized_mini != 0:
        rep.fail("E.batch.divisibility", "responses_per_step must be divisible by normalized mini-batch size", **out)
    else:
        out["mini_batches_per_rollout"] = responses // normalized_mini
        rep.pass_("E.batch.divisibility", "normalized mini-batch divides rollout batch", **out)
    if dp <= 0:
        rep.fail("E.parallel.dp", "world size is not divisible by tensor*sequence parallel size", **out)
    else:
        rep.pass_("E.parallel.dp", "parallel topology yields positive data parallel size", **out)
    per_gpu_micro = get(cfg, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu")
    if per_gpu_micro is not None and dp > 0 and normalized_mini > 0:
        global_micro = int(per_gpu_micro) * dp
        out["actor_micro_batch_size_global_responses"] = global_micro
        if global_micro <= 0 or normalized_mini % global_micro != 0:
            rep.fail("E.batch.actor_micro", "actor micro-batch does not divide normalized mini-batch", **out)
        else:
            rep.pass_("E.batch.actor_micro", "actor micro-batch divides normalized mini-batch", **out)
    if n > 0:
        groups = [list(range(i * n, (i + 1) * n)) for i in range(train_prompts)]
        exact = all(len(group) == n for group in groups)
        rep.pass_("E.grpo.group_shape", "simulated groups contain exactly rollout.n responses" if exact else "bad group shape", group_count=len(groups), rollout_n=n)
    return out


def load_tokenizer_and_config(rep: Reporter, cfg: dict[str, Any], model_path: str | None) -> tuple[Any | None, Any | None, dict[str, Any]]:
    info: dict[str, Any] = {}
    if not model_path or not Path(model_path).exists():
        rep.warn("F.model.deep", "deep model/tokenizer checks skipped because local model path is unavailable", model_path=model_path)
        return None, None, info
    try:
        from transformers import AutoConfig, AutoTokenizer

        auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        info = {
            "model_type": getattr(auto_cfg, "model_type", None),
            "max_position_embeddings": getattr(auto_cfg, "max_position_embeddings", None),
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        }
        if tokenizer.eos_token_id is None or not getattr(tokenizer, "chat_template", None):
            rep.fail("F.model.tokenizer", "tokenizer lacks eos token or chat template", **info)
        else:
            rep.pass_("F.model.tokenizer", "tokenizer/config loaded on CPU without model weights", **info)
        return tokenizer, auto_cfg, info
    except Exception as exc:
        rep.fail("F.model.deep", "failed to load AutoConfig/tokenizer on CPU", model_path=model_path, error=str(exc))
        return None, None, info


def check_context_lengths(rep: Reporter, cfg: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    max_prompt = int(get(cfg, "data.max_prompt_length", 0) or 0)
    max_response = int(get(cfg, "data.max_response_length", 0) or 0)
    total = max_prompt + max_response
    model_limit = model_info.get("max_position_embeddings")
    rollout_limit = get(cfg, "actor_rollout_ref.rollout.max_model_len") or model_limit
    out = {"max_prompt_tokens": max_prompt, "max_response_tokens": max_response, "total_context_tokens": total, "model_context_tokens": model_limit, "rollout_context_tokens": rollout_limit}
    if rollout_limit and total > int(rollout_limit):
        rep.fail("F.length.context", "max_prompt_length + max_response_length exceeds rollout context", **out)
    elif model_limit and total > int(model_limit):
        rep.fail("F.length.context", "max_prompt_length + max_response_length exceeds model context", **out)
    else:
        rep.pass_("F.length.context", "configured prompt+response length fits known context limits", **out)
    return out


def check_reward(rep: Reporter, cfg: dict[str, Any], mode: str, strict_formal: bool, root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kwargs = get(cfg, "reward.custom_reward_function.reward_kwargs", {}) or {}
    timeout_mode = str(kwargs.get("verify_timeout_mode") or os.environ.get("SIGNAL_FORGE_VERIFY_TIMEOUT_MODE") or "inline")
    timeout_seconds = kwargs.get("verify_timeout_seconds") or os.environ.get("SIGNAL_FORGE_VERIFY_TIMEOUT_SECONDS")
    fallback = kwargs.get("verify_timeout_fallback", os.environ.get("SIGNAL_FORGE_VERIFY_TIMEOUT_FALLBACK"))
    out.update({"verify_timeout_mode": timeout_mode, "verify_timeout_seconds": timeout_seconds, "verify_timeout_fallback": fallback})
    manager = root / "verl" / "verl" / "experimental" / "reward_loop" / "reward_manager" / "naive.py"
    manager_text = manager.read_text(encoding="utf-8") if manager.exists() else ""
    adapter_path = root / "src" / "RewardScope" / "src" / "rewardscope" / "verification" / "math_verify.py"
    adapter_text = adapter_path.read_text(encoding="utf-8") if adapter_path.exists() else ""
    threaded = "run_in_executor" in manager_text
    parsing_none = "parsing_timeout=None" in adapter_text
    out["reward_execution"] = "Ray worker process -> asyncio task -> ThreadPoolExecutor" if threaded else "unknown"
    out["rewardscope_parsing_timeout_none"] = parsing_none
    if threaded and parsing_none:
        rep.pass_("G.reward.thread_compat", "RewardScope disables Math-Verify signal timeout on threaded reward path", **out)
    elif threaded:
        rep.fail("G.reward.thread_compat", "Math-Verify signal timeout may run in ThreadPoolExecutor reward path", **out)
    else:
        rep.warn("G.reward.thread_compat", "could not prove reward thread execution model", **out)
    if timeout_mode == "process" and timeout_seconds:
        rep.pass_("G.reward.external_timeout", "process-isolated verifier hard timeout is configured", **out)
    elif strict_formal:
        rep.fail("G.reward.external_timeout", "parsing_timeout=None has no configured external hard timeout under strict/formal mode", **out)
    else:
        rep.warn("G.reward.external_timeout", "parsing_timeout=None has no configured external hard timeout in smoke mode", **out)
    if mode != "deep":
        rep.pass_("G.reward.probe", "reward adapter runtime probe skipped in fast mode")
        rep.pass_("G.reward.advantage", "GRPO advantage runtime simulation skipped in fast mode")
        return out

    try:
        from concurrent.futures import ThreadPoolExecutor
        from signal_forge.rewards.math_verify_adapter import compute_score

        probes = [
            ("correct", "The final answer is \\boxed{2}.", "2"),
            ("wrong", "The final answer is \\boxed{3}.", "2"),
            ("malformed", "No boxed answer here", "2"),
            ("equivalent", "The final answer is \\boxed{1/2}.", "0.5"),
        ]
        results = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for label, response, gt in probes:
                results.append((label, pool.submit(compute_score, "gsm8k", response, gt, **kwargs).result()))
        scores = {label: float(result["score"]) for label, result in results}
        if scores["correct"] > scores["wrong"] and scores["correct"] > scores["malformed"]:
            rep.pass_("G.reward.probe", "reward adapter threaded probe returns finite ordered scores", scores=scores)
        else:
            rep.fail("G.reward.probe", "reward adapter probe ordering is unexpected", scores=scores)
        out["probe_scores"] = scores
    except Exception as exc:
        rep.fail("G.reward.probe", "reward adapter threaded probe failed", error=str(exc))
    try:
        n = int(get(cfg, "actor_rollout_ref.rollout.n", 8) or 8)
        all_wrong = simulate_grpo(n, [0.0] * n)
        mixed = simulate_grpo(n, [1.0 if i % 2 else 0.0 for i in range(n)])
        all_correct = simulate_grpo(n, [1.0] * n)
        if all_wrong["advantages_finite"] and mixed["advantages_finite"] and all_correct["advantages_finite"]:
            rep.pass_("G.reward.advantage", "GRPO advantage simulation is finite for all-wrong/mixed/all-correct groups", all_wrong=all_wrong, mixed=mixed, all_correct=all_correct)
        else:
            rep.fail("G.reward.advantage", "GRPO advantage simulation produced non-finite values", all_wrong=all_wrong, mixed=mixed, all_correct=all_correct)
        if mixed["advantage_std"] <= 0:
            rep.warn("G.reward.signal", "mixed probe has near-zero advantage variance", mixed=mixed)
    except Exception as exc:
        rep.fail("G.reward.advantage", "GRPO advantage simulation failed", error=str(exc))
    return out



def validate_reward_outputs(rep: Reporter, outputs: list[Any], expected_count: int) -> None:
    if len(outputs) != expected_count:
        rep.fail("G.reward.output_count", "reward output count does not match response count", expected=expected_count, actual=len(outputs))
        return
    bad = []
    for index, item in enumerate(outputs):
        score = item.get("score", item) if isinstance(item, dict) else item
        try:
            value = float(score)
        except Exception:
            bad.append({"index": index, "score": repr(score), "reason": "not_float"})
            continue
        if not math.isfinite(value):
            bad.append({"index": index, "score": value, "reason": "non_finite"})
    if bad:
        rep.fail("G.reward.finite", "reward output contains NaN/Inf or non-scalar values", bad=bad)
    else:
        rep.pass_("G.reward.finite", "reward output count and scalar finiteness are valid", expected=expected_count)

def check_algorithm(rep: Reporter, cfg: dict[str, Any]) -> dict[str, Any]:
    out = {key: get(cfg, key) for key in IMPORTANT_KEYS if key.startswith("algorithm.") or key.startswith("actor_rollout_ref.actor.") or key.startswith("actor_rollout_ref.rollout.")}
    adv = str(get(cfg, "algorithm.adv_estimator"))
    if adv != "grpo":
        rep.fail("H.algorithm.adv_estimator", "advantage estimator is not GRPO", adv_estimator=adv)
    else:
        rep.pass_("H.algorithm.adv_estimator", "advantage estimator is GRPO", adv_estimator=adv)
    if bool(get(cfg, "critic.enable", False)):
        rep.warn("H.algorithm.critic", "critic appears enabled for GRPO recipe", critic_enable=get(cfg, "critic.enable"))
    else:
        rep.pass_("H.algorithm.critic", "critic is not enabled for GRPO recipe")
    if bool(get(cfg, "algorithm.use_kl_in_reward", False)) and bool(get(cfg, "actor_rollout_ref.actor.use_kl_loss", False)):
        rep.warn("H.algorithm.kl_double", "KL appears enabled both in reward and actor loss")
    else:
        rep.pass_("H.algorithm.kl_double", "no double KL configuration detected")
    rep.pass_("H.algorithm.print", "algorithm hyperparameters captured", **out)
    return out


def check_runtime(rep: Reporter, cfg: dict[str, Any], strict_formal: bool = False) -> dict[str, Any]:
    requested = int(get(cfg, "trainer.n_gpus_per_node", 0) or 0) * int(get(cfg, "trainer.nnodes", 1) or 1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_count = None if visible in (None, "") else len([x for x in visible.split(",") if x.strip()])
    nvidia = run_text(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"], timeout=3.0)
    nvidia_count = len([line for line in nvidia.splitlines() if line.strip()]) if nvidia else 0
    out = {"requested_gpus": requested, "CUDA_VISIBLE_DEVICES": visible, "visible_count": visible_count, "nvidia_smi_count": nvidia_count, "nvidia_smi": nvidia}
    available = visible_count if visible_count is not None else nvidia_count
    allow_no_gpu_boot = os.environ.get("ALLOW_NO_GPU_BOOT", "1").lower() in {"1", "true", "yes", "on"}
    require_gpu_for_preflight = os.environ.get("REQUIRE_GPU_FOR_PREFLIGHT", "0").lower() in {"1", "true", "yes", "on"}
    out["allow_no_gpu_boot"] = allow_no_gpu_boot
    out["require_gpu_for_preflight"] = require_gpu_for_preflight
    if requested and available and requested > available:
        rep.fail("I.gpu.count", "requested GPU count exceeds visible/available GPUs", **out)
    elif requested and not available:
        if strict_formal and require_gpu_for_preflight and not allow_no_gpu_boot:
            rep.fail("I.gpu.count", "requested GPUs are not visible in this environment", **out)
        elif allow_no_gpu_boot and not require_gpu_for_preflight:
            rep.pass_("I.gpu.count", "requested GPUs are not visible; no-GPU boot is explicitly allowed for setup/preflight", **out)
        else:
            rep.warn("I.gpu.count", "requested GPUs are not visible in this no-card/static environment", **out)
    elif requested:
        rep.pass_("I.gpu.count", "requested GPU count is compatible with visible/available GPUs", **out)
    else:
        rep.warn("I.gpu.count", "requested GPU count is zero or unknown", **out)
    shm = shutil.disk_usage("/dev/shm") if Path("/dev/shm").exists() else None
    disk = shutil.disk_usage(os.getcwd())
    out["disk_free_gib"] = round(disk.free / 1024**3, 2)
    out["dev_shm_free_gib"] = round(shm.free / 1024**3, 2) if shm else None
    rep.pass_("I.resources.disk", "disk and /dev/shm inspected", **out)
    ray = run_text(["pgrep", "-af", "ray"], timeout=2.0)
    if ray:
        rep.warn("I.resources.ray", "stale Ray processes may exist; preflight does not kill them", processes=ray.splitlines()[:20])
    else:
        rep.pass_("I.resources.ray", "no Ray processes detected by pgrep")
    gpu_util = float(get(cfg, "actor_rollout_ref.rollout.gpu_memory_utilization", 0) or 0)
    if not (0 < gpu_util <= 1):
        rep.fail("I.vllm.gpu_memory_utilization", "vLLM GPU memory utilization must be in (0, 1]", value=gpu_util)
    else:
        rep.pass_("I.vllm.gpu_memory_utilization", "vLLM GPU memory utilization is in range", value=gpu_util)
    return out


def check_logging(rep: Reporter, cfg: dict[str, Any], launch_script: Path, paths: dict[str, str | None]) -> dict[str, Any]:
    logger_cfg = get(cfg, "trainer.logger", [])
    loggers = [str(x).lower() for x in as_list(logger_cfg)]
    out = {
        "WANDB_MODE": os.environ.get("WANDB_MODE"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "WANDB_NAME": os.environ.get("WANDB_NAME"),
        "WANDB_DIR": os.environ.get("WANDB_DIR"),
        "trainer_logger": loggers,
    }
    if "wandb" not in loggers and os.environ.get("EXPECT_WANDB", "0").lower() in {"1", "true", "yes"}:
        rep.fail("J.logging.wandb", "W&B logging expected but trainer.logger does not include wandb", **out)
    elif "wandb" in loggers and os.environ.get("WANDB_MODE") == "disabled":
        rep.warn("J.logging.wandb", "trainer.logger includes wandb but WANDB_MODE=disabled", **out)
    else:
        rep.pass_("J.logging.wandb", "W&B logging configuration inspected", **out)
    text = launch_script.read_text(encoding="utf-8") if launch_script.exists() else ""
    if "set -euo pipefail" in text:
        rep.pass_("J.shell.pipefail", "launch script uses set -euo pipefail")
    else:
        rep.fail("J.shell.pipefail", "launch script does not use set -euo pipefail")
    if "| tee" in text and "pipefail" not in text:
        rep.warn("J.shell.tee", "tee pipeline may hide training exit status")
    elif "| tee" in text:
        rep.pass_("J.shell.tee", "tee pipeline is protected by pipefail")
    exp = str(get(cfg, "trainer.experiment_name", ""))
    project = str(get(cfg, "trainer.project_name", ""))
    if not exp or not project:
        rep.fail("J.logging.names", "project/experiment names must be nonempty", project=project, experiment=exp)
    else:
        rep.pass_("J.logging.names", "project/experiment names are set", project=project, experiment=exp)
    return out


def benchmark_parser(rep: Reporter, enabled: bool, reward_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    cases = {
        "valid_boxed_integer": ("gsm8k", "The final answer is \\boxed{2}.", "2"),
        "fraction": ("gsm8k", "The final answer is \\boxed{1/2}.", "0.5"),
        "equivalent_expression": ("gsm8k", "The final answer is \\boxed{1+1}.", "2"),
        "malformed_latex": ("gsm8k", "The final answer is \\boxed{\\frac{1}{ }.", "1"),
        "missing_boxed": ("gsm8k", "The answer is 2.", "2"),
        "multiple_boxed": ("gsm8k", "First \\boxed{1}, final \\boxed{2}.", "2"),
        "very_long_response": ("gsm8k", "scratch " * 500 + "\\boxed{2}", "2"),
        "deeply_nested": ("gsm8k", "\\boxed{" + "{" * 24 + "2" + "}" * 24 + "}", "2"),
    }
    if not enabled:
        return {"enabled": False}
    from signal_forge.rewards.math_verify_adapter import compute_score

    kwargs = dict(reward_kwargs or {})
    kwargs.setdefault("verify_timeout_mode", "process")
    kwargs.setdefault("verify_timeout_seconds", float(os.environ.get("PREFLIGHT_PARSER_TIMEOUT_SECONDS", "3")))
    kwargs.setdefault("verify_timeout_fallback", True)
    kwargs.setdefault("verify_timeout_fallback_score", 0.0)
    kwargs.setdefault("verifier_max_input_chars", int(os.environ.get("PREFLIGHT_VERIFIER_MAX_INPUT_CHARS", "20000")))
    latencies = []
    extraction_failures = 0
    exceptions = 0
    per_case = {}
    for name, (source, response, gt) in cases.items():
        started = time.perf_counter()
        try:
            result = compute_score(source, response, gt, **kwargs)
            latency = (time.perf_counter() - started) * 1000.0
            extraction_failures += int(not result.get("extraction_ok"))
            per_case[name] = {"latency_ms": latency, "score": result.get("score"), "extraction_ok": result.get("extraction_ok")}
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000.0
            exceptions += 1
            per_case[name] = {"latency_ms": latency, "exception": f"{type(exc).__name__}: {exc}"}
        latencies.append(latency)
    out = {
        "enabled": True,
        "count": len(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_max": max(latencies),
        "extraction_failures": extraction_failures,
        "exceptions": exceptions,
        "cases": per_case,
    }
    rep.pass_("G.reward.parser_benchmark", "representative parser benchmark completed", **out)
    return out



def fallback_base_config() -> dict[str, Any]:
    return {
        "algorithm": {"adv_estimator": "gae", "use_kl_in_reward": False, "norm_adv_by_std_in_grpo": True},
        "data": {
            "train_files": None,
            "val_files": None,
            "train_batch_size": 1024,
            "max_prompt_length": 512,
            "max_response_length": 512,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "prompt_key": "prompt",
            "train_max_samples": -1,
            "val_max_samples": -1,
        },
        "actor_rollout_ref": {
            "model": {"path": None, "trust_remote_code": False},
            "actor": {
                "ppo_mini_batch_size": 256,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": None,
                "ppo_epochs": 1,
                "use_dynamic_bsz": False,
                "use_kl_loss": False,
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "entropy_coeff": 0,
                "ulysses_sequence_parallel_size": 1,
                "optim": {"lr": 1e-6},
            },
            "rollout": {
                "name": None,
                "n": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "max_model_len": None,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 1024,
                "val_kwargs": {"n": 1, "temperature": 0, "top_p": 1.0, "top_k": -1, "do_sample": False},
            },
            "ref": {},
        },
        "critic": {"enable": None},
        "reward": {
            "reward_manager": {"source": "register", "name": "naive"},
            "custom_reward_function": {"path": None, "name": "compute_score", "reward_kwargs": {}},
            "reward_model": {"enable": False},
        },
        "trainer": {
            "logger": ["console"],
            "project_name": None,
            "experiment_name": None,
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "default_local_dir": None,
            "rollout_data_dir": None,
            "validation_data_dir": None,
            "total_epochs": 1,
            "total_training_steps": None,
            "save_freq": -1,
            "test_freq": -1,
            "resume_mode": "disable",
            "resume_from_path": None,
            "use_v1": False,
        },
        "_preflight_config_source": "launch_array_fallback",
    }


def write_fallback_config(path: Path, overrides: list[str]) -> None:
    cfg = apply_overrides(fallback_base_config(), overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

def run_preflight(args: argparse.Namespace) -> tuple[Reporter, dict[str, Any]]:
    root = Path(args.project_root).resolve()
    launch = Path(args.launch_script).resolve()
    cfg = load_config(Path(args.resolved_config))
    cfg = apply_overrides(cfg, args.override or [])
    rep = Reporter(strict=args.strict)
    report: dict[str, Any] = {"mode": args.mode, "strict": args.strict, "formal": args.formal, "started_at": time.time()}
    report["provenance"] = check_provenance(rep, cfg, root, launch, args.mode)
    paths = check_paths(rep, cfg, root, allow_existing_output=args.allow_existing_output)
    report["paths"] = paths
    tokenizer = auto_cfg = None
    model_info: dict[str, Any] = {}
    if args.mode == "deep":
        tokenizer, auto_cfg, model_info = load_tokenizer_and_config(rep, cfg, paths.get("model"))
    report["dataset"] = check_dataset(rep, cfg, paths, args.mode == "deep", tokenizer)
    report["steps"] = check_steps(rep, cfg, report["dataset"])
    report["batches"] = check_batches(rep, cfg, root)
    if args.mode == "deep":
        report["model"] = model_info
    report["context"] = check_context_lengths(rep, cfg, model_info)
    report["reward"] = check_reward(rep, cfg, args.mode, args.strict or args.formal, root)
    report["parser_benchmark"] = benchmark_parser(
        rep,
        args.mode == "deep" and args.benchmark_parser,
        get(cfg, "reward.custom_reward_function.reward_kwargs", {}) or {},
    )
    report["algorithm"] = check_algorithm(rep, cfg)
    report["runtime"] = check_runtime(rep, cfg, args.strict or args.formal)
    report["logging"] = check_logging(rep, cfg, launch, paths)
    report["checks"] = [c.__dict__ for c in rep.checks]
    report["summary"] = {"pass": sum(c.status == "PASS" for c in rep.checks), "warn": rep.warn_count, "fail": rep.fail_count}
    report["finished_at"] = time.time()
    return rep, report


def print_human(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"Preflight summary: PASS={summary['pass']} WARN={summary['warn']} FAIL={summary['fail']}")
    print("Derived quantities:")
    batches = report.get("batches", {})
    steps = report.get("steps", {})
    dataset = report.get("dataset", {})
    context = report.get("context", {})
    lines = [
        ("train prompt batch", batches.get("train_prompt_batch_prompts"), "prompts"),
        ("rollout.n", batches.get("rollout_n_responses_per_prompt"), "responses/prompt"),
        ("responses per optimizer step", batches.get("responses_per_step_responses"), "responses"),
        ("normalized PPO mini-batch", batches.get("normalized_ppo_mini_batch_size_responses"), "responses"),
        ("mini-batches per rollout", batches.get("mini_batches_per_rollout"), "mini-batches"),
        ("world size", batches.get("world_size_gpus"), "GPUs"),
        ("effective data parallel size", batches.get("effective_data_parallel_size"), "per-GPU partitions"),
        ("steps per epoch", steps.get("steps_per_epoch"), "steps"),
        ("trainer total epochs", steps.get("total_epochs"), "epochs"),
        ("requested total training steps", steps.get("total_training_steps"), "steps"),
        ("train raw rows", dataset.get("train", {}).get("raw_rows"), "prompts"),
        ("train effective rows", dataset.get("train", {}).get("effective_rows"), "prompts"),
        ("max prompt length", context.get("max_prompt_tokens"), "tokens"),
        ("max response length", context.get("max_response_tokens"), "tokens"),
        ("total context", context.get("total_context_tokens"), "tokens"),
    ]
    for label, value, unit in lines:
        if value is not None:
            print(f"  - {label}: {value} {unit}")
    for check in report["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--launch-script", default="src/scripts_a0/08_run_a0_0p5b_regression.sh")
    parser.add_argument("--resolved-config")
    parser.add_argument("--write-fallback-config", default=None)
    parser.add_argument("--mode", choices=["fast", "deep"], default="fast")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--benchmark-parser", action="store_true")
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    if args.write_fallback_config:
        write_fallback_config(Path(args.write_fallback_config), args.override or [])
        return 0
    if not args.resolved_config:
        parser.error("--resolved-config is required unless --write-fallback-config is used")

    rep, report = run_preflight(args)
    print_human(report)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.json_report:
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_report).write_text(payload + "\n", encoding="utf-8")
        print(f"JSON report: {args.json_report}")
    else:
        print(payload)
    return 1 if rep.fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
