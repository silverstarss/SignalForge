"""Minimal QAT core fallback for non-QAT GRPO runs.

The original file was accidentally excluded from one AutoDL bundle by a broad
core.* tar exclude rule. Experiment A0 does not enable QAT, so these definitions
only need to satisfy veRL imports and keep QAT disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn

from verl.base_config import BaseConfig


@dataclass
class QATConfig(BaseConfig):
    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    raise ValueError("quantization_config_path is required when QAT is enabled")


def apply_qat(model: nn.Module, config: QATConfig | dict[str, Any]) -> nn.Module:
    if isinstance(config, dict):
        config = QATConfig(**config)
    if getattr(config, "enable", False):
        raise RuntimeError("QAT is not available in this AutoDL bundle; disable QAT for A0/A.")
    return model


def enable_qat_fuse(model: nn.Module):
    model._qat_fuse_enabled = False


def invalidate_all_scales(model: nn.Module):
    return None
