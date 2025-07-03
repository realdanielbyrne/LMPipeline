"""
Pipeline algorithms for modular fine-tuning.
"""

from .base import BaseStage, StageConfig, StageResult
from .sft import SFTStage, SFTConfig
from .dpo import DPOStage, DPOConfig
from .rlaif import RLAIFStage, RLAIFConfig
from .rl import RLStage, RLConfig
from .cot_distillation import CoTDistillationStage, CoTDistillationConfig

__all__ = [
    "BaseStage",
    "StageConfig",
    "StageResult",
    "SFTStage",
    "SFTConfig",
    "DPOStage",
    "DPOConfig",
    "RLAIFStage",
    "RLAIFConfig",
    "RLStage",
    "RLConfig",
    "CoTDistillationStage",
    "CoTDistillationConfig",
]
