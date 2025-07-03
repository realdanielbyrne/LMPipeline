"""
Fine-tuning package for language models with LoRA/QLoRA support.
Modular pipeline for SFT, DPO, RLAIF, RL, and CoT Distillation.
"""

__version__ = "0.1.0"

from .pipeline import Pipeline, PipelineConfig
from .stages import (
    BaseStage,
    SFTStage,
    DPOStage,
    RLAIFStage,
    RLStage,
    CoTDistillationStage,
)

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "BaseStage",
    "SFTStage",
    "DPOStage",
    "RLAIFStage",
    "RLStage",
    "CoTDistillationStage",
]
