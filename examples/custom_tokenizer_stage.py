"""
Example: Custom Tokenizer Training Stage
This algorithm fine-tunes a tokenizer on domain-specific data.

This is a complete working example demonstrating how to implement
a custom algorithm for the LMPipeline framework.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmpipeline.algorithms.base import BaseStage, StageConfig, StageResult

logger = logging.getLogger(__name__)


@dataclass
class CustomTokenizerConfig(StageConfig):
    """Configuration for custom tokenizer training."""
    
    # Tokenizer-specific parameters
    vocab_size: int = field(
        default=32000,
        metadata={"help": "Target vocabulary size for the tokenizer"}
    )
    training_corpus_path: str = field(
        default="",
        metadata={"help": "Path to text corpus for tokenizer training"}
    )
    special_tokens: list = field(
        default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>"],
        metadata={"help": "Special tokens to include in vocabulary"}
    )
    min_frequency: int = field(
        default=2,
        metadata={"help": "Minimum frequency for token inclusion"}
    )
    tokenizer_type: str = field(
        default="bpe",
        metadata={"help": "Type of tokenizer to train (bpe, wordpiece, unigram)"}
    )


class CustomTokenizerStage(BaseStage):
    """Custom tokenizer training algorithm."""
    
    def __init__(self, config: CustomTokenizerConfig):
        super().__init__(config)
        self.config: CustomTokenizerConfig = config
    
    @property
    def stage_name(self) -> str:
        return "custom_tokenizer"
    
    def validate_config(self) -> None:
        """Validate tokenizer configuration."""
        if self.config.vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")
            
        if not self.config.training_corpus_path:
            raise ValueError("Training corpus path is required")
            
        corpus_path = Path(self.config.training_corpus_path)
        if not corpus_path.exists():
            raise ValueError(f"Training corpus not found: {corpus_path}")
            
        if self.config.min_frequency < 1:
            raise ValueError("Minimum frequency must be at least 1")
            
        valid_types = ["bpe", "wordpiece", "unigram"]
        if self.config.tokenizer_type not in valid_types:
            raise ValueError(f"Tokenizer type must be one of: {valid_types}")
    
    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute tokenizer training."""
        try:
            self.logger.info("Starting custom tokenizer training")
            self.setup_logging()
            
            # Train the custom tokenizer
            new_tokenizer = self._train_tokenizer()
            
            # Resize model embeddings to match new vocabulary
            model = self._resize_model_embeddings(model, new_tokenizer)
            
            # Save the updated model and tokenizer
            model_path, tokenizer_path = self.save_model_and_tokenizer(
                model, new_tokenizer
            )
            
            # Collect metrics
            metrics = {
                "original_vocab_size": len(tokenizer.get_vocab()),
                "new_vocab_size": len(new_tokenizer.get_vocab()),
                "vocab_size_target": self.config.vocab_size,
                "training_corpus_size": self._get_corpus_size(),
                "tokenizer_type": self.config.tokenizer_type
            }
            
            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
                artifacts={
                    "tokenizer_config": f"{tokenizer_path}/tokenizer_config.json",
                    "vocab_file": f"{tokenizer_path}/vocab.json",
                    "training_corpus": self.config.training_corpus_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tokenizer training failed: {str(e)}")
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e)
            )
        finally:
            self.cleanup_logging()
    
    def _train_tokenizer(self) -> AutoTokenizer:
        """Train a new tokenizer on the corpus."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        
        # Initialize tokenizer based on type
        if self.config.tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE())
            trainer = trainers.BpeTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
                min_frequency=self.config.min_frequency
            )
        elif self.config.tokenizer_type == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece())
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
                min_frequency=self.config.min_frequency
            )
        elif self.config.tokenizer_type == "unigram":
            tokenizer = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
                min_frequency=self.config.min_frequency
            )
        
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Train on corpus
        self.logger.info(
            f"Training {self.config.tokenizer_type} tokenizer on "
            f"{self.config.training_corpus_path}"
        )
        tokenizer.train([self.config.training_corpus_path], trainer)
        
        # Save tokenizer temporarily and load as HuggingFace tokenizer
        temp_path = Path(self.config.output_dir) / "temp_tokenizer"
        temp_path.mkdir(exist_ok=True)
        tokenizer.save(str(temp_path / "tokenizer.json"))
        
        # Convert to HuggingFace tokenizer
        return AutoTokenizer.from_pretrained(str(temp_path), use_fast=True)
    
    def _resize_model_embeddings(self, model, new_tokenizer):
        """Resize model embeddings to match new vocabulary."""
        old_vocab_size = model.config.vocab_size
        new_vocab_size = len(new_tokenizer.get_vocab())
        
        if old_vocab_size != new_vocab_size:
            self.logger.info(
                f"Resizing embeddings: {old_vocab_size} -> {new_vocab_size}"
            )
            model.resize_token_embeddings(new_vocab_size)
            
        return model
    
    def _get_corpus_size(self) -> int:
        """Get the size of the training corpus."""
        try:
            with open(self.config.training_corpus_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0


# Example usage configuration
EXAMPLE_CONFIG = {
    "model_name_or_path": "microsoft/DialoGPT-small",
    "output_dir": "./outputs/custom_tokenizer_example",
    "stages": ["custom_tokenizer"],
    "stage_configs": {
        "custom_tokenizer": {
            "vocab_size": 16000,
            "training_corpus_path": "data/domain_corpus.txt",
            "special_tokens": ["<pad>", "<unk>", "<s>", "</s>", "<domain>"],
            "min_frequency": 3,
            "tokenizer_type": "bpe",
            "use_wandb": True,
            "wandb_project": "custom-tokenizer-experiments"
        }
    }
}


if __name__ == "__main__":
    """
    Example of how to use the custom tokenizer stage.
    This demonstrates both standalone usage and pipeline integration.
    """
    import tempfile
    from pathlib import Path
    
    # Create a sample corpus for demonstration
    temp_dir = tempfile.mkdtemp()
    corpus_path = Path(temp_dir) / "sample_corpus.txt"
    
    sample_text = """
    This is a sample domain-specific corpus for tokenizer training.
    It contains specialized vocabulary and terminology.
    The tokenizer will learn to efficiently encode this type of text.
    Domain-specific tokens like <domain> should be preserved.
    """
    
    corpus_path.write_text(sample_text * 100)  # Repeat for larger corpus
    
    # Create configuration
    config = CustomTokenizerConfig(
        stage_name="example_tokenizer",
        output_dir=temp_dir,
        vocab_size=1000,
        training_corpus_path=str(corpus_path),
        min_frequency=1
    )
    
    # Create and validate stage
    stage = CustomTokenizerStage(config)
    stage.validate_config()
    
    print(f"Custom tokenizer stage created successfully!")
    print(f"Stage name: {stage.stage_name}")
    print(f"Output directory: {config.output_dir}")
    print(f"Training corpus: {config.training_corpus_path}")
