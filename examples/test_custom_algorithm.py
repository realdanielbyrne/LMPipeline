#!/usr/bin/env python3
"""
Comprehensive test suite for custom algorithm implementation.

This demonstrates testing best practices for custom LMPipeline algorithms,
including unit tests, integration tests, and mocking strategies.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the custom algorithm (adjust import path as needed)
try:
    from custom_tokenizer_stage import CustomTokenizerStage, CustomTokenizerConfig
except ImportError:
    # Alternative import for when running from different directory
    import sys

    sys.path.append(".")
    from custom_tokenizer_stage import CustomTokenizerStage, CustomTokenizerConfig

from src.lmpipeline.algorithms.base import StageResult
from lmpipeline.algorithms.base import StageResult as LMPipelineStageResult
from src.lmpipeline.pipeline import Pipeline, PipelineConfig


class TestCustomTokenizerConfig(unittest.TestCase):
    """Test the custom tokenizer configuration class."""

    def test_config_creation(self):
        """Test creating configuration with default values."""
        config = CustomTokenizerConfig(
            stage_name="test_tokenizer", output_dir="/tmp/test"
        )

        self.assertEqual(config.stage_name, "test_tokenizer")
        self.assertEqual(config.output_dir, "/tmp/test")
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.tokenizer_type, "bpe")
        self.assertTrue(config.enabled)

    def test_config_with_custom_values(self):
        """Test creating configuration with custom values."""
        config = CustomTokenizerConfig(
            stage_name="custom_test",
            output_dir="/tmp/custom",
            vocab_size=16000,
            tokenizer_type="wordpiece",
            min_frequency=5,
        )

        self.assertEqual(config.vocab_size, 16000)
        self.assertEqual(config.tokenizer_type, "wordpiece")
        self.assertEqual(config.min_frequency, 5)


class TestCustomTokenizerStage(unittest.TestCase):
    """Test the custom tokenizer stage implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        # Create a sample corpus file
        self.corpus_path = Path(self.temp_dir) / "test_corpus.txt"
        sample_text = "This is a test corpus for tokenizer training.\n" * 100
        self.corpus_path.write_text(sample_text)

        # Create configuration
        self.config = CustomTokenizerConfig(
            stage_name="test_tokenizer",
            output_dir=self.temp_dir,
            vocab_size=1000,
            training_corpus_path=str(self.corpus_path),
            min_frequency=1,
        )

        # Create stage instance
        self.stage = CustomTokenizerStage(self.config)

    def test_stage_name(self):
        """Test stage name property."""
        self.assertEqual(self.stage.stage_name, "custom_tokenizer")

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        # Should not raise any exception
        self.stage.validate_config()

    def test_config_validation_invalid_vocab_size(self):
        """Test validation failure for invalid vocabulary size."""
        self.config.vocab_size = -1
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("Vocabulary size must be positive", str(context.exception))

    def test_config_validation_missing_corpus(self):
        """Test validation failure for missing corpus file."""
        self.config.training_corpus_path = "nonexistent_file.txt"
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("Training corpus not found", str(context.exception))

    def test_config_validation_invalid_tokenizer_type(self):
        """Test validation failure for invalid tokenizer type."""
        self.config.tokenizer_type = "invalid_type"
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("Tokenizer type must be one of", str(context.exception))

    def test_config_validation_invalid_min_frequency(self):
        """Test validation failure for invalid minimum frequency."""
        self.config.min_frequency = 0
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("Minimum frequency must be at least 1", str(context.exception))

    def test_get_corpus_size(self):
        """Test corpus size calculation."""
        size = self.stage._get_corpus_size()
        self.assertEqual(size, 100)  # 100 lines in our test corpus

    def test_get_corpus_size_nonexistent_file(self):
        """Test corpus size calculation for nonexistent file."""
        self.config.training_corpus_path = "nonexistent.txt"
        size = self.stage._get_corpus_size()
        self.assertEqual(size, 0)

    def test_train_tokenizer_config_validation(self):
        """Test tokenizer training configuration validation."""
        # Test different tokenizer types
        for tokenizer_type in ["bpe", "wordpiece", "unigram"]:
            self.config.tokenizer_type = tokenizer_type
            # Should not raise exception for valid types
            self.stage.validate_config()

    def test_resize_model_embeddings(self):
        """Test model embedding resizing."""
        # Create mock model with config
        mock_model_instance = Mock()
        mock_model_instance.config.vocab_size = 1000
        mock_model_instance.resize_token_embeddings = Mock()

        # Create mock tokenizer with different vocab size
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.get_vocab.return_value = {
            f"token_{i}": i for i in range(2000)
        }  # 2000 tokens

        # Test resizing
        result = self.stage._resize_model_embeddings(
            mock_model_instance, mock_tokenizer_instance
        )

        # Verify resize was called
        mock_model_instance.resize_token_embeddings.assert_called_once_with(2000)
        self.assertEqual(result, mock_model_instance)

    @patch.object(CustomTokenizerStage, "_train_tokenizer")
    @patch.object(CustomTokenizerStage, "_resize_model_embeddings")
    def test_execute_success(self, mock_resize, mock_train):
        """Test successful stage execution."""
        # Setup mocks
        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {f"token_{i}": i for i in range(500)}

        mock_new_tokenizer = Mock()
        mock_new_tokenizer.get_vocab.return_value = {
            f"token_{i}": i for i in range(1000)
        }
        mock_new_tokenizer.save_pretrained = Mock()

        # Mock the internal methods
        mock_train.return_value = mock_new_tokenizer
        mock_resize.return_value = mock_model

        # Execute the stage
        result = self.stage.execute(mock_model, mock_tokenizer)

        # Verify results
        self.assertIsInstance(result, (StageResult, LMPipelineStageResult))
        self.assertTrue(result.success)
        self.assertEqual(result.stage_name, "custom_tokenizer")
        self.assertIn("original_vocab_size", result.metrics)
        self.assertIn("new_vocab_size", result.metrics)
        self.assertIn("tokenizer_config", result.artifacts)

    def test_execute_failure_invalid_config(self):
        """Test stage execution failure due to invalid configuration."""
        # Use invalid configuration
        self.config.training_corpus_path = "nonexistent.txt"

        mock_model = Mock()
        mock_tokenizer = Mock()

        # Execute the stage
        result = self.stage.execute(mock_model, mock_tokenizer)

        # Verify failure
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.stage_name, "custom_tokenizer")


class TestPipelineIntegration(unittest.TestCase):
    """Test integration of custom algorithm with the pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        # Create a sample corpus file
        self.corpus_path = Path(self.temp_dir) / "test_corpus.txt"
        sample_text = "This is a test corpus for pipeline integration.\n" * 50
        self.corpus_path.write_text(sample_text)

    @patch("src.lmpipeline.pipeline.AutoModelForCausalLM")
    @patch("src.lmpipeline.pipeline.AutoTokenizer")
    def test_custom_algorithm_registration(self, mock_tokenizer, mock_model):
        """Test registering custom algorithm with pipeline."""
        # Register the custom stage
        Pipeline.register_stage("custom_tokenizer", CustomTokenizerStage)

        # Verify registration
        self.assertIn("custom_tokenizer", Pipeline.STAGE_REGISTRY)
        self.assertEqual(
            Pipeline.STAGE_REGISTRY["custom_tokenizer"], CustomTokenizerStage
        )

    def test_pipeline_config_creation(self):
        """Test creating pipeline configuration with custom algorithm."""
        # Register custom stage
        Pipeline.register_stage("custom_tokenizer", CustomTokenizerStage)

        # Create pipeline configuration
        config = PipelineConfig(
            model_name_or_path="microsoft/DialoGPT-small",
            output_dir=self.temp_dir,
            stages=["custom_tokenizer"],
            stage_configs={
                "custom_tokenizer": {
                    "vocab_size": 1000,
                    "training_corpus_path": str(self.corpus_path),
                    "min_frequency": 1,
                    "tokenizer_type": "bpe",
                }
            },
        )

        # Verify configuration
        self.assertEqual(config.stages, ["custom_tokenizer"])
        self.assertIn("custom_tokenizer", config.stage_configs)
        self.assertEqual(config.stage_configs["custom_tokenizer"]["vocab_size"], 1000)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_missing_corpus_file(self):
        """Test handling of missing corpus file."""
        config = CustomTokenizerConfig(
            stage_name="test",
            output_dir=self.temp_dir,
            training_corpus_path="nonexistent.txt",
        )
        stage = CustomTokenizerStage(config)

        with self.assertRaises(ValueError):
            stage.validate_config()

    def test_empty_corpus_file(self):
        """Test handling of empty corpus file."""
        corpus_path = Path(self.temp_dir) / "empty_corpus.txt"
        corpus_path.write_text("")

        config = CustomTokenizerConfig(
            stage_name="test",
            output_dir=self.temp_dir,
            training_corpus_path=str(corpus_path),
        )
        stage = CustomTokenizerStage(config)

        # Should validate successfully
        stage.validate_config()

        # Corpus size should be 0
        self.assertEqual(stage._get_corpus_size(), 0)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
