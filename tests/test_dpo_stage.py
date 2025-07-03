"""
Unit tests for DPO (Direct Preference Optimization) stage.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import AutoTokenizer

from src.lmpipeline.algorithms.dpo import (
    DPOConfig,
    DPODataset,
    DPOStage,
    PreferenceDatasetFormatter,
)


class TestPreferenceDatasetFormatter(unittest.TestCase):
    """Test the PreferenceDatasetFormatter class."""

    def test_detect_standard_format(self):
        """Test detection of standard prompt-chosen-rejected format."""
        data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain ML",
                "chosen": "ML is machine learning.",
                "rejected": "No idea.",
            },
        ]

        format_keys = PreferenceDatasetFormatter.detect_format(data)
        self.assertEqual(format_keys, ("prompt", "chosen", "rejected"))

    def test_detect_instruction_format(self):
        """Test detection of instruction-preferred-rejected format."""
        data = [
            {
                "instruction": "What is AI?",
                "preferred": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "instruction": "Explain ML",
                "preferred": "ML is machine learning.",
                "rejected": "No idea.",
            },
        ]

        format_keys = PreferenceDatasetFormatter.detect_format(data)
        self.assertEqual(format_keys, ("instruction", "preferred", "rejected"))

    def test_convert_standard_format(self):
        """Test conversion of standard format."""
        item = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "I don't know.",
        }
        format_keys = ("prompt", "chosen", "rejected")

        result = PreferenceDatasetFormatter.convert_to_standard_format(
            item, format_keys
        )
        expected = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "I don't know.",
        }
        self.assertEqual(result, expected)

    def test_convert_instruction_format(self):
        """Test conversion of instruction format."""
        item = {
            "instruction": "What is AI?",
            "preferred": "AI is artificial intelligence.",
            "rejected": "I don't know.",
        }
        format_keys = ("instruction", "preferred", "rejected")

        result = PreferenceDatasetFormatter.convert_to_standard_format(
            item, format_keys
        )
        expected = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "I don't know.",
        }
        self.assertEqual(result, expected)

    def test_detect_format_empty_data(self):
        """Test error handling for empty dataset."""
        with self.assertRaises(ValueError) as context:
            PreferenceDatasetFormatter.detect_format([])
        self.assertIn("empty", str(context.exception))

    def test_detect_format_unknown(self):
        """Test error handling for unknown format."""
        data = [{"unknown1": "value1", "unknown2": "value2"}]
        with self.assertRaises(ValueError) as context:
            PreferenceDatasetFormatter.detect_format(data)
        self.assertIn("Could not detect", str(context.exception))


class TestDPODataset(unittest.TestCase):
    """Test the DPODataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.eos_token = "</s>"

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain ML",
                "chosen": "ML is machine learning.",
                "rejected": "No idea.",
            },
        ]

        dataset = DPODataset(
            data, self.tokenizer, max_length=512, auto_detect_format=True
        )
        self.assertEqual(len(dataset), 2)

    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
        ]

        dataset = DPODataset(
            data, self.tokenizer, max_length=512, auto_detect_format=True
        )
        item = dataset[0]

        self.assertIn("prompt", item)
        self.assertIn("chosen", item)
        self.assertIn("rejected", item)
        self.assertEqual(item["prompt"], "What is AI?")
        self.assertEqual(item["chosen"], "AI is artificial intelligence.")
        self.assertEqual(item["rejected"], "I don't know.")

    def test_dataset_auto_format_detection(self):
        """Test automatic format detection in dataset."""
        data = [
            {
                "instruction": "What is AI?",
                "preferred": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
        ]

        dataset = DPODataset(
            data, self.tokenizer, max_length=512, auto_detect_format=True
        )
        item = dataset[0]

        # Should be converted to standard format
        self.assertEqual(item["prompt"], "What is AI?")
        self.assertEqual(item["chosen"], "AI is artificial intelligence.")
        self.assertEqual(item["rejected"], "I don't know.")


class TestDPOConfig(unittest.TestCase):
    """Test the DPOConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DPOConfig(
            stage_name="dpo",
            output_dir="/tmp/test",
            preference_dataset_path="test.json",
        )

        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.num_train_epochs, 1)
        self.assertEqual(config.learning_rate, 5e-7)
        self.assertTrue(config.auto_detect_format)


class TestDPOStage(unittest.TestCase):
    """Test the DPOStage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DPOConfig(
            stage_name="dpo",
            output_dir=self.temp_dir,
            preference_dataset_path="test.json",
            beta=0.1,
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            validation_split=0.0,  # No validation for tests
        )
        self.stage = DPOStage(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stage_name(self):
        """Test stage name property."""
        self.assertEqual(self.stage.stage_name, "dpo")

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        # Should not raise any exception
        self.stage.validate_config()

    def test_validate_config_missing_dataset(self):
        """Test configuration validation with missing dataset path."""
        self.config.preference_dataset_path = ""
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("preference_dataset_path is required", str(context.exception))

    def test_validate_config_invalid_beta(self):
        """Test configuration validation with invalid beta."""
        self.config.beta = -0.1
        with self.assertRaises(ValueError) as context:
            self.stage.validate_config()
        self.assertIn("beta must be positive", str(context.exception))

    def test_load_preference_dataset_json(self):
        """Test loading preference dataset from JSON file."""
        # Create test data file
        test_data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain ML",
                "chosen": "ML is machine learning.",
                "rejected": "No idea.",
            },
        ]

        test_file = Path(self.temp_dir) / "test_preferences.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        self.config.preference_dataset_path = str(test_file)
        data = self.stage._load_preference_dataset()

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["prompt"], "What is AI?")

    def test_load_preference_dataset_jsonl(self):
        """Test loading preference dataset from JSONL file."""
        # Create test data file
        test_data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain ML",
                "chosen": "ML is machine learning.",
                "rejected": "No idea.",
            },
        ]

        test_file = Path(self.temp_dir) / "test_preferences.jsonl"
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        self.config.preference_dataset_path = str(test_file)
        data = self.stage._load_preference_dataset()

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["prompt"], "What is AI?")

    def test_split_preference_dataset(self):
        """Test splitting preference dataset."""
        data = [
            {
                "prompt": f"Question {i}",
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}",
            }
            for i in range(10)
        ]

        self.config.validation_split = 0.2
        train_data, val_data = self.stage._split_preference_dataset(data)

        self.assertEqual(len(train_data), 8)
        self.assertEqual(len(val_data), 2)

    def test_error_handling(self):
        """Test error handling functionality."""
        # Test memory error
        memory_error = Exception("CUDA out of memory")
        result = self.stage._handle_training_error(memory_error)
        self.assertIn("GPU memory error", result)
        self.assertIn("reducing batch size", result)

        # Test TRL import error
        import_error = Exception("No module named 'trl'")
        result = self.stage._handle_training_error(import_error)
        self.assertIn("TRL library not found", result)
        self.assertIn("pip install trl", result)

        # Test dataset format error
        format_error = Exception("dataset format key error")
        result = self.stage._handle_training_error(format_error)
        self.assertIn("Dataset format error", result)
        self.assertIn("auto_detect_format", result)


class TestDPOPipelineIntegration(unittest.TestCase):
    """Test DPO stage integration with the pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test preference dataset
        self.preference_data = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "Explain ML",
                "chosen": "ML is machine learning.",
                "rejected": "No idea.",
            },
            {
                "prompt": "What is NLP?",
                "chosen": "NLP is natural language processing.",
                "rejected": "Not sure.",
            },
        ]

        self.preference_file = Path(self.temp_dir) / "preferences.json"
        with open(self.preference_file, "w") as f:
            json.dump(self.preference_data, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dpo_stage_execution_mock_failure(self):
        """Test DPO stage execution when TRL is not available."""
        # Create DPO config
        config = DPOConfig(
            stage_name="dpo",
            output_dir=self.temp_dir,
            preference_dataset_path=str(self.preference_file),
            beta=0.1,
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            validation_split=0.0,
        )

        # Create DPO stage
        stage = DPOStage(config)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 1000000
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "</s>"

        # Execute stage - should fail due to missing TRL
        result = stage.execute(mock_model, mock_tokenizer)

        # Verify failure is handled properly (TRL not installed)
        self.assertFalse(result.success)
        self.assertEqual(result.stage_name, "dpo")
        self.assertIn("TRL library", result.error_message)

    def test_dpo_stage_with_previous_result(self):
        """Test DPO stage with previous stage result."""
        from src.lmpipeline.algorithms.base import StageResult

        # Create temporary directories for the mock paths
        sft_model_dir = Path(self.temp_dir) / "sft_model"
        sft_tokenizer_dir = Path(self.temp_dir) / "sft_tokenizer"
        sft_model_dir.mkdir()
        sft_tokenizer_dir.mkdir()

        # Create mock previous result (e.g., from SFT stage)
        previous_result = StageResult(
            stage_name="sft",
            success=True,
            model_path=str(sft_model_dir),
            tokenizer_path=str(sft_tokenizer_dir),
            metrics={"sft_loss": 0.3},
        )

        config = DPOConfig(
            stage_name="dpo",
            output_dir=self.temp_dir,
            preference_dataset_path=str(self.preference_file),
            beta=0.1,
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            validation_split=0.0,
        )

        stage = DPOStage(config)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 1000000
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "</s>"

        # Test prepare_model_and_tokenizer with previous result
        prepared_model, prepared_tokenizer = stage.prepare_model_and_tokenizer(
            mock_model, mock_tokenizer, previous_result
        )

        # Should return the same model and tokenizer (prepared)
        self.assertEqual(prepared_model, mock_model)
        self.assertEqual(prepared_tokenizer, mock_tokenizer)

    def test_dpo_stage_error_handling_integration(self):
        """Test DPO stage error handling in pipeline context."""
        # Create config with invalid dataset path
        config = DPOConfig(
            stage_name="dpo",
            output_dir=self.temp_dir,
            preference_dataset_path="nonexistent.json",  # Invalid path
            beta=0.1,
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            validation_split=0.0,
        )

        stage = DPOStage(config)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "</s>"

        # Execute stage - should fail gracefully
        result = stage.execute(mock_model, mock_tokenizer)

        # Verify failure is handled properly
        self.assertFalse(result.success)
        self.assertEqual(result.stage_name, "dpo")
        self.assertIn("Failed to load preference dataset", result.error_message)
        self.assertEqual(result.model_path, "")
        self.assertEqual(result.tokenizer_path, "")


if __name__ == "__main__":
    unittest.main()
