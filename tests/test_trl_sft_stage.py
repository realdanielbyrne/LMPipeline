"""
Unit tests for TRL SFT Stage implementation.

Tests feature parity with existing SFT implementation, chat template handling,
dataset format detection, error handling, and integration with the pipeline framework.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig
from src.lmpipeline.algorithms.base import StageResult


class TestTRLSFTConfig(unittest.TestCase):
    """Test TRL SFT configuration class."""

    def test_config_creation(self):
        """Test creating TRL SFT configuration."""
        config = TRLSFTConfig(
            stage_name="test_trl_sft",
            output_dir="./test_output",
            dataset_name_or_path="test_dataset.jsonl",
            max_seq_length=1024,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            packing=True,
            packing_strategy="ffd",
            completion_only_loss=True,
            assistant_only_loss=False,
            neftune_noise_alpha=5.0,
            use_liger_kernel=True,
        )

        # Test basic SFT config inheritance
        self.assertEqual(config.stage_name, "test_trl_sft")
        self.assertEqual(config.output_dir, "./test_output")
        self.assertEqual(config.dataset_name_or_path, "test_dataset.jsonl")
        self.assertEqual(config.max_seq_length, 1024)
        self.assertEqual(config.num_train_epochs, 1)
        self.assertEqual(config.per_device_train_batch_size, 2)

        # Test TRL-specific options
        self.assertTrue(config.packing)
        self.assertEqual(config.packing_strategy, "ffd")
        self.assertTrue(config.completion_only_loss)
        self.assertFalse(config.assistant_only_loss)
        self.assertEqual(config.neftune_noise_alpha, 5.0)
        self.assertTrue(config.use_liger_kernel)

    def test_config_defaults(self):
        """Test default values in TRL SFT configuration."""
        config = TRLSFTConfig(
            stage_name="test", output_dir="./test", dataset_name_or_path="test.jsonl"
        )

        # Test TRL-specific defaults
        self.assertFalse(config.packing)
        self.assertEqual(config.packing_strategy, "ffd")
        self.assertIsNone(config.completion_only_loss)
        self.assertFalse(config.assistant_only_loss)
        self.assertIsNone(config.neftune_noise_alpha)
        self.assertFalse(config.use_liger_kernel)
        self.assertEqual(config.dataset_text_field, "text")


class TestTRLSFTStage(unittest.TestCase):
    """Test TRL SFT stage implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TRLSFTConfig(
            stage_name="test_trl_sft",
            output_dir=self.temp_dir,
            dataset_name_or_path="test_dataset.jsonl",
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            validation_split=0.1,
            auto_detect_format=True,
        )
        self.stage = TRLSFTStage(self.config)

    def test_stage_name(self):
        """Test stage name property."""
        self.assertEqual(self.stage.stage_name, "trl_sft")

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        # Should not raise any exceptions
        self.stage.validate_config()

    def test_validate_config_failures(self):
        """Test configuration validation failures."""
        # Test missing dataset path
        config = TRLSFTConfig(
            stage_name="test",
            output_dir=self.temp_dir,
            dataset_name_or_path="",
        )
        stage = TRLSFTStage(config)
        with self.assertRaises(ValueError):
            stage.validate_config()

        # Test invalid max_seq_length
        config = TRLSFTConfig(
            stage_name="test",
            output_dir=self.temp_dir,
            dataset_name_or_path="test.jsonl",
            max_seq_length=0,
        )
        stage = TRLSFTStage(config)
        with self.assertRaises(ValueError):
            stage.validate_config()

        # Test invalid packing strategy
        config = TRLSFTConfig(
            stage_name="test",
            output_dir=self.temp_dir,
            dataset_name_or_path="test.jsonl",
            packing_strategy="invalid",
        )
        stage = TRLSFTStage(config)
        with self.assertRaises(ValueError):
            stage.validate_config()

    def test_detect_chat_format_true(self):
        """Test chat format detection with messages field."""
        chat_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        result = self.stage._detect_chat_format(chat_data)
        self.assertTrue(result)

    def test_detect_chat_format_false(self):
        """Test chat format detection without messages field."""
        regular_data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]

        result = self.stage._detect_chat_format(regular_data)
        self.assertFalse(result)

    def test_detect_chat_format_empty(self):
        """Test chat format detection with empty data."""
        result = self.stage._detect_chat_format([])
        self.assertFalse(result)

    @patch("src.lmpipeline.algorithms.trl_sft.setup_lora")
    def test_prepare_model_and_tokenizer(self, mock_setup_lora):
        """Test model and tokenizer preparation."""
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        # Mock setup_lora to return the model
        mock_setup_lora.return_value = mock_model

        # Call the method
        result_model, result_tokenizer = self.stage.prepare_model_and_tokenizer(
            mock_model, mock_tokenizer
        )

        # Verify setup_lora was called with correct parameters
        mock_setup_lora.assert_called_once_with(
            model=mock_model,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            lora_bias=self.config.lora_bias,
        )

        # Verify pad token was set
        self.assertEqual(mock_tokenizer.pad_token, "<eos>")

        # Verify returned values
        self.assertEqual(result_model, mock_model)
        self.assertEqual(result_tokenizer, mock_tokenizer)

    @patch("src.lmpipeline.utils.dataset_utils.DatasetFormatter")
    def test_prepare_dataset_for_trl_regular_format(self, mock_formatter):
        """Test dataset preparation for regular (non-chat) format."""
        # Mock data
        regular_data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Mock formatter
        mock_formatter.detect_format.return_value = ("instruction", "response")
        mock_formatter.convert_to_standard_format.return_value = {
            "instruction": "What is AI?",
            "response": "AI is artificial intelligence.",
        }

        # Call the method
        result = self.stage._prepare_dataset_for_trl(regular_data, mock_tokenizer)

        # Verify format detection was called
        mock_formatter.detect_format.assert_called_once_with(regular_data)
        mock_formatter.convert_to_standard_format.assert_called_once()

        # Verify result
        self.assertEqual(len(result), 1)

    def test_prepare_dataset_for_trl_chat_format(self):
        """Test dataset preparation for chat format."""
        # Mock chat data
        chat_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        # Mock tokenizer
        mock_tokenizer = Mock()

        # Call the method
        result = self.stage._prepare_dataset_for_trl(chat_data, mock_tokenizer)

        # For chat format, data should be returned as-is
        self.assertEqual(result, chat_data)

    @patch("src.lmpipeline.algorithms.trl_sft.Dataset")
    @patch("trl.SFTTrainer")
    @patch("trl.SFTConfig")
    @patch("peft.LoraConfig")
    def test_create_trl_trainer(
        self, mock_lora_config, mock_sft_config, mock_sft_trainer, mock_dataset
    ):
        """Test TRL trainer creation."""
        # Mock data
        train_data = [{"text": "sample text"}]
        val_data = [{"text": "validation text"}]

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Mock dataset creation
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_dataset.from_list.side_effect = [mock_train_dataset, mock_eval_dataset]

        # Mock config creation
        mock_config_instance = Mock()
        mock_sft_config.return_value = mock_config_instance

        # Mock PEFT config
        mock_peft_config_instance = Mock()
        mock_lora_config.return_value = mock_peft_config_instance

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_sft_trainer.return_value = mock_trainer_instance

        # Call the method
        result = self.stage._create_trl_trainer(
            mock_model, mock_tokenizer, train_data, val_data
        )

        # Verify dataset creation
        self.assertEqual(mock_dataset.from_list.call_count, 2)
        mock_dataset.from_list.assert_any_call(train_data)
        mock_dataset.from_list.assert_any_call(val_data)

        # Verify trainer creation
        mock_sft_trainer.assert_called_once_with(
            model=mock_model,
            args=mock_config_instance,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            processing_class=mock_tokenizer,
            peft_config=mock_peft_config_instance,
        )

        # Verify result
        self.assertEqual(result, mock_trainer_instance)

    @patch("trl.SFTConfig")
    def test_create_trl_sft_config(self, mock_sft_config):
        """Test TRL SFTConfig creation."""
        # Mock config instance
        mock_config_instance = Mock()
        mock_sft_config.return_value = mock_config_instance

        # Call the method
        result = self.stage._create_trl_sft_config()

        # Verify SFTConfig was called
        mock_sft_config.assert_called_once()

        # Get the call arguments
        call_args = mock_sft_config.call_args[1]

        # Verify key parameters are mapped correctly
        self.assertEqual(call_args["output_dir"], self.config.output_dir)
        self.assertEqual(call_args["num_train_epochs"], self.config.num_train_epochs)
        self.assertEqual(call_args["max_length"], self.config.max_seq_length)
        self.assertEqual(call_args["packing"], self.config.packing)

        # Verify result
        self.assertEqual(result, mock_config_instance)

    def test_trl_import_error(self):
        """Test handling of TRL import error."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'trl'")
        ):
            with self.assertRaises(ImportError) as context:
                self.stage._create_trl_trainer(Mock(), Mock(), [], [])

            self.assertIn("TRL library is required", str(context.exception))

    @patch("src.lmpipeline.algorithms.trl_sft.load_dataset_from_path")
    @patch("src.lmpipeline.algorithms.trl_sft.split_dataset")
    @patch.object(TRLSFTStage, "_create_trl_trainer")
    @patch.object(TRLSFTStage, "save_model_and_tokenizer")
    @patch.object(TRLSFTStage, "setup_logging")
    @patch.object(TRLSFTStage, "cleanup_logging")
    def test_execute_success(
        self,
        mock_cleanup,
        mock_setup,
        mock_save,
        mock_create_trainer,
        mock_split,
        mock_load,
    ):
        """Test successful execution of TRL SFT training."""
        # Mock data loading
        sample_data = [{"instruction": "test", "response": "response"}]
        train_data = sample_data[:1]
        val_data = []

        mock_load.return_value = sample_data
        mock_split.return_value = (train_data, val_data)

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.state.log_history = [{"train_loss": 0.5, "eval_loss": 0.4}]
        mock_create_trainer.return_value = mock_trainer

        # Mock save
        mock_save.return_value = ("/path/to/model", "/path/to/tokenizer")

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Execute
        result = self.stage.execute(mock_model, mock_tokenizer)

        # Verify calls
        mock_setup.assert_called_once()
        mock_load.assert_called_once_with(
            dataset_name_or_path=self.config.dataset_name_or_path,
            dataset_config_name=self.config.dataset_config_name,
        )
        mock_split.assert_called_once_with(sample_data, self.config.validation_split)
        mock_create_trainer.assert_called_once()
        mock_trainer.train.assert_called_once_with(
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )
        mock_save.assert_called_once_with(mock_model, mock_tokenizer)
        mock_cleanup.assert_called_once()

        # Verify result
        self.assertIsInstance(result, StageResult)
        self.assertTrue(result.success)
        self.assertEqual(result.stage_name, "trl_sft")
        self.assertEqual(result.model_path, "/path/to/model")
        self.assertEqual(result.tokenizer_path, "/path/to/tokenizer")
        self.assertIn("train_loss", result.metrics)
        self.assertIn("eval_loss", result.metrics)

    @patch("src.lmpipeline.algorithms.trl_sft.load_dataset_from_path")
    @patch.object(TRLSFTStage, "setup_logging")
    @patch.object(TRLSFTStage, "cleanup_logging")
    def test_execute_failure(self, mock_cleanup, mock_setup, mock_load):
        """Test execution failure handling."""
        # Mock data loading to raise an exception
        mock_load.side_effect = Exception("Dataset loading failed")

        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Execute
        result = self.stage.execute(mock_model, mock_tokenizer)

        # Verify calls
        mock_setup.assert_called_once()
        mock_cleanup.assert_called_once()

        # Verify result
        self.assertIsInstance(result, StageResult)
        self.assertFalse(result.success)
        self.assertEqual(result.stage_name, "trl_sft")
        self.assertEqual(result.model_path, "")
        self.assertEqual(result.tokenizer_path, "")
        self.assertIn("Dataset loading failed", result.error_message)


class TestTRLSFTIntegration(unittest.TestCase):
    """Integration tests for TRL SFT stage with pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create sample dataset file
        self.dataset_path = Path(self.temp_dir) / "test_dataset.jsonl"
        sample_data = [
            {
                "instruction": "What is AI?",
                "response": "AI is artificial intelligence.",
            },
            {"instruction": "What is ML?", "response": "ML is machine learning."},
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
        ]

        with open(self.dataset_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

    def test_pipeline_registration(self):
        """Test that TRL SFT stage can be registered with pipeline."""
        from src.lmpipeline.pipeline import Pipeline

        # Check if TRL SFT stage is registered
        self.assertIn("trl_sft", Pipeline.STAGE_REGISTRY)
        self.assertEqual(Pipeline.STAGE_REGISTRY["trl_sft"], TRLSFTStage)

    @patch("src.lmpipeline.pipeline.AutoModelForCausalLM")
    @patch("src.lmpipeline.pipeline.AutoTokenizer")
    def test_pipeline_config_with_trl_sft(self, mock_tokenizer, mock_model):
        """Test creating pipeline configuration with TRL SFT stage."""
        from src.lmpipeline.pipeline import PipelineConfig

        config = PipelineConfig(
            model_name_or_path="microsoft/DialoGPT-small",
            output_dir=self.temp_dir,
            stages=["trl_sft"],
            stage_configs={
                "trl_sft": {
                    "dataset_name_or_path": str(self.dataset_path),
                    "max_seq_length": 512,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 2,
                    "packing": True,
                    "completion_only_loss": True,
                    "neftune_noise_alpha": 5.0,
                }
            },
        )

        # Verify configuration
        self.assertEqual(config.stages, ["trl_sft"])
        self.assertIn("trl_sft", config.stage_configs)
        trl_config = config.stage_configs["trl_sft"]
        self.assertTrue(trl_config["packing"])
        self.assertTrue(trl_config["completion_only_loss"])
        self.assertEqual(trl_config["neftune_noise_alpha"], 5.0)

    def test_feature_parity_with_sft(self):
        """Test that TRL SFT config has feature parity with regular SFT config."""
        from src.lmpipeline.algorithms.sft import SFTConfig

        # Create both configs with same parameters
        base_params = {
            "stage_name": "test",
            "output_dir": self.temp_dir,
            "dataset_name_or_path": str(self.dataset_path),
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        }

        sft_config = SFTConfig(**base_params)
        trl_sft_config = TRLSFTConfig(**base_params)

        # Verify all base parameters are the same
        for param, value in base_params.items():
            self.assertEqual(getattr(sft_config, param), getattr(trl_sft_config, param))

        # Verify TRL SFT has additional parameters
        self.assertTrue(hasattr(trl_sft_config, "packing"))
        self.assertTrue(hasattr(trl_sft_config, "completion_only_loss"))
        self.assertTrue(hasattr(trl_sft_config, "neftune_noise_alpha"))


if __name__ == "__main__":
    unittest.main()
