#!/usr/bin/env python3
"""
Unit tests for the SFT trainer module.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from transformers import AutoTokenizer

# Import the modules to test
from fnsft.sft_trainer import (
    ModelArguments,
    DataArguments,
    QuantizationArguments,
    LoRAArguments,
    InstructionDataset,
    load_quantization_config,
    load_dataset_from_path,
    split_dataset,
    load_config_from_yaml,
    upload_to_hub,
)


class TestDataArguments(unittest.TestCase):
    """Test data argument dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        args = DataArguments(dataset_name_or_path="test_dataset")
        self.assertEqual(args.max_seq_length, 2048)
        self.assertEqual(args.validation_split, 0.1)
        self.assertIn("Instruction", args.instruction_template)


class TestQuantizationConfig(unittest.TestCase):
    """Test quantization configuration."""
    
    def test_4bit_config(self):
        """Test 4-bit quantization configuration."""
        args = QuantizationArguments(use_4bit=True, use_8bit=False)
        config = load_quantization_config(args)
        
        self.assertIsNotNone(config)
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
    
    def test_8bit_config(self):
        """Test 8-bit quantization configuration."""
        args = QuantizationArguments(use_4bit=False, use_8bit=True)
        config = load_quantization_config(args)
        
        self.assertIsNotNone(config)
        self.assertTrue(config.load_in_8bit)
    
    def test_no_quantization(self):
        """Test no quantization."""
        args = QuantizationArguments(use_4bit=False, use_8bit=False)
        config = load_quantization_config(args)
        
        self.assertIsNone(config)


class TestInstructionDataset(unittest.TestCase):
    """Test instruction dataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token = None
        self.tokenizer.eos_token = "<eos>"
        self.tokenizer.eos_token_id = 2
        
        # Mock tokenizer call
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        self.sample_data = [
            {
                "instruction": "What is 2+2?",
                "response": "2+2 equals 4."
            },
            {
                "text": "This is a sample text."
            }
        ]
    
    def test_dataset_creation(self):
        """Test dataset creation with instruction-response format."""
        dataset = InstructionDataset(
            data=self.sample_data[:1],  # Only instruction-response format
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        self.assertEqual(len(dataset), 1)
        self.assertEqual(self.tokenizer.pad_token, "<eos>")
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        dataset = InstructionDataset(
            data=self.sample_data[:1],
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        item = dataset[0]
        
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertTrue(torch.equal(item["input_ids"], item["labels"]))


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading functions."""
    
    def test_load_json_file(self):
        """Test loading from JSON file."""
        sample_data = [
            {"instruction": "Test instruction", "response": "Test response"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        try:
            args = DataArguments(dataset_name_or_path=temp_file)
            data = load_dataset_from_path(args)
            
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["instruction"], "Test instruction")
        finally:
            os.unlink(temp_file)
    
    def test_load_jsonl_file(self):
        """Test loading from JSONL file."""
        sample_data = [
            {"instruction": "Test 1", "response": "Response 1"},
            {"instruction": "Test 2", "response": "Response 2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            args = DataArguments(dataset_name_or_path=temp_file)
            data = load_dataset_from_path(args)
            
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["instruction"], "Test 1")
            self.assertEqual(data[1]["instruction"], "Test 2")
        finally:
            os.unlink(temp_file)
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        data = [{"text": f"Sample {i}"} for i in range(100)]
        
        train_data, val_data = split_dataset(data, validation_split=0.2)
        
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(val_data), 20)
        
        # Test no validation split
        train_data, val_data = split_dataset(data, validation_split=0.0)
        self.assertEqual(len(train_data), 100)
        self.assertEqual(len(val_data), 0)


class TestConfigLoading(unittest.TestCase):
    """Test configuration file loading."""
    
    def test_yaml_config_loading(self):
        """Test loading YAML configuration."""
        config_data = {
            "model_name_or_path": "test_model",
            "learning_rate": 1e-4,
            "num_train_epochs": 5,
            "use_4bit": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_config_from_yaml(temp_file)
            
            self.assertEqual(config["model_name_or_path"], "test_model")
            self.assertEqual(config["learning_rate"], 1e-4)
            self.assertEqual(config["num_train_epochs"], 5)
            self.assertTrue(config["use_4bit"])
        finally:
            os.unlink(temp_file)


class TestLoRAArguments(unittest.TestCase):
    """Test LoRA argument dataclass."""

    def test_default_values(self):
        """Test default LoRA values."""
        args = LoRAArguments()

        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.lora_dropout, 0.1)
        self.assertEqual(args.lora_bias, "none")
        self.assertIsNone(args.lora_target_modules)


class TestHubUpload(unittest.TestCase):
    """Test Hugging Face Hub upload functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.push_to_hub = MagicMock()

        # Create a temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_path, exist_ok=True)

        # Create mock model files
        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        with open(os.path.join(self.model_path, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA"}, f)

        with open(os.path.join(self.model_path, "adapter_model.safetensors"), "w") as f:
            f.write("mock adapter weights")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_invalid_repo_id(self):
        """Test upload with invalid repository ID."""
        with self.assertRaises(ValueError) as context:
            upload_to_hub(
                model_path=self.model_path,
                tokenizer=self.mock_tokenizer,
                repo_id="invalid_repo_id"  # Missing username/org
            )

        self.assertIn("repo_id must be in format", str(context.exception))

    def test_nonexistent_model_path(self):
        """Test upload with non-existent model path."""
        with self.assertRaises(ValueError) as context:
            upload_to_hub(
                model_path="/nonexistent/path",
                tokenizer=self.mock_tokenizer,
                repo_id="user/repo"
            )

        self.assertIn("Model path does not exist", str(context.exception))

    @patch('fnsft.sft_trainer.HfApi')
    @patch('fnsft.sft_trainer.whoami')
    @patch('fnsft.sft_trainer.AutoPeftModelForCausalLM')
    def test_successful_upload_with_peft(self, mock_peft_model, mock_whoami, mock_hf_api):
        """Test successful upload of PEFT model."""
        # Mock HF API
        mock_api = MagicMock()
        mock_hf_api.return_value = mock_api
        mock_api.repo_info.return_value = {"name": "test_repo"}

        # Mock whoami
        mock_whoami.return_value = {"name": "test_user"}

        # Mock PEFT model
        mock_model = MagicMock()
        mock_peft_model.from_pretrained.return_value = mock_model
        mock_model.push_to_hub = MagicMock()

        # Test upload
        upload_to_hub(
            model_path=self.model_path,
            tokenizer=self.mock_tokenizer,
            repo_id="test_user/test_repo",
            commit_message="Test upload",
            private=False
        )

        # Verify tokenizer upload was called
        self.mock_tokenizer.push_to_hub.assert_called_once()

        # Verify model upload was called
        mock_model.push_to_hub.assert_called_once()

    @patch('fnsft.sft_trainer.HfApi')
    @patch('fnsft.sft_trainer.whoami')
    def test_adapter_only_upload(self, mock_whoami, mock_hf_api):
        """Test upload of only LoRA adapter files."""
        # Mock HF API
        mock_api = MagicMock()
        mock_hf_api.return_value = mock_api
        mock_api.repo_info.return_value = {"name": "test_repo"}
        mock_api.upload_file = MagicMock()

        # Mock whoami
        mock_whoami.return_value = {"name": "test_user"}

        # Test adapter-only upload
        upload_to_hub(
            model_path=self.model_path,
            tokenizer=self.mock_tokenizer,
            repo_id="test_user/test_repo",
            push_adapter_only=True
        )

        # Verify tokenizer upload was called
        self.mock_tokenizer.push_to_hub.assert_called_once()

        # Verify adapter file upload was called
        mock_api.upload_file.assert_called()

    @patch('fnsft.sft_trainer.HfApi')
    @patch('fnsft.sft_trainer.whoami')
    def test_repository_creation(self, mock_whoami, mock_hf_api):
        """Test automatic repository creation."""
        # Mock HF API
        mock_api = MagicMock()
        mock_hf_api.return_value = mock_api

        # Mock repository not found, then successful creation
        from fnsft.sft_trainer import RepositoryNotFoundError
        mock_api.repo_info.side_effect = RepositoryNotFoundError("Not found")
        mock_api.create_repo = MagicMock()
        mock_api.upload_file = MagicMock()

        # Mock whoami
        mock_whoami.return_value = {"name": "test_user"}

        # Test upload with repository creation
        upload_to_hub(
            model_path=self.model_path,
            tokenizer=self.mock_tokenizer,
            repo_id="test_user/new_repo",
            private=True,
            push_adapter_only=True
        )

        # Verify repository creation was called
        mock_api.create_repo.assert_called_once_with(
            repo_id="test_user/new_repo",
            repo_type="model",
            private=True,
            exist_ok=True
        )


if __name__ == "__main__":
    unittest.main()
