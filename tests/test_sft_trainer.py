#!/usr/bin/env python3
"""
Unit tests for the SFT trainer module.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import torch
from transformers import AutoTokenizer

# Import the modules to test
from fnsft.sft_trainer import (
    ModelArguments,
    DataArguments,
    QuantizationArguments,
    LoRAArguments,
    InstructionDataset,
    DatasetFormatter,
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


class TestDatasetFormatter(unittest.TestCase):
    """Test the DatasetFormatter class for automatic format detection and conversion."""

    def test_detect_alpaca_format(self):
        """Test detection of Alpaca format (instruction + input + output)."""
        data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "Explain ML", "input": "in simple terms", "output": "ML is machine learning."}
        ]

        detected_format = DatasetFormatter.detect_format(data)
        self.assertEqual(detected_format, ("instruction", "input", "output"))

    def test_detect_prompt_completion_format(self):
        """Test detection of prompt-completion format."""
        data = [
            {"prompt": "What is AI?", "completion": "AI is artificial intelligence."},
            {"prompt": "Explain ML", "completion": "ML is machine learning."}
        ]

        detected_format = DatasetFormatter.detect_format(data)
        self.assertEqual(detected_format, ("prompt", "completion"))

    def test_detect_qa_format(self):
        """Test detection of question-answer format."""
        data = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."},
            {"question": "Explain ML", "answer": "ML is machine learning."}
        ]

        detected_format = DatasetFormatter.detect_format(data)
        self.assertEqual(detected_format, ("question", "answer"))

    def test_detect_conversational_format(self):
        """Test detection of conversational format."""
        data = [
            {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence."}]},
            {"messages": [{"role": "user", "content": "Explain ML"}, {"role": "assistant", "content": "ML is machine learning."}]}
        ]

        detected_format = DatasetFormatter.detect_format(data)
        self.assertEqual(detected_format, ("messages",))

    def test_detect_text_format(self):
        """Test detection of text format."""
        data = [
            {"text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI is artificial intelligence."},
            {"text": "### Instruction:\nExplain ML\n\n### Response:\nML is machine learning."}
        ]

        detected_format = DatasetFormatter.detect_format(data)
        self.assertEqual(detected_format, ("text",))

    def test_convert_alpaca_format(self):
        """Test conversion of Alpaca format."""
        item = {"instruction": "What is AI?", "input": "explain briefly", "output": "AI is artificial intelligence."}
        format_keys = ("instruction", "input", "output")

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertIn("instruction", converted)
        self.assertIn("response", converted)
        self.assertIn("explain briefly", converted["instruction"])
        self.assertEqual(converted["response"], "AI is artificial intelligence.")

    def test_convert_alpaca_format_empty_input(self):
        """Test conversion of Alpaca format with empty input."""
        item = {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."}
        format_keys = ("instruction", "input", "output")

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertEqual(converted["instruction"], "What is AI?")
        self.assertEqual(converted["response"], "AI is artificial intelligence.")

    def test_convert_prompt_completion_format(self):
        """Test conversion of prompt-completion format."""
        item = {"prompt": "What is AI?", "completion": "AI is artificial intelligence."}
        format_keys = ("prompt", "completion")

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertEqual(converted["instruction"], "What is AI?")
        self.assertEqual(converted["response"], "AI is artificial intelligence.")

    def test_convert_qa_format(self):
        """Test conversion of question-answer format."""
        item = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        format_keys = ("question", "answer")

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertEqual(converted["instruction"], "What is AI?")
        self.assertEqual(converted["response"], "AI is artificial intelligence.")

    def test_convert_conversational_format(self):
        """Test conversion of conversational format."""
        item = {"messages": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."}
        ]}
        format_keys = ("messages",)

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertEqual(converted["instruction"], "What is AI?")
        self.assertEqual(converted["response"], "AI is artificial intelligence.")

    def test_convert_conversational_format_multi_turn(self):
        """Test conversion of multi-turn conversational format."""
        item = {"messages": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "Tell me more"},
            {"role": "assistant", "content": "AI involves machine learning and neural networks."}
        ]}
        format_keys = ("messages",)

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertIn("What is AI?", converted["instruction"])
        self.assertIn("Tell me more", converted["instruction"])
        self.assertIn("AI is artificial intelligence.", converted["response"])
        self.assertIn("AI involves machine learning", converted["response"])

    def test_convert_text_format(self):
        """Test conversion of text format (should pass through unchanged)."""
        item = {"text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI is artificial intelligence."}
        format_keys = ("text",)

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        self.assertEqual(converted["text"], item["text"])

    def test_detect_format_empty_dataset(self):
        """Test format detection with empty dataset."""
        with self.assertRaises(ValueError):
            DatasetFormatter.detect_format([])

    def test_detect_format_invalid_item(self):
        """Test format detection with invalid item."""
        with self.assertRaises(ValueError):
            DatasetFormatter.detect_format(["not a dict"])

    def test_infer_and_convert_unknown_format(self):
        """Test inference and conversion of unknown format."""
        item = {"query": "What is AI?", "result": "AI is artificial intelligence."}
        format_keys = ("query", "result")

        converted = DatasetFormatter.convert_to_standard_format(item, format_keys)

        # Should infer query as instruction and result as response
        self.assertEqual(converted["instruction"], "What is AI?")
        self.assertEqual(converted["response"], "AI is artificial intelligence.")


class TestEnhancedInstructionDataset(unittest.TestCase):
    """Test the enhanced InstructionDataset class with auto-detection."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.pad_token = None
        self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.eos_token_id = 50256

        # Mock tokenizer call
        mock_encoding = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        self.tokenizer.return_value = mock_encoding

    def test_auto_detect_alpaca_format(self):
        """Test auto-detection with Alpaca format."""
        data = [
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "Explain ML", "input": "briefly", "output": "ML is machine learning."}
        ]

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        self.assertEqual(dataset.detected_format, ("instruction", "input", "output"))
        self.assertEqual(len(dataset), 2)

    def test_auto_detect_prompt_completion_format(self):
        """Test auto-detection with prompt-completion format."""
        data = [
            {"prompt": "What is AI?", "completion": "AI is artificial intelligence."},
            {"prompt": "Explain ML", "completion": "ML is machine learning."}
        ]

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        self.assertEqual(dataset.detected_format, ("prompt", "completion"))
        self.assertEqual(len(dataset), 2)

    def test_auto_detect_disabled(self):
        """Test with auto-detection disabled."""
        data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."},
            {"instruction": "Explain ML", "response": "ML is machine learning."}
        ]

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=False
        )

        self.assertIsNone(dataset.detected_format)
        self.assertEqual(len(dataset), 2)

    def test_getitem_with_auto_detection(self):
        """Test __getitem__ with auto-detection enabled."""
        data = [
            {"prompt": "What is AI?", "completion": "AI is artificial intelligence."}
        ]

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        item = dataset[0]

        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertTrue(torch.equal(item["input_ids"], item["labels"]))

    def test_getitem_conversion_error_fallback(self):
        """Test fallback when conversion fails."""
        data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]

        # Create dataset with auto-detection but force a conversion error
        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        # Manually set a bad format to trigger fallback
        dataset.detected_format = ("bad", "format")

        # Should still work due to fallback logic
        item = dataset[0]
        self.assertIn("input_ids", item)

    def test_empty_dataset_with_auto_detection(self):
        """Test auto-detection with empty dataset."""
        data = []

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        self.assertIsNone(dataset.detected_format)
        self.assertEqual(len(dataset), 0)

    def test_unsupported_format_fallback(self):
        """Test fallback handling for completely unsupported format."""
        data = [
            {"unknown_field": "some value"}
        ]

        dataset = InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=512,
            auto_detect_format=True
        )

        # Should work due to fallback logic (converts to text format)
        item = dataset[0]
        self.assertIn("input_ids", item)


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
