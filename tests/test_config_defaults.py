"""
Unit tests for the configuration defaults system.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from lmpipeline.utils.config_defaults import ConfigDefaults


class TestConfigDefaults:
    """Test cases for ConfigDefaults class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_default_directory_paths(self):
        """Test default directory path getters."""
        assert ConfigDefaults.get_default_models_dir() == "./models"
        assert ConfigDefaults.get_default_checkpoints_dir() == "./models/checkpoints"
        assert ConfigDefaults.get_default_output_dir() == "./models/output"
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides for default paths."""
        os.environ[ConfigDefaults.ENV_MODELS_DIR] = "/custom/models"
        os.environ[ConfigDefaults.ENV_CHECKPOINTS_DIR] = "/custom/checkpoints"
        os.environ[ConfigDefaults.ENV_OUTPUT_DIR] = "/custom/output"
        
        assert ConfigDefaults.get_default_models_dir() == "/custom/models"
        assert ConfigDefaults.get_default_checkpoints_dir() == "/custom/checkpoints"
        assert ConfigDefaults.get_default_output_dir() == "/custom/output"
    
    def test_ensure_directory_exists_success(self):
        """Test successful directory creation."""
        test_dir = os.path.join(self.temp_dir, "test_directory")
        assert ConfigDefaults.ensure_directory_exists(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
    
    def test_ensure_directory_exists_already_exists(self):
        """Test directory creation when directory already exists."""
        test_dir = os.path.join(self.temp_dir, "existing_directory")
        os.makedirs(test_dir)
        
        assert ConfigDefaults.ensure_directory_exists(test_dir)
        assert os.path.exists(test_dir)
    
    def test_ensure_directory_exists_file_conflict(self):
        """Test directory creation when a file exists with the same name."""
        test_path = os.path.join(self.temp_dir, "conflict_path")
        # Create a file with the same name
        with open(test_path, 'w') as f:
            f.write("test")
        
        assert not ConfigDefaults.ensure_directory_exists(test_path)
    
    @patch('lmpipeline.utils.config_defaults.Path.mkdir')
    def test_ensure_directory_exists_permission_error(self, mock_mkdir):
        """Test directory creation with permission error."""
        mock_mkdir.side_effect = PermissionError("Permission denied")
        test_dir = os.path.join(self.temp_dir, "permission_test")
        
        assert not ConfigDefaults.ensure_directory_exists(test_dir)
    
    def test_extract_model_base_name_huggingface(self):
        """Test model base name extraction from HuggingFace model names."""
        test_cases = [
            ("microsoft/DialoGPT-medium", "dialogpt-medium"),
            ("meta-llama/Llama-2-7b-hf", "llama-2-7b"),
            ("mistralai/Mistral-7B-Instruct-v0.1", "mistral-7b-instruct-v0.1"),
        ]
        
        for input_name, expected in test_cases:
            result = ConfigDefaults.extract_model_base_name(input_name)
            assert result == expected
    
    def test_extract_model_base_name_local_path(self):
        """Test model base name extraction from local paths."""
        test_cases = [
            ("/path/to/my-model", "my-model"),
            ("./local-model-chat", "local-model"),
            ("model-base", "model"),
        ]
        
        for input_path, expected in test_cases:
            result = ConfigDefaults.extract_model_base_name(input_path)
            assert result == expected
    
    def test_generate_model_name_basic(self):
        """Test basic model name generation."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium",
            stages=["sft"],
            quantization_config=None,
            torch_dtype="auto",
            convert_to_gguf=False
        )
        assert result == "dialogpt-medium-finetuned"
    
    def test_generate_model_name_with_quantization(self):
        """Test model name generation with quantization."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="meta-llama/Llama-2-7b-hf",
            stages=["sft", "dpo"],
            quantization_config={"use_4bit": True, "use_8bit": False},
            torch_dtype="float16",
            convert_to_gguf=False
        )
        assert result == "llama-2-7b-finetuned-4bit-fp16"
    
    def test_generate_model_name_with_gguf(self):
        """Test model name generation with GGUF conversion."""
        result = ConfigDefaults.generate_model_name(
            base_model_name="microsoft/DialoGPT-medium",
            stages=["sft"],
            quantization_config={"use_4bit": True, "use_8bit": False},
            torch_dtype="float16",
            convert_to_gguf=True,
            gguf_quantization="q4_0"
        )
        assert result == "dialogpt-medium-finetuned-4bit-fp16-gguf-q4_0"
    
    def test_apply_storage_defaults_no_output_dir(self):
        """Test applying storage defaults when no output directory is specified."""
        config_dict = {
            "model_name_or_path": "test-model",
            "stages": ["sft"]
        }
        
        with patch.object(ConfigDefaults, 'ensure_directory_exists', return_value=True):
            result = ConfigDefaults.apply_storage_defaults_with_fallback(config_dict)
        
        assert "output_dir" in result
        assert result["output_dir"] == ConfigDefaults.get_default_output_dir()
    
    def test_apply_storage_defaults_with_existing_output_dir(self):
        """Test applying storage defaults when output directory is already specified."""
        config_dict = {
            "model_name_or_path": "test-model",
            "output_dir": "/custom/output",
            "stages": ["sft"]
        }
        
        result = ConfigDefaults.apply_storage_defaults_with_fallback(config_dict)
        assert result["output_dir"] == "/custom/output"
    
    def test_apply_model_naming_defaults_gguf(self):
        """Test applying model naming defaults for GGUF conversion."""
        config_dict = {
            "model_name_or_path": "test-model",
            "output_dir": "/test/output",
            "stages": ["sft"],
            "convert_to_gguf": True,
            "gguf_quantization": "q4_0"
        }
        
        result = ConfigDefaults.apply_model_naming_defaults(config_dict)
        
        assert "gguf_output_path" in result
        assert result["gguf_output_path"].endswith(".gguf")
        assert "test-model-finetuned-gguf-q4_0.gguf" in result["gguf_output_path"]
    
    def test_get_fallback_directory(self):
        """Test fallback directory selection."""
        preferred_dir = "/nonexistent/directory"
        
        with patch.object(ConfigDefaults, 'ensure_directory_exists') as mock_ensure:
            # First call (for ./outputs) returns True
            mock_ensure.side_effect = [True]
            
            result = ConfigDefaults.get_fallback_directory(preferred_dir)
            assert result == "./outputs"
    
    def test_get_fallback_directory_all_fail(self):
        """Test fallback directory when all options fail."""
        preferred_dir = "/nonexistent/directory"
        
        with patch.object(ConfigDefaults, 'ensure_directory_exists', return_value=False):
            result = ConfigDefaults.get_fallback_directory(preferred_dir)
            assert result == "."
    
    def test_validate_and_create_directories(self):
        """Test directory validation and creation."""
        config_dict = {
            "output_dir": os.path.join(self.temp_dir, "output"),
            "stage_configs": {
                "sft": {
                    "checkpoint_dir": os.path.join(self.temp_dir, "checkpoints", "sft")
                }
            }
        }
        
        failed_dirs = ConfigDefaults.validate_and_create_directories(config_dict)
        assert len(failed_dirs) == 0
        assert os.path.exists(config_dict["output_dir"])
        assert os.path.exists(config_dict["stage_configs"]["sft"]["checkpoint_dir"])
    
    def test_apply_all_defaults_integration(self):
        """Test the complete default application process."""
        config_dict = {
            "model_name_or_path": "microsoft/DialoGPT-medium",
            "stages": ["sft"],
            "convert_to_gguf": True,
            "gguf_quantization": "q4_0",
            "stage_configs": {
                "sft": {
                    "use_4bit": True
                }
            }
        }
        
        with patch.object(ConfigDefaults, 'ensure_directory_exists', return_value=True):
            result = ConfigDefaults.apply_all_defaults(config_dict)
        
        # Check that defaults were applied
        assert "output_dir" in result
        assert "gguf_output_path" in result
        assert result["gguf_output_path"].endswith(".gguf")
        
        # Check stage configs were updated
        sft_config = result["stage_configs"]["sft"]
        assert "checkpoint_dir" in sft_config
        assert "save_steps" in sft_config
        assert "save_total_limit" in sft_config
