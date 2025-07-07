#!/usr/bin/env python3
"""
Basic test script for TRL SFT implementation without heavy dependencies.
Tests the core logic and configuration handling.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_config_creation():
    """Test TRL SFT configuration creation."""
    print("Testing TRLSFTConfig creation...")

    # Mock the dependencies that aren't available
    import unittest.mock as mock

    # Mock all the heavy dependencies
    sys.modules["torch"] = mock.MagicMock()
    sys.modules["transformers"] = mock.MagicMock()
    sys.modules["datasets"] = mock.MagicMock()
    sys.modules["peft"] = mock.MagicMock()
    sys.modules["yaml"] = mock.MagicMock()

    # Mock the pipeline module to avoid import issues
    pipeline_mock = mock.MagicMock()
    sys.modules["lmpipeline.pipeline"] = pipeline_mock

    try:
        # Import directly from the module file
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "trl_sft",
            str(
                Path(__file__).parent
                / "src"
                / "lmpipeline"
                / "algorithms"
                / "trl_sft.py"
            ),
        )
        trl_sft_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trl_sft_module)

        TRLSFTConfig = trl_sft_module.TRLSFTConfig

        # Test basic config creation
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

        # Verify basic properties
        assert config.stage_name == "test_trl_sft"
        assert config.output_dir == "./test_output"
        assert config.dataset_name_or_path == "test_dataset.jsonl"
        assert config.max_seq_length == 1024
        assert config.num_train_epochs == 1
        assert config.per_device_train_batch_size == 2

        # Verify TRL-specific properties
        assert config.packing == True
        assert config.packing_strategy == "ffd"
        assert config.completion_only_loss == True
        assert config.assistant_only_loss == False
        assert config.neftune_noise_alpha == 5.0
        assert config.use_liger_kernel == True

        print("✓ TRLSFTConfig creation and property access works")
        return True

    except Exception as e:
        print(f"✗ TRLSFTConfig creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_stage_creation():
    """Test TRL SFT stage creation."""
    print("Testing TRLSFTStage creation...")

    try:
        from lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig

        config = TRLSFTConfig(
            stage_name="test", output_dir="./test", dataset_name_or_path="test.jsonl"
        )

        stage = TRLSFTStage(config)

        # Test stage properties
        assert stage.stage_name == "trl_sft"
        assert stage.config == config

        print("✓ TRLSFTStage creation works")
        return True

    except Exception as e:
        print(f"✗ TRLSFTStage creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chat_format_detection():
    """Test chat format detection logic."""
    print("Testing chat format detection...")

    try:
        from lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig

        config = TRLSFTConfig(
            stage_name="test", output_dir="./test", dataset_name_or_path="test.jsonl"
        )
        stage = TRLSFTStage(config)

        # Test chat format detection
        chat_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]

        regular_data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]

        empty_data = []

        # Test detection
        assert stage._detect_chat_format(chat_data) == True
        assert stage._detect_chat_format(regular_data) == False
        assert stage._detect_chat_format(empty_data) == False

        print("✓ Chat format detection works correctly")
        return True

    except Exception as e:
        print(f"✗ Chat format detection failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    try:
        from lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig

        # Test valid config
        valid_config = TRLSFTConfig(
            stage_name="test",
            output_dir="./test",
            dataset_name_or_path="test.jsonl",
            max_seq_length=512,
            num_train_epochs=1,
            per_device_train_batch_size=2,
        )

        stage = TRLSFTStage(valid_config)
        stage.validate_config()  # Should not raise

        # Test invalid configs
        invalid_configs = [
            # Missing dataset path
            TRLSFTConfig(
                stage_name="test",
                output_dir="./test",
                dataset_name_or_path="",
            ),
            # Invalid max_seq_length
            TRLSFTConfig(
                stage_name="test",
                output_dir="./test",
                dataset_name_or_path="test.jsonl",
                max_seq_length=0,
            ),
            # Invalid packing strategy
            TRLSFTConfig(
                stage_name="test",
                output_dir="./test",
                dataset_name_or_path="test.jsonl",
                packing_strategy="invalid",
            ),
        ]

        for i, config in enumerate(invalid_configs):
            stage = TRLSFTStage(config)
            try:
                stage.validate_config()
                print(f"✗ Invalid config {i} should have failed validation")
                return False
            except ValueError:
                pass  # Expected

        print("✓ Configuration validation works correctly")
        return True

    except Exception as e:
        print(f"✗ Configuration validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running TRL SFT basic tests...")
    print("=" * 50)

    tests = [
        test_config_creation,
        test_stage_creation,
        test_chat_format_detection,
        test_config_validation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! TRL SFT implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
