"""
Integration tests for the complete training state persistence system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.lmpipeline.pipeline import Pipeline, PipelineConfig
from src.lmpipeline.utils.training_state import TrainingStateManager


class TestStatePersistenceIntegration(unittest.TestCase):
    """Test the complete state persistence system integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dict = {
            "model_name_or_path": "microsoft/DialoGPT-small",
            "output_dir": self.temp_dir,
            "stages": ["sft"],
            "enable_state_persistence": True,
            "auto_resume": True,
            "force_restart": False,
            "stage_configs": {
                "sft": {
                    "dataset_name_or_path": "test_dataset.json",
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "max_seq_length": 128,
                    "use_wandb": False,
                }
            },
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_config_with_state_persistence(self):
        """Test that PipelineConfig properly handles state persistence options."""
        config = PipelineConfig(**self.config_dict)

        self.assertTrue(config.enable_state_persistence)
        self.assertTrue(config.auto_resume)
        self.assertFalse(config.force_restart)

    @patch("src.lmpipeline.pipeline.AutoModelForCausalLM")
    @patch("src.lmpipeline.pipeline.AutoTokenizer")
    def test_pipeline_initialization_with_state_manager(
        self, mock_tokenizer, mock_model
    ):
        """Test that Pipeline properly initializes state manager."""
        # Mock the model and tokenizer loading
        mock_model.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()

        config = PipelineConfig(**self.config_dict)

        # This should not raise an exception
        pipeline = Pipeline(config)

        # Verify state manager is initialized
        self.assertIsNotNone(pipeline.state_manager)
        self.assertIsInstance(pipeline.state_manager, TrainingStateManager)

        # Verify state file path
        expected_state_file = Path(self.temp_dir) / "training_status.json"
        self.assertEqual(pipeline.state_manager.state_file, expected_state_file)

    def test_state_manager_integration_with_pipeline_config(self):
        """Test that state manager properly processes pipeline configuration."""
        config = PipelineConfig(**self.config_dict)

        state_manager = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,
            enable_persistence=True,
        )

        # Verify state is created
        self.assertIsNotNone(state_manager.state)
        self.assertEqual(state_manager.state.output_dir, self.temp_dir)
        self.assertEqual(len(state_manager.state.stages), 1)
        self.assertIn("sft", state_manager.state.stages)

    def test_config_hash_consistency(self):
        """Test that configuration hash is consistent for same configs."""
        config1 = PipelineConfig(**self.config_dict)
        config2 = PipelineConfig(**self.config_dict.copy())

        manager1 = TrainingStateManager(
            output_dir=self.temp_dir + "_1",
            pipeline_config=config1.__dict__,
            enable_persistence=True,
        )

        manager2 = TrainingStateManager(
            output_dir=self.temp_dir + "_2",
            pipeline_config=config2.__dict__,
            enable_persistence=True,
        )

        self.assertEqual(manager1.config_hash, manager2.config_hash)

    def test_config_hash_changes_with_different_configs(self):
        """Test that configuration hash changes when config changes."""
        config1 = PipelineConfig(**self.config_dict)

        config2_dict = self.config_dict.copy()
        config2_dict["stages"] = ["sft", "dpo"]
        config2 = PipelineConfig(**config2_dict)

        manager1 = TrainingStateManager(
            output_dir=self.temp_dir + "_1",
            pipeline_config=config1.__dict__,
            enable_persistence=True,
        )

        manager2 = TrainingStateManager(
            output_dir=self.temp_dir + "_2",
            pipeline_config=config2.__dict__,
            enable_persistence=True,
        )

        self.assertNotEqual(manager1.config_hash, manager2.config_hash)

    def test_state_file_creation_and_loading(self):
        """Test that state file is created and can be loaded."""
        config = PipelineConfig(**self.config_dict)

        # Create first manager
        manager1 = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,
            enable_persistence=True,
        )

        # State file is created on first save operation
        manager1.save_state()

        # Verify state file exists
        self.assertTrue(manager1.state_file.exists())

        # Modify state
        manager1.start_stage("sft", total_epochs=3)
        manager1.update_stage_progress("sft", current_epoch=1, current_step=100)

        # Create second manager (should load existing state)
        manager2 = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,
            enable_persistence=True,
        )

        # Verify state was loaded correctly
        self.assertEqual(manager2.state.stages["sft"].status, "in_progress")
        self.assertEqual(manager2.state.stages["sft"].current_epoch, 1)
        self.assertEqual(manager2.state.stages["sft"].current_step, 100)

    def test_force_restart_functionality(self):
        """Test that force_restart ignores existing state."""
        config_dict = self.config_dict.copy()
        config_dict["force_restart"] = True
        config = PipelineConfig(**config_dict)

        # Create initial state
        manager1 = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=self.config_dict,  # Use original config
            enable_persistence=True,
        )
        manager1.start_stage("sft")
        manager1.complete_stage("sft", "/tmp/model", "/tmp/tokenizer")

        # Verify state file exists and has completed stage
        self.assertTrue(manager1.state_file.exists())
        self.assertEqual(manager1.state.stages["sft"].status, "completed")

        # Create new manager with force_restart
        manager2 = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,  # Use force_restart config
            enable_persistence=True,
        )

        # Should have fresh state despite existing file
        self.assertEqual(manager2.state.stages["sft"].status, "not_started")

    def test_disabled_persistence(self):
        """Test that persistence can be disabled."""
        config_dict = self.config_dict.copy()
        config_dict["enable_state_persistence"] = False
        config = PipelineConfig(**config_dict)

        state_manager = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,
            enable_persistence=False,
        )

        # State should be None when disabled
        self.assertIsNone(state_manager.state)

        # Operations should not crash
        state_manager.start_stage("sft")
        state_manager.update_stage_progress("sft", current_epoch=1)
        state_manager.complete_stage("sft", "/tmp/model", "/tmp/tokenizer")

        # No state file should be created
        self.assertFalse(state_manager.state_file.exists())

    def test_wandb_logger_creation(self):
        """Test W&B logger creation from pipeline config."""
        from src.lmpipeline.utils.wandb_integration import create_wandb_logger

        # Test with W&B disabled
        config_dict = {"use_wandb": False}

        logger = create_wandb_logger(config_dict, "test_pipeline")
        self.assertIsNone(logger)

        # Test with W&B enabled
        config_dict = {
            "use_wandb": True,
            "wandb_project": "test_project",
            "wandb_run_name": "test_run",
        }

        logger = create_wandb_logger(config_dict, "test_pipeline")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.project_name, "test_project")
        self.assertEqual(logger.run_name, "test_run")

    def test_state_summary_functionality(self):
        """Test state summary generation."""
        config = PipelineConfig(**self.config_dict)

        manager = TrainingStateManager(
            output_dir=config.output_dir,
            pipeline_config=config.__dict__,
            enable_persistence=True,
        )

        # Get initial summary
        summary = manager.get_state_summary()

        self.assertIn("pipeline_id", summary)
        self.assertIn("current_stage", summary)
        self.assertIn("progress", summary)
        self.assertIn("stages", summary)
        self.assertEqual(summary["progress"], "0/1 stages completed")

        # Complete a stage and check summary
        manager.start_stage("sft")
        manager.complete_stage("sft", "/tmp/model", "/tmp/tokenizer")

        summary = manager.get_state_summary()
        self.assertEqual(summary["progress"], "1/1 stages completed")
        self.assertEqual(summary["stages"]["sft"], "completed")


if __name__ == "__main__":
    unittest.main()
