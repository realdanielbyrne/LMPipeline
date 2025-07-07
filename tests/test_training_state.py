"""
Unit tests for the training state persistence system.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.lmpipeline.utils.training_state import (
    StageProgress,
    TrainingState,
    TrainingStateManager,
)


class TestStageProgress(unittest.TestCase):
    """Test StageProgress class."""

    def test_stage_progress_creation(self):
        """Test creating a StageProgress instance."""
        progress = StageProgress(
            stage_name="sft",
            status="in_progress",
            current_epoch=2,
            total_epochs=5,
            current_step=100,
            total_steps=500,
        )
        
        self.assertEqual(progress.stage_name, "sft")
        self.assertEqual(progress.status, "in_progress")
        self.assertEqual(progress.current_epoch, 2)
        self.assertEqual(progress.total_epochs, 5)
        self.assertEqual(progress.current_step, 100)
        self.assertEqual(progress.total_steps, 500)

    def test_stage_progress_serialization(self):
        """Test StageProgress to_dict and from_dict methods."""
        progress = StageProgress(
            stage_name="dpo",
            status="completed",
            start_time=time.time(),
            end_time=time.time() + 3600,
            metrics={"loss": 0.5, "accuracy": 0.85},
        )
        
        # Test serialization
        progress_dict = progress.to_dict()
        self.assertIsInstance(progress_dict, dict)
        self.assertEqual(progress_dict["stage_name"], "dpo")
        self.assertEqual(progress_dict["status"], "completed")
        
        # Test deserialization
        restored_progress = StageProgress.from_dict(progress_dict)
        self.assertEqual(restored_progress.stage_name, progress.stage_name)
        self.assertEqual(restored_progress.status, progress.status)
        self.assertEqual(restored_progress.metrics, progress.metrics)


class TestTrainingState(unittest.TestCase):
    """Test TrainingState class."""

    def test_training_state_creation(self):
        """Test creating a TrainingState instance."""
        state = TrainingState(
            pipeline_id="test_pipeline_123",
            config_hash="abc123",
            current_stage="sft",
            output_dir="/tmp/test",
            model_name_or_path="test/model",
        )
        
        self.assertEqual(state.pipeline_id, "test_pipeline_123")
        self.assertEqual(state.config_hash, "abc123")
        self.assertEqual(state.current_stage, "sft")
        self.assertEqual(state.output_dir, "/tmp/test")
        self.assertEqual(state.model_name_or_path, "test/model")

    def test_training_state_serialization(self):
        """Test TrainingState serialization with stages."""
        state = TrainingState(
            pipeline_id="test_pipeline",
            config_hash="hash123",
            current_stage="dpo",
        )
        
        # Add stage progress
        state.stages["sft"] = StageProgress(
            stage_name="sft",
            status="completed",
            metrics={"loss": 0.3},
        )
        state.stages["dpo"] = StageProgress(
            stage_name="dpo",
            status="in_progress",
            current_epoch=1,
        )
        
        # Test serialization
        state_dict = state.to_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertIn("stages", state_dict)
        self.assertIn("sft", state_dict["stages"])
        self.assertIn("dpo", state_dict["stages"])
        
        # Test deserialization
        restored_state = TrainingState.from_dict(state_dict)
        self.assertEqual(restored_state.pipeline_id, state.pipeline_id)
        self.assertEqual(len(restored_state.stages), 2)
        self.assertIsInstance(restored_state.stages["sft"], StageProgress)
        self.assertEqual(restored_state.stages["sft"].status, "completed")


class TestTrainingStateManager(unittest.TestCase):
    """Test TrainingStateManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "model_name_or_path": "test/model",
            "stages": ["sft", "dpo"],
            "output_dir": self.temp_dir,
            "stage_configs": {
                "sft": {"dataset_name_or_path": "test_dataset.json"},
                "dpo": {"dataset_name_or_path": "test_dataset.json"},
            },
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_state_manager_initialization(self):
        """Test TrainingStateManager initialization."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        self.assertIsNotNone(manager.state)
        self.assertEqual(len(manager.state.stages), 2)
        self.assertIn("sft", manager.state.stages)
        self.assertIn("dpo", manager.state.stages)

    def test_config_hash_generation(self):
        """Test configuration hash generation."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        # Same config should produce same hash
        hash1 = manager._generate_config_hash(self.config)
        hash2 = manager._generate_config_hash(self.config.copy())
        self.assertEqual(hash1, hash2)
        
        # Different config should produce different hash
        modified_config = self.config.copy()
        modified_config["stages"] = ["sft"]
        hash3 = manager._generate_config_hash(modified_config)
        self.assertNotEqual(hash1, hash3)

    def test_state_persistence(self):
        """Test state saving and loading."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        # Start a stage
        manager.start_stage("sft", total_epochs=3)
        
        # Update progress
        manager.update_stage_progress(
            "sft", current_epoch=1, current_step=50, metrics={"loss": 0.5}
        )
        
        # Complete the stage
        manager.complete_stage(
            "sft", "/tmp/model", "/tmp/tokenizer", {"final_loss": 0.3}
        )
        
        # Check state file exists
        self.assertTrue(manager.state_file.exists())
        
        # Create new manager and verify state is loaded
        manager2 = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        self.assertIsNotNone(manager2.state)
        self.assertEqual(manager2.state.stages["sft"].status, "completed")
        self.assertEqual(manager2.state.stages["sft"].current_epoch, 1)
        self.assertEqual(manager2.state.stages["sft"].current_step, 50)

    def test_resume_point_detection(self):
        """Test resume point detection."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        # Initially should resume from first stage
        resume_point = manager.get_resume_point()
        self.assertEqual(resume_point, "sft")
        
        # Complete first stage
        manager.complete_stage("sft", "/tmp/model", "/tmp/tokenizer")
        
        # Should now resume from second stage
        resume_point = manager.get_resume_point()
        self.assertEqual(resume_point, "dpo")
        
        # Complete all stages
        manager.complete_stage("dpo", "/tmp/model2", "/tmp/tokenizer2")
        
        # Should have no resume point
        resume_point = manager.get_resume_point()
        self.assertIsNone(resume_point)

    def test_file_path_validation(self):
        """Test file path validation."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        # Create temporary files
        temp_model = Path(self.temp_dir) / "model"
        temp_tokenizer = Path(self.temp_dir) / "tokenizer"
        temp_model.mkdir()
        temp_tokenizer.mkdir()
        
        # Complete stage with existing paths
        manager.complete_stage("sft", str(temp_model), str(temp_tokenizer))
        
        # Validation should pass
        self.assertTrue(manager.validate_file_paths())
        
        # Remove files
        temp_model.rmdir()
        
        # Validation should fail
        self.assertFalse(manager.validate_file_paths())

    def test_stage_failure_handling(self):
        """Test stage failure handling."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=True,
        )
        
        # Start and fail a stage
        manager.start_stage("sft")
        manager.fail_stage("sft", "Test error message")
        
        # Check stage status
        self.assertEqual(manager.state.stages["sft"].status, "failed")
        self.assertEqual(manager.state.stages["sft"].error_message, "Test error message")
        
        # Resume point should be the failed stage
        resume_point = manager.get_resume_point()
        self.assertEqual(resume_point, "sft")

    def test_disabled_persistence(self):
        """Test behavior when persistence is disabled."""
        manager = TrainingStateManager(
            output_dir=self.temp_dir,
            pipeline_config=self.config,
            enable_persistence=False,
        )
        
        self.assertIsNone(manager.state)
        
        # Operations should not crash
        manager.start_stage("sft")
        manager.update_stage_progress("sft", current_epoch=1)
        manager.complete_stage("sft", "/tmp/model", "/tmp/tokenizer")
        
        # No state file should be created
        self.assertFalse(manager.state_file.exists())


if __name__ == "__main__":
    unittest.main()
