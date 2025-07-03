#!/usr/bin/env python3
"""
Unit tests for the pipeline system.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.lmpipeline.pipeline import Pipeline, PipelineConfig
from src.lmpipeline.algorithms.base import BaseStage, StageConfig, StageResult


class MockStage(BaseStage):
    """Mock stage for testing."""

    def __init__(self, config: StageConfig):
        super().__init__(config)
        self.executed = False

    @property
    def stage_name(self) -> str:
        return "mock"

    def validate_config(self) -> None:
        pass

    def execute(self, model, tokenizer, previous_result=None) -> StageResult:
        self.executed = True
        model_path, tokenizer_path = self.save_model_and_tokenizer(model, tokenizer)
        return StageResult(
            stage_name=self.stage_name,
            success=True,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            metrics={"test_metric": 1.0},
        )


class TestPipelineConfig(unittest.TestCase):
    """Test pipeline configuration."""

    def test_pipeline_config_creation(self):
        """Test creating pipeline configuration."""
        config = PipelineConfig(
            model_name_or_path="test/model",
            output_dir="/tmp/test",
            stages=["sft", "dpo"],
            stage_configs={
                "sft": {"dataset_name_or_path": "test.json"},
                "dpo": {"preference_dataset_path": "prefs.json"},
            },
        )

        self.assertEqual(config.model_name_or_path, "test/model")
        self.assertEqual(config.output_dir, "/tmp/test")
        self.assertEqual(config.stages, ["sft", "dpo"])
        self.assertIn("sft", config.stage_configs)
        self.assertIn("dpo", config.stage_configs)

    def test_pipeline_config_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
model_name_or_path: "test/model"
output_dir: "/tmp/test"
stages:
  - "sft"
stage_configs:
  sft:
    dataset_name_or_path: "test.json"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = PipelineConfig.from_yaml(f.name)
                self.assertEqual(config.model_name_or_path, "test/model")
                self.assertEqual(config.stages, ["sft"])
            finally:
                os.unlink(f.name)


class TestPipeline(unittest.TestCase):
    """Test pipeline orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PipelineConfig(
            model_name_or_path="test/model",
            output_dir=self.temp_dir,
            stages=["mock"],
            stage_configs={"mock": {}},
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stage_registration(self):
        """Test stage registration."""
        Pipeline.register_stage("mock", MockStage)
        self.assertIn("mock", Pipeline.STAGE_REGISTRY)
        self.assertEqual(Pipeline.STAGE_REGISTRY["mock"], MockStage)

    @patch("fnsft.pipeline.AutoModelForCausalLM")
    @patch("fnsft.pipeline.AutoTokenizer")
    def test_pipeline_initialization(self, mock_tokenizer, mock_model):
        """Test pipeline initialization."""
        # Register mock stage
        Pipeline.register_stage("mock", MockStage)

        pipeline = Pipeline(self.config)

        self.assertEqual(len(pipeline.stages), 1)
        self.assertIsInstance(pipeline.stages[0], MockStage)
        self.assertEqual(pipeline.stages[0].stage_name, "mock")

    @patch("fnsft.pipeline.AutoModelForCausalLM")
    @patch("fnsft.pipeline.AutoTokenizer")
    def test_pipeline_execution(self, mock_tokenizer_class, mock_model_class):
        """Test pipeline execution."""
        # Register mock stage
        Pipeline.register_stage("mock", MockStage)

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        pipeline = Pipeline(self.config)

        # Execute pipeline
        results = pipeline.execute()

        # Check results
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].stage_name, "mock")
        self.assertIn("test_metric", results[0].metrics)

        # Check that stage was executed
        self.assertTrue(pipeline.stages[0].executed)

    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        # Register mock stage
        Pipeline.register_stage("mock", MockStage)

        pipeline = Pipeline(self.config)

        # Create temporary directories for the test
        model_path = os.path.join(self.temp_dir, "model")
        tokenizer_path = os.path.join(self.temp_dir, "tokenizer")
        Path(model_path).mkdir()
        Path(tokenizer_path).mkdir()

        # Add mock results
        pipeline.stage_results = [
            StageResult(
                stage_name="mock",
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics={"loss": 0.5},
            )
        ]

        summary = pipeline.get_summary()

        self.assertEqual(summary["total_stages"], 1)
        self.assertEqual(summary["executed_stages"], 1)
        self.assertEqual(summary["successful_stages"], 1)
        self.assertEqual(summary["failed_stages"], 0)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(len(summary["stage_results"]), 1)


class TestStageResult(unittest.TestCase):
    """Test stage result validation."""

    def test_stage_result_validation(self):
        """Test stage result validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model")
            tokenizer_path = os.path.join(temp_dir, "tokenizer")

            # Create dummy files
            Path(model_path).mkdir()
            Path(tokenizer_path).mkdir()

            result = StageResult(
                stage_name="test",
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )

            self.assertEqual(result.stage_name, "test")
            self.assertTrue(result.success)

    def test_stage_result_validation_failure(self):
        """Test stage result validation with missing paths."""
        with self.assertRaises(ValueError):
            StageResult(
                stage_name="test",
                success=True,
                model_path="/nonexistent/path",
                tokenizer_path="/nonexistent/path",
            )


if __name__ == "__main__":
    unittest.main()
