"""
Unit tests for the W&B integration utilities.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from src.lmpipeline.utils.wandb_integration import WandBLogger, create_wandb_logger


class TestWandBLogger(unittest.TestCase):
    """Test WandBLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_name = "test_project"
        self.run_name = "test_run"
        self.tags = ["test", "pipeline"]
        self.notes = "Test run notes"

    def test_logger_without_wandb(self):
        """Test logger behavior when wandb is not installed."""
        # Mock the import to fail
        with patch.dict("sys.modules", {"wandb": None}):
            logger = WandBLogger(
                project_name=self.project_name,
                run_name=self.run_name,
                tags=self.tags,
                notes=self.notes,
            )

            self.assertIsNone(logger.wandb)

            # Operations should not crash
            config = {"learning_rate": 0.001, "batch_size": 32}
            result = logger.init_run(config, "sft")
            self.assertFalse(result)

            logger.log_metrics({"loss": 0.5})
            logger.log_hyperparameters({"lr": 0.001})
            logger.finish_run()

    @patch("src.lmpipeline.utils.wandb_integration.wandb")
    def test_logger_with_wandb(self, mock_wandb):
        """Test logger behavior when wandb is available."""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(
            project_name=self.project_name,
            run_name=self.run_name,
            tags=self.tags,
            notes=self.notes,
        )

        self.assertIsNotNone(logger.wandb)

        # Test run initialization
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_name_or_path": "/local/path/to/model",
            "output_dir": "/local/output",
        }

        result = logger.init_run(config, "sft")
        self.assertTrue(result)

        # Verify wandb.init was called with correct parameters
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args
        self.assertEqual(call_args[1]["project"], self.project_name)
        self.assertEqual(call_args[1]["name"], f"{self.run_name}-sft")
        self.assertIn("stage:sft", call_args[1]["tags"])

        # Verify sensitive paths are excluded from config
        logged_config = call_args[1]["config"]
        self.assertNotIn("model_name_or_path", logged_config)
        self.assertNotIn("output_dir", logged_config)
        self.assertIn("learning_rate", logged_config)
        self.assertIn("batch_size", logged_config)

    @patch("src.lmpipeline.utils.wandb_integration.wandb")
    def test_metrics_logging(self, mock_wandb):
        """Test metrics logging functionality."""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(self.project_name)
        logger.init_run({}, "test")

        # Test logging valid metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.85,
            "learning_rate": 0.001,
            "invalid_metric": "string_value",  # Should be filtered out
            "tensor_metric": Mock(item=lambda: 0.75),  # Should be converted
        }

        logger.log_metrics(metrics, step=100)

        # Verify wandb.log was called
        mock_wandb.log.assert_called_once()
        logged_metrics = mock_wandb.log.call_args[0][0]

        # Check that only valid metrics are logged
        self.assertIn("loss", logged_metrics)
        self.assertIn("accuracy", logged_metrics)
        self.assertIn("learning_rate", logged_metrics)
        self.assertIn("tensor_metric", logged_metrics)
        self.assertNotIn("invalid_metric", logged_metrics)

        # Check tensor conversion
        self.assertEqual(logged_metrics["tensor_metric"], 0.75)

    @patch("src.lmpipeline.utils.wandb_integration.wandb")
    def test_hyperparameter_logging(self, mock_wandb):
        """Test hyperparameter logging functionality."""
        mock_run = Mock()
        mock_config = Mock()
        mock_run.config = mock_config
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(self.project_name)
        logger.init_run({}, "test")

        # Test logging hyperparameters
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
            "model_path": "/local/path",  # Should be filtered
        }

        logger.log_hyperparameters(hyperparams)

        # Verify config.update was called
        mock_config.update.assert_called_once()
        logged_params = mock_config.update.call_args[0][0]

        # Check that sensitive paths are excluded
        self.assertIn("learning_rate", logged_params)
        self.assertIn("batch_size", logged_params)
        self.assertIn("epochs", logged_params)
        self.assertNotIn("model_path", logged_params)

    @patch("src.lmpipeline.utils.wandb_integration.wandb")
    def test_model_info_logging(self, mock_wandb):
        """Test model information logging."""
        mock_run = Mock()
        mock_config = Mock()
        mock_run.config = mock_config
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(self.project_name)
        logger.init_run({}, "test")

        # Test logging model info
        model_info = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 50257,
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "local_path": "/some/local/path",  # Should be ignored
        }

        logger.log_model_info(model_info)

        # Verify config.update was called
        mock_config.update.assert_called_once()
        logged_info = mock_config.update.call_args[0][0]

        # Check that model architecture info is logged with proper prefixes
        self.assertIn("model.hidden_size", logged_info)
        self.assertIn("model.num_hidden_layers", logged_info)
        self.assertIn("model.type", logged_info)
        self.assertIn("model.architectures", logged_info)
        self.assertNotIn("local_path", logged_info)

    @patch("src.lmpipeline.utils.wandb_integration.wandb")
    def test_stage_completion_logging(self, mock_wandb):
        """Test stage completion logging."""
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run

        logger = WandBLogger(self.project_name)
        logger.init_run({}, "test")

        # Test logging stage completion
        stage_name = "sft"
        duration = 3600.5  # seconds
        final_metrics = {
            "final_loss": 0.25,
            "final_accuracy": 0.92,
            "invalid_metric": "string",  # Should be filtered
        }

        logger.log_stage_completion(stage_name, duration, final_metrics)

        # Verify wandb.log was called
        mock_wandb.log.assert_called_once()
        logged_data = mock_wandb.log.call_args[0][0]

        # Check completion data
        self.assertIn(f"{stage_name}.duration_seconds", logged_data)
        self.assertIn(f"{stage_name}.completed", logged_data)
        self.assertIn(f"{stage_name}.final.final_loss", logged_data)
        self.assertIn(f"{stage_name}.final.final_accuracy", logged_data)
        self.assertNotIn(f"{stage_name}.final.invalid_metric", logged_data)

        self.assertEqual(logged_data[f"{stage_name}.duration_seconds"], duration)
        self.assertTrue(logged_data[f"{stage_name}.completed"])

    def test_config_cleaning(self):
        """Test configuration cleaning for W&B."""
        logger = WandBLogger(self.project_name)

        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_name_or_path": "/local/path/to/model",
            "output_dir": "/local/output",
            "dataset_name_or_path": "/local/dataset.json",
            "wandb_project": "secret_project",
            "nested_dict": {"key": "value"},
            "list_param": [1, 2, 3],
            "none_param": None,
            "bool_param": True,
        }

        clean_config = logger._clean_config_for_wandb(config)

        # Check that sensitive fields are excluded
        self.assertNotIn("model_name_or_path", clean_config)
        self.assertNotIn("output_dir", clean_config)
        self.assertNotIn("dataset_name_or_path", clean_config)
        self.assertNotIn("wandb_project", clean_config)

        # Check that safe fields are included
        self.assertIn("learning_rate", clean_config)
        self.assertIn("batch_size", clean_config)
        self.assertIn("nested_dict", clean_config)
        self.assertIn("list_param", clean_config)
        self.assertIn("none_param", clean_config)
        self.assertIn("bool_param", clean_config)

        # Check model name extraction
        self.assertIn("model_name", clean_config)
        self.assertEqual(clean_config["model_name"], "model")

    def test_serializable_check(self):
        """Test serializable value checking."""
        logger = WandBLogger(self.project_name)

        # Test serializable values
        self.assertTrue(logger._is_serializable(None))
        self.assertTrue(logger._is_serializable(True))
        self.assertTrue(logger._is_serializable(42))
        self.assertTrue(logger._is_serializable(3.14))
        self.assertTrue(logger._is_serializable("string"))
        self.assertTrue(logger._is_serializable([1, 2, 3]))
        self.assertTrue(logger._is_serializable({"key": "value"}))

        # Test non-serializable values
        self.assertFalse(logger._is_serializable(object()))
        self.assertFalse(logger._is_serializable(lambda x: x))


class TestCreateWandBLogger(unittest.TestCase):
    """Test create_wandb_logger function."""

    def test_create_logger_enabled(self):
        """Test creating logger when W&B is enabled."""
        config = {
            "use_wandb": True,
            "wandb_project": "test_project",
            "wandb_run_name": "test_run",
            "stages": ["sft", "dpo"],
        }
        pipeline_id = "test_pipeline_123"

        logger = create_wandb_logger(config, pipeline_id)

        self.assertIsNotNone(logger)
        self.assertEqual(logger.project_name, "test_project")
        self.assertEqual(logger.run_name, "test_run")
        self.assertIn("lmpipeline", logger.tags)
        self.assertIn("stage:sft", logger.tags)
        self.assertIn("stage:dpo", logger.tags)

    def test_create_logger_disabled(self):
        """Test creating logger when W&B is disabled."""
        config = {"use_wandb": False}
        pipeline_id = "test_pipeline_123"

        logger = create_wandb_logger(config, pipeline_id)

        self.assertIsNone(logger)

    def test_create_logger_default_values(self):
        """Test creating logger with default values."""
        config = {"use_wandb": True}
        pipeline_id = "test_pipeline_123"

        logger = create_wandb_logger(config, pipeline_id)

        self.assertIsNotNone(logger)
        self.assertEqual(logger.project_name, "lmpipeline")
        self.assertEqual(logger.run_name, f"pipeline-{pipeline_id}")


if __name__ == "__main__":
    unittest.main()
