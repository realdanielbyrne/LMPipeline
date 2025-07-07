#!/usr/bin/env python3
"""
Main entry point for the modular fine-tuning pipeline.

This script provides a complete solution for multi-stage fine-tuning of language models
using a configurable pipeline that supports SFT, DPO, RLAIF, RL, CoT Distillation, and custom algorithm stages.

Author: Daniel Byrne
License: MIT
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .pipeline import Pipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Modular Fine-Tuning Pipeline for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SFT-only pipeline
  python -m lmpipeline.pipeline_main --config configs/sft_only_config.yaml

  # Run full multi-stage pipeline
  python -m lmpipeline.pipeline_main --config configs/pipeline_config.yaml

  # Run specific stages only
  python -m lmpipeline.pipeline_main --config configs/pipeline_config.yaml --stages sft dpo

  # Override model path
  python -m lmpipeline.pipeline_main --config configs/sft_only_config.yaml --model_name_or_path "microsoft/DialoGPT-large"
        """,
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    # Override options
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Override model path from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=None,
        help="Override stages to run (e.g., --stages sft dpo)",
    )

    # Global options
    parser.add_argument(
        "--log_level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level",
    )
    parser.add_argument(
        "--save_final_model",
        action="store_true",
        help="Save final model after all stages",
    )
    parser.add_argument(
        "--no_save_final_model",
        dest="save_final_model",
        action="store_false",
        help="Don't save final model",
    )
    parser.add_argument(
        "--cleanup_intermediate",
        action="store_true",
        help="Remove intermediate models to save disk space",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration and show pipeline plan without executing",
    )

    # Hugging Face Hub upload options
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the final model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Repository ID for Hugging Face Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--hub_commit_message",
        type=str,
        default=None,
        help="Commit message for Hub upload",
    )
    parser.add_argument(
        "--hub_private", action="store_true", help="Create private repository on Hub"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--push_adapter_only",
        action="store_true",
        help="Only upload LoRA adapter files to Hub (not the full model)",
    )

    # GGUF conversion options
    parser.add_argument(
        "--convert_to_gguf",
        action="store_true",
        help="Convert final model to GGUF format",
    )
    parser.add_argument(
        "--gguf_quantization",
        type=str,
        default="q4_0",
        help="GGUF quantization type (q4_0, q8_0, f16, etc.)",
    )
    parser.add_argument(
        "--gguf_output_path",
        type=str,
        default=None,
        help="Output path for GGUF file (defaults to output_dir/model.gguf)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = PipelineConfig.from_yaml(args.config)

        # Apply command line overrides
        if args.model_name_or_path:
            config.model_name_or_path = args.model_name_or_path
            logger.info(f"Override model: {args.model_name_or_path}")

        if args.output_dir:
            config.output_dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")

        if args.stages:
            config.stages = args.stages
            logger.info(f"Override stages: {args.stages}")

        if args.log_level:
            config.log_level = args.log_level

        if hasattr(args, "save_final_model") and args.save_final_model is not None:
            config.save_final_model = args.save_final_model

        if args.cleanup_intermediate:
            config.cleanup_intermediate = args.cleanup_intermediate

        # Apply post-processing overrides
        if args.push_to_hub:
            config.push_to_hub = args.push_to_hub
        if args.hub_repo_id:
            config.hub_repo_id = args.hub_repo_id
        if args.hub_commit_message:
            config.hub_commit_message = args.hub_commit_message
        if args.hub_private:
            config.hub_private = args.hub_private
        if args.hub_token:
            config.hub_token = args.hub_token
        if args.push_adapter_only:
            config.push_adapter_only = args.push_adapter_only
        if args.convert_to_gguf:
            config.convert_to_gguf = args.convert_to_gguf
        if args.gguf_quantization:
            config.gguf_quantization = args.gguf_quantization
        if args.gguf_output_path:
            config.gguf_output_path = args.gguf_output_path

        # Validate configuration
        logger.info("Validating configuration...")
        if not config.model_name_or_path:
            raise ValueError("model_name_or_path is required")

        if not config.output_dir:
            raise ValueError("output_dir is required")

        if not config.stages:
            raise ValueError("At least one stage must be specified")

        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline = Pipeline(config)

        # Show pipeline plan
        logger.info("Pipeline execution plan:")
        logger.info(f"  Model: {config.model_name_or_path}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Stages: {' -> '.join(config.stages)}")
        logger.info(f"  Save final model: {config.save_final_model}")
        logger.info(f"  Cleanup intermediate: {config.cleanup_intermediate}")

        # Show applied defaults information
        from .utils.config_defaults import ConfigDefaults

        if config.convert_to_gguf and config.gguf_output_path:
            logger.info(f"  GGUF output path: {config.gguf_output_path}")

        # Show environment variable overrides if any
        env_vars = [
            (ConfigDefaults.ENV_MODELS_DIR, ConfigDefaults.get_default_models_dir()),
            (
                ConfigDefaults.ENV_CHECKPOINTS_DIR,
                ConfigDefaults.get_default_checkpoints_dir(),
            ),
            (ConfigDefaults.ENV_OUTPUT_DIR, ConfigDefaults.get_default_output_dir()),
        ]

        active_env_vars = [(var, val) for var, val in env_vars if var in os.environ]
        if active_env_vars:
            logger.info("  Environment variable overrides:")
            for var, val in active_env_vars:
                logger.info(f"    {var}: {val}")

        # Show post-processing plan
        if config.convert_to_gguf or config.push_to_hub:
            logger.info("Post-processing steps:")
            if config.convert_to_gguf:
                logger.info(f"  GGUF conversion: {config.gguf_quantization}")
                if config.gguf_output_path:
                    logger.info(f"    Output path: {config.gguf_output_path}")
            if config.push_to_hub:
                logger.info(f"  Hub upload: {config.hub_repo_id}")
                if config.hub_private:
                    logger.info("    Private repository")
                if config.push_adapter_only:
                    logger.info("    Adapter-only upload")

        if args.dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            logger.info("Pipeline would execute the following stages:")
            for i, stage_name in enumerate(config.stages, 1):
                stage_config = config.stage_configs.get(stage_name, {})
                logger.info(f"  {i}. {stage_name.upper()}")
                if stage_config:
                    logger.info(f"     Config keys: {list(stage_config.keys())}")
            return

        # Execute pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.execute()

        # Show results
        summary = pipeline.get_summary()
        logger.info("Pipeline execution completed!")
        logger.info(f"Summary:")
        logger.info(f"  Total stages: {summary['total_stages']}")
        logger.info(f"  Executed stages: {summary['executed_stages']}")
        logger.info(f"  Successful stages: {summary['successful_stages']}")
        logger.info(f"  Failed stages: {summary['failed_stages']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2%}")

        # Show stage results
        for result in summary["stage_results"]:
            status = "✅" if result["success"] else "❌"
            logger.info(f"  {status} {result['stage_name'].upper()}")
            if result["error"]:
                logger.error(f"    Error: {result['error']}")
            elif result["metrics"]:
                key_metrics = {
                    k: v
                    for k, v in result["metrics"].items()
                    if k in ["eval_loss", "train_loss", "accuracy", "perplexity"]
                }
                if key_metrics:
                    logger.info(f"    Metrics: {key_metrics}")

        # Final model location
        if config.save_final_model and results and results[-1].success:
            final_model_path = Path(config.output_dir) / "final_model"
            logger.info(f"Final model saved to: {final_model_path}")

        # Exit with appropriate code
        if summary["failed_stages"] > 0:
            logger.error("Pipeline completed with failures")
            sys.exit(1)
        else:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
