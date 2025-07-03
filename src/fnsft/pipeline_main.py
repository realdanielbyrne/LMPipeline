#!/usr/bin/env python3
"""
Main entry point for the modular fine-tuning pipeline.

This script provides a complete solution for multi-stage fine-tuning of language models
using a configurable pipeline that supports SFT, DPO, RLAIF, RL, and CoT Distillation.

Author: Daniel Byrne
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import Pipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
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
  python -m fnsft.pipeline_main --config configs/sft_only_config.yaml
  
  # Run full multi-stage pipeline
  python -m fnsft.pipeline_main --config configs/pipeline_config.yaml
  
  # Run specific stages only
  python -m fnsft.pipeline_main --config configs/pipeline_config.yaml --stages sft dpo
  
  # Override model path
  python -m fnsft.pipeline_main --config configs/sft_only_config.yaml --model_name_or_path "microsoft/DialoGPT-large"
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Override options
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None,
        help="Override model path from config"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--stages", 
        type=str, 
        nargs="+", 
        default=None,
        help="Override stages to run (e.g., --stages sft dpo)"
    )
    
    # Global options
    parser.add_argument(
        "--log_level", 
        type=str, 
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level"
    )
    parser.add_argument(
        "--save_final_model", 
        action="store_true",
        help="Save final model after all stages"
    )
    parser.add_argument(
        "--no_save_final_model", 
        dest="save_final_model", 
        action="store_false",
        help="Don't save final model"
    )
    parser.add_argument(
        "--cleanup_intermediate", 
        action="store_true",
        help="Remove intermediate models to save disk space"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Validate configuration and show pipeline plan without executing"
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
        
        if hasattr(args, 'save_final_model') and args.save_final_model is not None:
            config.save_final_model = args.save_final_model
        
        if args.cleanup_intermediate:
            config.cleanup_intermediate = args.cleanup_intermediate
        
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
        for result in summary['stage_results']:
            status = "✅" if result['success'] else "❌"
            logger.info(f"  {status} {result['stage_name'].upper()}")
            if result['error']:
                logger.error(f"    Error: {result['error']}")
            elif result['metrics']:
                key_metrics = {k: v for k, v in result['metrics'].items() 
                             if k in ['eval_loss', 'train_loss', 'accuracy', 'perplexity']}
                if key_metrics:
                    logger.info(f"    Metrics: {key_metrics}")
        
        # Final model location
        if config.save_final_model and results and results[-1].success:
            final_model_path = Path(config.output_dir) / "final_model"
            logger.info(f"Final model saved to: {final_model_path}")
        
        # Exit with appropriate code
        if summary['failed_stages'] > 0:
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
