"""
Main application orchestrator for TRI (Text Re-Identification) system.

This module coordinates all components using dependency injection and provides
the main entry point for the application.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Any

from tri.config import RuntimeConfig, canonicalize_config_from_dict
from tri.core import (
    TRIDataProcessor, TRIDatasetBuilder, TRIModelManager, TRIPredictor,
    TRIConfigManager, TRIStorageManager, TRIWorkflowOrchestrator
)
from tri.cli import (
    parse_arguments, load_config_file, print_welcome, print_goodbye,
    print_error, setup_logging, confirm_configuration, print_data_statistics,
    print_phase_start, print_phase_complete, print_results, handle_keyboard_interrupt, 
    print_execution_estimate, print_model_info, print_resource_usage
)

logger = logging.getLogger(__name__)


def create_tri_orchestrator() -> TRIWorkflowOrchestrator:
    """Create TRI workflow orchestrator with dependency injection."""
    
    config_manager = TRIConfigManager()
    storage_manager = TRIStorageManager()
    data_processor = TRIDataProcessor()
    dataset_builder = TRIDatasetBuilder()
    model_manager = TRIModelManager()
    predictor = TRIPredictor()
    
    
    orchestrator = TRIWorkflowOrchestrator(
        config_manager=config_manager,
        data_processor=data_processor,
        dataset_builder=dataset_builder,
        model_manager=model_manager,
        predictor=predictor,
        storage_manager=storage_manager
    )
    
    logger.info("orchestrator_created", extra={"components": 7})
    return orchestrator


def run_tri_workflow(config: RuntimeConfig, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete TRI workflow.
    
    Args:
        config: Runtime configuration
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing results (evaluation results or annotation results)
    """
    logger.info("workflow_start", extra={"config": config.output_folder_path})
    
    
    orchestrator = create_tri_orchestrator()
    
    
    orchestrator.config_manager.validate_config(config)
    
    try:
        
        
        if verbose:
            print_phase_start("data_processing")
        
        data_info = orchestrator.run_data_processing(config)
        
        if verbose:
            print_data_statistics(data_info, config)
            print_phase_complete("data_processing")
        
        
        if verbose:
            print_phase_start("model_building")
        
        model_info = orchestrator.run_model_building(data_info, config)
        
        if verbose:
            print_model_info(model_info)
            print_phase_complete("model_building")
        
        
        if verbose:
            print_phase_start("prediction")
        
        results = orchestrator.run_prediction(model_info, config)
        
        if verbose:
            print_phase_complete("prediction")
            print_results(results)
            print_resource_usage()
        
        logger.info("workflow_complete", extra={
            "results": {name: res.get('eval_Accuracy', 0) for name, res in results.items()}
        })
        
        return results
    
    except Exception as e:
        logger.error("workflow_error", extra={"error": str(e), "type": type(e).__name__})
        raise


def main() -> int:
    """
    Main entry point for TRI application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print_welcome()
    
    try:
        
        setup_logging(verbose=True)
        
        
        config_file_path = parse_arguments()
        
        
        config_dict = load_config_file(config_file_path)
        config = canonicalize_config_from_dict(config_dict)
        
        
        if not confirm_configuration(config):
            print("\n❌ Configuration not confirmed. Exiting.")
            return 1
        
        
        print_execution_estimate(config)
        
        
        results = run_tri_workflow(config, verbose=True)
        
        print_goodbye()
        return 0
    
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
        return 1
    
    except Exception as e:
        print_error(e)
        logger.exception("main_error")
        return 1


def run_tri_from_dict(config_dict: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Run TRI workflow from configuration dictionary (programmatic interface).
    
    Args:
        config_dict: Configuration dictionary
        verbose: Whether to show progress output
        
    Returns:
        Dictionary containing results (evaluation or annotation)
    """
    config = canonicalize_config_from_dict(config_dict)
    return run_tri_workflow(config, verbose=verbose)


def run_tri_from_config(config: RuntimeConfig, verbose: bool = True) -> Dict[str, Any]:
    """
    Run TRI workflow from RuntimeConfig object (programmatic interface).
    
    Args:
        config: Runtime configuration object
        verbose: Whether to show progress output
        
    Returns:
        Dictionary containing results (evaluation or annotation)
    """
    return run_tri_workflow(config, verbose=verbose)


if __name__ == "__main__":
    sys.exit(main())