"""
Main entry point using the refactored DPMLM architecture.

This module demonstrates the new high-level, type-safe, plug-and-play
interface for differential privacy text processing.
"""

import argparse
import logging
from typing import Optional, Dict, Any


import dpmlm
from pii import DataLabels, TorchTokenClassifier, PIIDeidentifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recode_text(text: str) -> str:
    """Decode escape sequences in text while preserving Unicode."""
    escape_sequences = {
        '\\n': '\n',
        '\\t': '\t', 
        '\\r': '\r',
        '\\"': '"',
        "\\'": "'",
        '\\\\': '\\'
    }
    
    for escaped, unescaped in escape_sequences.items():
        text = text.replace(escaped, unescaped)
    return text


def create_annotator(annotator_path: str, data_out: str) -> Optional[PIIDeidentifier]:
    """Create PII annotator if path is provided."""
    if not annotator_path:
        return None
        
    logger.info("Loading PII annotator from: %s", annotator_path)
    
    
    unique_labels = ['CODE', 'DEM', 'ORG', 'QUANTITY', 'LOC', 'DATETIME', 'MISC', 'PERSON']
    labels = ['O'] + [f'B-{label}' for label in unique_labels] + [f'I-{label}' for label in unique_labels]
    
    try:
        labels_obj = DataLabels(labels)
        with TorchTokenClassifier(annotator_path, labels_obj) as (model, tokenizer):
            annotator = PIIDeidentifier(data_out, model, tokenizer, labels_obj)
            logger.info("PII annotator loaded successfully")
            return annotator
    except (ImportError, FileNotFoundError, ValueError) as e:
        logger.error("Failed to load PII annotator: %s", e)
        return None


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration dictionary from command line arguments."""
    config = {
        "epsilon": args.epsilon,
        "device": "auto"
    }
    
    if args.type == "dpmlm":
        config.update({
            "process_pii_only": args.annotator is not None,
            "use_temperature": True,
            "add_probability": 0.15 if args.plus else 0.0,
            "delete_probability": 0.05 if args.plus else 0.0
        })
    
    return config


def main():
    """Main entry point with improved architecture."""
    parser = argparse.ArgumentParser(
        description="Differential Privacy Text Processing with High-Level Interface"
    )
    parser.add_argument("text", type=str, nargs='?', help="Input text to process")
    parser.add_argument(
        "--type", "-t", 
        type=str, 
        default="dpmlm",
        choices=["dpmlm", "dpprompt", "dpparaphrase", "dpbart"],
        help="Type of DP mechanism to use"
    )
    parser.add_argument(
        "--plus", 
        action="store_true",
        help="Use 'plus' method with addition/deletion (DPMLM only)"
    )
    parser.add_argument(
        "--annotator", 
        type=str, 
        default=None,
        help="Path to PII annotator model"
    )
    parser.add_argument(
        "--epsilon", "-e", 
        type=float, 
        default=1.0,
        help="Privacy parameter epsilon"
    )
    parser.add_argument(
        "--data_out", 
        type=str, 
        default="pii/outputs",
        help="Output directory for PII processing"
    )
    parser.add_argument(
        "--config", 
        type=str,
        default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Use preset configuration"
    )
    parser.add_argument(
        "--list-mechanisms",
        action="store_true",
        help="List available mechanisms and exit"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true", 
        help="List available presets and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    
    if args.list_mechanisms:
        print("Available mechanisms:")
        for mechanism in dpmlm.list_mechanisms():
            print(f"  - {mechanism}")
        return
        
    if args.list_presets:
        print("Available presets:")
        for preset in dpmlm.list_presets():
            print(f"  - {preset}")
        return
    
    
    if not args.text:
        parser.error("Input text is required for text processing. Use --list-mechanisms or --list-presets to see available options.")
    
    
    text = recode_text(args.text)
    logger.info("Processing text of length: %d characters", len(text))
    
    try:
        
        config = None
        if args.config:
            import json
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Loaded configuration from: %s", args.config)
        else:
            config = build_config_from_args(args)
        
        
        annotator = None
        if args.annotator and args.type == "dpmlm":
            annotator = create_annotator(args.annotator, args.data_out)
        
        
        if args.preset:
            logger.info("Using preset configuration: %s", args.preset)
            mechanism = dpmlm.create_from_preset(
                args.preset, 
                annotator=annotator,
                override_config={"epsilon": args.epsilon}
            )
        else:
            logger.info("Creating %s mechanism", args.type)
            mechanism = dpmlm.create_mechanism(
                args.type,
                config=config,
                annotator=annotator
            )
        
        
        logger.info("Applying differential privacy with epsilon=%.3f", args.epsilon)
        result = mechanism.privatize(
            text, 
            epsilon=args.epsilon,
            plus=args.plus,
            method="patch" if args.type == "dpmlm" else "default"
        )
        
        
        print("\n" + "="*80)
        print("DIFFERENTIAL PRIVACY TEXT PROCESSING RESULTS")
        print("="*80)
        print(f"Mechanism: {args.type}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Plus method: {'Yes' if args.plus else 'No'}")
        print(f"PII annotation: {'Yes' if annotator else 'No'}")
        print()
        
        print("Original text:")
        print(repr(result.original_text))
        print()
        
        print("Private text:")
        print(repr(result.private_text))
        print()
        
        print("Statistics:")
        print(f"  Perturbed tokens: {result.perturbed_tokens}/{result.total_tokens}")
        print(f"  Perturbation rate: {result.perturbation_rate:.2%}")
        
        if result.added_tokens > 0 or result.deleted_tokens > 0:
            print(f"  Added tokens: {result.added_tokens}")
            print(f"  Deleted tokens: {result.deleted_tokens}")
        
        if result.metadata:
            print("Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        
        print("="*80)
        
    except (ImportError, ValueError, FileNotFoundError) as e:
        logger.error("Error processing text: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
