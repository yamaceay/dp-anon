"""
Main entry point using the refactored DPMLM architecture.

This module demonstrates the new high-level, type-safe, plug-and-play
interface for differential privacy text processing.
"""

import argparse
import logging
import sys
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
    if args.type == "dpmlm":
        risk_type = (args.dpmlm_risk_type or "uniform").lower()
        dp_config = {
            "epsilon": args.epsilon,
            "device": "auto",
            "process_pii_only": args.dpmlm_annotator is not None,
            "use_temperature": True,
            "add_probability": 0.15 if args.dpmlm_plus else 0.0,
            "delete_probability": 0.05 if args.dpmlm_plus else 0.0,
        }
        risk_config = {
            "explainability_mode": risk_type,
            "pii_threshold": args.dpmlm_pii_threshold,
        }
        return {"dpmlm": dp_config, "risk": risk_config}

    return {
        "epsilon": args.epsilon,
        "device": "auto"
    }


def normalise_dpmlm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure configuration has explicit DPMLM/risk sections."""
    config = dict(config or {})
    has_nested = any(
        key in config
        for key in ("dpmlm", "dpmlm_config", "risk", "risk_settings", "risk_config")
    )

    if not has_nested:
        return {"dpmlm": config, "risk": {}}

    if "dpmlm" not in config:
        if "dpmlm_config" in config:
            config["dpmlm"] = config.pop("dpmlm_config")
        else:
            config["dpmlm"] = {}

    if "risk" not in config:
        if "risk_settings" in config:
            config["risk"] = config.pop("risk_settings")
        elif "risk_config" in config:
            config["risk"] = config.pop("risk_config")
        else:
            config["risk"] = {}

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
        "--dpmlm-plus",
        dest="dpmlm_plus",
        action="store_true",
        help="Use 'plus' method with addition/deletion (DPMLM only)"
    )
    parser.add_argument(
        "--dpmlm-annotator",
        dest="dpmlm_annotator",
        type=str,
        default=None,
        help="Path to PII annotator model (mutually exclusive with non-uniform risk scoring)"
    )
    parser.add_argument(
        "--dpmlm-risk",
        "--dpmlm-risk-type",
        dest="dpmlm_risk_type",
        type=str,
        default="uniform",
        choices=["uniform", "shap", "greedy"],
        help="Explainability mode for DPMLM risk allocation"
    )
    parser.add_argument(
        "--dpmlm-risk-model",
        dest="dpmlm_risk_model",
        type=str,
        default=None,
        help="Transformers model name or path for risk-aware scoring (required for SHAP/greedy)"
    )
    parser.add_argument(
        "--dpmlm-pii-threshold",
        dest="dpmlm_pii_threshold",
        type=float,
        default=0.0,
        help="Score threshold for keeping PII predictions when process_pii_only is enabled"
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
        default="outputs/DPMLM",
        help="Output directory for anonymized data"
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
        seen = set()
        for mechanism in dpmlm.list_mechanisms():
            if mechanism in seen:
                continue
            print(f"  - {mechanism}")
            if mechanism == "dpmlm":
                print("      * uniform (default)")
                print("      * shap")
                print("      * greedy")
            seen.add(mechanism)
        return
        
    if args.list_presets:
        print("Available presets:")
        for preset in dpmlm.list_presets():
            print(f"  - {preset}")
        return
    
    
    text_input = args.text
    stdin_buffer: Optional[str] = None
    if (text_input is None or text_input == "-") and not sys.stdin.isatty():
        stdin_buffer = sys.stdin.read()
        text_input = stdin_buffer if text_input in (None, "-") else text_input

    if not text_input:
        parser.error("Input text is required. Provide as an argument or pipe via stdin.")


    text = recode_text(text_input.rstrip("\n"))
    logger.info("Processing text of length: %d characters", len(text))
    
    try:
        
        config: Dict[str, Any]
        if args.config:
            import json
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Loaded configuration from: %s", args.config)
        else:
            config = build_config_from_args(args)

        risk_type = (args.dpmlm_risk_type or "uniform").lower() if args.type == "dpmlm" else "uniform"

        risk_pipeline = None
        if args.type == "dpmlm" and risk_type == "uniform" and args.dpmlm_risk_model:
            logger.warning("Ignoring --dpmlm-risk-model because DPMLM risk type is 'uniform'.")

        if args.type == "dpmlm" and risk_type != "uniform":
            if not args.dpmlm_risk_model:
                parser.error("Non-uniform DPMLM risk scoring requires --dpmlm-risk-model with a text-classification model.")
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as exc:
                parser.error(f"Transformers is required for risk-aware scoring: {exc}")

            try:
                import torch
                device = (
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu"
                )
            except ImportError as exc:
                parser.error(f"PyTorch is required for risk-aware scoring: {exc}")

            logger.info("Loading risk scoring pipeline: %s", args.dpmlm_risk_model)
            risk_pipeline = hf_pipeline(
                "text-classification",
                model=args.dpmlm_risk_model,
                tokenizer=args.dpmlm_risk_model,
                top_k=None,
                device=device,
                truncation=True,
                max_length=512,
            )

        annotator_obj: Optional[PIIDeidentifier] = None
        if args.type == "dpmlm" and args.dpmlm_annotator:
            annotator_obj = create_annotator(args.dpmlm_annotator, args.data_out)

        if args.type == "dpmlm":
            config = normalise_dpmlm_config(config)
            dp_section = config.setdefault("dpmlm", {})
            risk_section = config.setdefault("risk", {})

            dp_section.setdefault("device", device)
            dp_section.setdefault("epsilon", args.epsilon)
            if annotator_obj is not None:
                dp_section["process_pii_only"] = True
            elif "process_pii_only" not in dp_section:
                dp_section["process_pii_only"] = False

            if args.dpmlm_plus:
                dp_section["add_probability"] = 0.15
                dp_section["delete_probability"] = 0.05
            else:
                dp_section.setdefault("add_probability", 0.0)
                dp_section.setdefault("delete_probability", 0.0)

            risk_section["explainability_mode"] = risk_type
            risk_section.setdefault("pii_threshold", args.dpmlm_pii_threshold)
            if risk_pipeline is not None:
                mask_token = getattr(getattr(risk_pipeline, "tokenizer", None), "mask_token", None)
                if mask_token:
                    risk_section.setdefault("mask_text", mask_token)
                risk_section["risk_pipeline"] = risk_pipeline
            if annotator_obj is not None:
                risk_section["annotator"] = annotator_obj

        if args.preset:
            logger.info("Using preset configuration: %s", args.preset)
            mechanism = dpmlm.create_from_preset(
                args.preset,
                override_config=config,
            )
        else:
            logger.info("Creating %s mechanism", args.type)
            mechanism = dpmlm.create_mechanism(
                args.type,
                config=config,
            )
        
        
        logger.info("Applying differential privacy with epsilon=%.3f", args.epsilon)
        result = mechanism.privatize(
            text,
            epsilon=args.epsilon,
            plus=args.dpmlm_plus,
            method="patch" if args.type == "dpmlm" else "default",
        )
        
        
        print("\n" + "="*80)
        print("DIFFERENTIAL PRIVACY TEXT PROCESSING RESULTS")
        print("="*80)
        print(f"Mechanism: {args.type}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Plus method: {'Yes' if args.dpmlm_plus else 'No'}")
        print(f"Risk type: {risk_type if args.type == 'dpmlm' else 'N/A'}")
        print(f"PII annotation: {'Yes' if annotator_obj else 'No'}")
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
            token_allocations = result.metadata.get("token_allocations")
            token_summary = result.metadata.get("token_allocations_summary", [])
            other_metadata = {
                key: value
                for key, value in result.metadata.items()
                if key not in {"token_allocations", "token_allocations_summary"}
            }

            if other_metadata:
                print("Metadata:")
                for key, value in other_metadata.items():
                    print(f"  {key}: {value}")

            if token_summary:
                print("Top risk-weighted tokens:")
                for entry in token_summary:
                    token_text = entry.get("token", "")
                    token_display = token_text.replace('\n', ' ').strip()
                    if len(token_display) > 30:
                        token_display = token_display[:27] + "..."
                    entity_fragment = ""
                    if entry.get("entity_type"):
                        entity_fragment = f" entity={entry['entity_type']}"

                    print(
                        "  #{rank:>2} [{start}-{end}] '{token}' weight={weight:.4g} "
                        "epsilon={epsilon:.4g} score={score:.4g}{entity}".format(
                            rank=entry.get("rank", 0),
                            start=entry.get("start"),
                            end=entry.get("end"),
                            token=token_display,
                            weight=entry.get("weight", 0.0),
                            epsilon=entry.get("epsilon", 0.0),
                            score=entry.get("score", 0.0),
                            entity=entity_fragment,
                        )
                    )
                if token_allocations is not None:
                    print(f"  (total tokens evaluated: {len(token_allocations)})")
            elif token_allocations is not None:
                print(f"Token allocations recorded: {len(token_allocations)}")
        
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
