"""
Main entry point using the refactored DPMLM architecture.

This module demonstrates the new high-level, type-safe, plug-and-play
interface for differential privacy text processing.
"""

import argparse
import json
import logging
import sys
from typing import Optional, Dict, Any

import dpmlm
from dpmlm.config_utils import (
    DPMLM_GENERIC_KEYS,
    coerce_dpmlm_config,
    prepare_dpmlm_model_config,
)
from pii import DataLabels, TorchTokenClassifier, PIIDeidentifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recode_text(text: str) -> str:
    """Decode escape sequences in text while preserving Unicode."""
    replacements = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        '\\"': '"',
        "\\'": "'",
    }
    for escaped, replacement in replacements.items():
        text = text.replace(escaped, replacement)
    text = text.replace("\\\\", "\\")
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
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load PII annotator: %s", exc)
        return None


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """Load JSON configuration if a path is supplied."""
    if not path:
        return {}

    logger.info("Loading configuration file: %s", path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a JSON object at the top level.")

    return data


def _filter_section(data: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if key in allowed}


def _print_token_allocation(entry: Dict[str, Any]) -> None:
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


def main() -> int:
    """Main entry point with improved architecture."""
    parser = argparse.ArgumentParser(
        description="Differential Privacy Text Processing with High-Level Interface"
    )
    parser.add_argument("text", type=str, nargs="?", help="Optional input text")
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="dpmlm",
        choices=["dpmlm", "dpprompt", "dpparaphrase", "dpbart"],
        help="Type of DP mechanism to use",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=1.0,
        help="Privacy parameter epsilon",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device preference (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to JSON configuration file containing model-specific parameters",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Use preset configuration",
    )
    parser.add_argument(
        "--data-out",
        dest="data_out",
        type=str,
        default="outputs/DPMLM",
        help="Output directory used when loading annotators",
    )
    parser.add_argument(
        "--list-mechanisms",
        action="store_true",
        help="List available mechanisms and exit",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_mechanisms:
        print("Available mechanisms:")
        for mechanism in sorted(set(dpmlm.list_mechanisms())):
            print(f"  - {mechanism}")
        return 0

    if args.list_presets:
        print("Available presets:")
        for preset in sorted(set(dpmlm.list_presets())):
            print(f"  - {preset}")
        return 0

    text_input = args.text
    stdin_buffer: Optional[str] = None
    if (text_input is None or text_input == "-") and not sys.stdin.isatty():
        stdin_buffer = sys.stdin.read()
        if text_input in (None, "-"):
            text_input = stdin_buffer

    try:
        raw_config = load_config_file(args.config)
    except (ValueError, RuntimeError) as exc:
        logger.error("Failed to load configuration: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    mechanism_type = args.type
    config_for_factory: Dict[str, Any]

    text_candidates = []
    if text_input:
        text_candidates.append(text_input)
    if stdin_buffer and stdin_buffer not in text_candidates:
        text_candidates.append(stdin_buffer)

    if mechanism_type == "dpmlm":
        structured = coerce_dpmlm_config(raw_config)
        runtime_section = structured.get("runtime", {})
        runtime_text = runtime_section.get("input_text")
        if isinstance(runtime_text, str):
            text_candidates.append(runtime_text)

        generic_cfg = _filter_section(structured.get("generic", {}), DPMLM_GENERIC_KEYS)
        if args.device:
            generic_cfg["device"] = args.device
        if args.seed is not None:
            generic_cfg["seed"] = args.seed
        if args.verbose:
            generic_cfg["verbose"] = True

        model_cfg = prepare_dpmlm_model_config(
            structured.get("model", {}),
            generic_cfg,
            annotator_loader=lambda path: create_annotator(path, args.data_out),
        )

        text_to_process = next((candidate for candidate in text_candidates if candidate), None)
        if not text_to_process:
            parser.error(
                "Input text is required. Provide it as an argument, via stdin, or in runtime.input_text."
            )

        text_to_process = recode_text(text_to_process.rstrip("\n"))
        logger.info("Processing text of length: %d characters", len(text_to_process))

        runtime_cfg = {"input_text": text_to_process}

        config_for_factory = {
            "generic": generic_cfg,
            "model": model_cfg,
            "runtime": runtime_cfg,
        }
    else:
        config_for_factory = dict(raw_config or {})
        if args.device:
            config_for_factory.setdefault("device", args.device)
        if args.seed is not None:
            config_for_factory.setdefault("seed", args.seed)
        if args.verbose:
            config_for_factory["verbose"] = True

        text_to_process = next((candidate for candidate in text_candidates if candidate), None)
        if not text_to_process:
            parser.error("Input text is required. Provide it as an argument or via stdin.")

        text_to_process = recode_text(text_to_process.rstrip("\n"))
        logger.info("Processing text of length: %d characters", len(text_to_process))

    try:
        if args.preset:
            logger.info("Using preset configuration: %s", args.preset)
            mechanism = dpmlm.create_from_preset(
                args.preset,
                override_config=config_for_factory,
            )
        else:
            logger.info("Creating %s mechanism", mechanism_type)
            mechanism = dpmlm.create_mechanism(
                mechanism_type,
                config=config_for_factory,
            )

        logger.info("Applying differential privacy with epsilon=%.3f", args.epsilon)
        result = mechanism.privatize(
            text_to_process,
            epsilon=args.epsilon,
        )

        print("\n" + "=" * 80)
        print("DIFFERENTIAL PRIVACY TEXT PROCESSING RESULTS")
        print("=" * 80)
        print(f"Mechanism: {args.type}")
        print(f"Epsilon: {args.epsilon}")
        if args.type == "dpmlm":
            model_section = config_for_factory.get("model", {})
            explainability = (model_section.get("explainability_mode") or "uniform").lower()
            pii_enabled = bool(model_section.get("annotator"))
            print(f"Explainability mode: {explainability}")
            print(f"PII annotation: {'Yes' if pii_enabled else 'No'}")
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
            other_metadata = {
                key: value
                for key, value in result.metadata.items()
                if key not in {"token_allocations"}
            }

            if other_metadata:
                print("Metadata:")
                for key, value in other_metadata.items():
                    print(f"  {key}: {value}")

            if token_allocations:
                print("Token risk allocations:")
                showed_token_allocations = token_allocations
                truncate_results = len(token_allocations) >= 10
                if truncate_results:
                    showed_token_allocations = token_allocations[:5] + token_allocations[-5:]

                for idx, entry in enumerate(showed_token_allocations):
                    if truncate_results and idx == 5:
                        print("  ...")
                    _print_token_allocation(entry)

        print("=" * 80)

    except (ImportError, ValueError, FileNotFoundError) as exc:
        logger.error("Error processing text: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
