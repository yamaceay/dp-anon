"""
Main entry point using the refactored DPMLM architecture.

This module demonstrates the new high-level, type-safe, plug-and-play
interface for differential privacy text processing.
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional, Dict, Any

from transformers import pipeline as hf_pipeline
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

import dpmlm
from dpmlm.config_utils import (
    DPMLM_GENERIC_KEYS,
    DPMLM_RUNTIME_KEYS,
    coerce_dpmlm_config,
    prepare_dpmlm_model_config,
)
from petre import PETRE
from pii import DataLabels, TorchTokenClassifier, PIIDeidentifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

TRI_PIPELINE_PATH = "models/tri_pipelines/trustpilot/www.amazon.com/TRI_Pipeline"
OUTPUT_DIR = "outputs/trustpilot/www.amazon.com/dpmlm"
DATASET_PATH = "data/trustpilot/www.amazon.com/train.json"

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


def main() -> int:
    """Main entry point with improved architecture."""
    parser = argparse.ArgumentParser(
        description="Differential Privacy Text Processing with High-Level Interface"
    )
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

    if mechanism_type == "dpmlm":
        structured = coerce_dpmlm_config(raw_config)
        generic_cfg = _filter_section(structured.get("generic", {}), DPMLM_GENERIC_KEYS)
        generic_cfg["device"] = args.device if args.device else device
        if args.seed is not None:
            generic_cfg["seed"] = args.seed
        if args.verbose:
            generic_cfg["verbose"] = True

        model_cfg = prepare_dpmlm_model_config(
            structured.get("model", {}),
            generic_cfg,
        )

        config_for_factory = {
            "generic": generic_cfg,
            "model": model_cfg,
        }
    else:
        config_for_factory = dict(raw_config or {})
        if args.device:
            config_for_factory.setdefault("device", args.device)
        if args.seed is not None:
            config_for_factory.setdefault("seed", args.seed)
        if args.verbose:
            config_for_factory["verbose"] = True

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

    dataset = load_dataset('json', data_files={'train': DATASET_PATH})['train']
    texts = [recode_text(row['review']) for row in dataset]
    review_ids = [str(row['review_id']) for row in dataset]
    sorted_ids = sorted(set(review_ids))
    name_to_label_idx = {name: idx for idx, name in enumerate(sorted_ids)}
    target_labels = [name_to_label_idx[name] for name in review_ids]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tri_pipeline = hf_pipeline(
        "text-classification",
        model=TRI_PIPELINE_PATH,
        tokenizer=TRI_PIPELINE_PATH,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )

    initial_ranks = evaluate(tri_pipeline, texts, target_labels)
    initial_ranks_file_path = f"{OUTPUT_DIR}/ranks_initial.csv"
    np.savetxt(initial_ranks_file_path, initial_ranks, delimiter=",", fmt="%d")
    logger.info("Initial ranks saved to %s", initial_ranks_file_path)

    anonymized_texts = []
    for idx, text in enumerate(tqdm(texts, desc="Applying differential privacy")):
        logger.info("Applying differential privacy with epsilon=%.3f", args.epsilon)
        try:
            anonymized_text = mechanism.privatize(
                text,
                epsilon=args.epsilon,
            ).private_text
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Privatization failed for document %d; storing empty text. Error: %s",
                idx,
                exc,
            )
            anonymized_text = ""
        anonymized_texts.append(anonymized_text)

    with open(f"{OUTPUT_DIR}/{args.type}_{args.epsilon}.json", "w", encoding="utf-8") as f:
        for orig, anon in zip(texts, anonymized_texts):
            json.dump({"original": orig, "anonymized": anon}, f)
            f.write("\n")

    ranks = evaluate(tri_pipeline, anonymized_texts, target_labels)
    ranks_file_path = f"{OUTPUT_DIR}/ranks_{args.type}_{args.epsilon}.csv"
    np.savetxt(ranks_file_path, ranks, delimiter=",", fmt="%d")
    logger.info("Ranks saved to %s", ranks_file_path)

def evaluate(
    tri_pipeline,
    anonymized_texts,
    target_labels,
    batch_size: int = 128,
) -> np.ndarray:
    """Return the rank of each target label on its (possibly anonymized) counterpart."""
    if len(anonymized_texts) != len(target_labels):
        raise ValueError("target_labels length must match texts length")

    if len(anonymized_texts) == 0:
        return np.empty(0, dtype=np.int32)

    empty_indices = {i for i, anon_text in enumerate(anonymized_texts) if not anon_text.strip()}
    eval_indices = [i for i in range(len(anonymized_texts)) if i not in empty_indices]
    eval_texts = [anonymized_texts[i] for i in eval_indices]

    if eval_texts:
        predictions = tri_pipeline(eval_texts, batch_size=batch_size)
    else:
        predictions = []

    if not isinstance(predictions, list):
        predictions = [predictions]

    config = getattr(tri_pipeline, "model", None)
    config = getattr(config, "config", None)
    id2label_conf = {}
    if config is not None:
        raw_mapping = getattr(config, "id2label", {}) or {}
        for key, value in raw_mapping.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                try:
                    idx = int(str(value).split("_")[-1])
                except (TypeError, ValueError):
                    continue
            id2label_conf[idx] = value

    def label_name(label_idx):
        if isinstance(label_idx, str):
            if label_idx in id2label_conf.values():
                return label_idx
            try:
                numeric = int(label_idx)
            except ValueError:
                return label_idx
            return id2label_conf.get(numeric, f"LABEL_{numeric}")
        if isinstance(label_idx, int):
            return id2label_conf.get(label_idx, f"LABEL_{label_idx}")
        raise TypeError(f"Unsupported label type: {type(label_idx)}")

    ranks = np.full(len(anonymized_texts), -1, dtype=np.int32)
    total_labels = len(id2label_conf)
    if not total_labels:
        label2id_conf = getattr(getattr(tri_pipeline.model, "config", None), "label2id", {}) or {}
        total_labels = len(label2id_conf)
    for pred_idx, doc_idx in enumerate(eval_indices):
        priv_pred = predictions[pred_idx]
        priv_outputs = priv_pred if isinstance(priv_pred, list) else [priv_pred]
        if not priv_outputs:
            continue

        target_label = label_name(target_labels[doc_idx])
        rank_lookup = {entry["label"]: position + 1 for position, entry in enumerate(priv_outputs)}
        default_rank = total_labels + 1 if total_labels else len(priv_outputs) + 1
        ranks[doc_idx] = rank_lookup.get(target_label, default_rank)

    for idx in empty_indices:
        logger.info("Skipping empty anonymized text at index %d", idx)

    return ranks

if __name__ == "__main__":
    sys.exit(main())
