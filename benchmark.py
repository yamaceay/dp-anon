"""Generic benchmarking script using dataset adapters and DPMLM."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import numpy as np

from dpmlm.config import DPMLMConfig
from dpmlm.config_utils import (
    DPMLM_GENERIC_KEYS,
    DPMLM_RUNTIME_KEYS,
    coerce_dpmlm_config,
    prepare_dpmlm_model_config,
)
from loaders import DatasetAdapter, get_adapter
from dpmlm.multi import MultiEpsilonDPMLM

try:
    import torch
except Exception:  # pragma: no cover - torch optional for CPU runs
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


def parse_epsilons(value: str) -> List[float]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Epsilon list must contain at least one value.")
    return [float(token) for token in tokens]


def load_config_file(path: Optional[str]) -> Dict:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_section(data: Dict, allowed: Iterable[str]) -> Dict:
    return {key: value for key, value in (data or {}).items() if key in allowed}


def build_dpmlm_config(
    raw_config: Dict,
    *,
    device: str,
    seed: Optional[int],
    verbose: bool,
) -> DPMLMConfig:
    structured = coerce_dpmlm_config(raw_config)
    generic_cfg = _filter_section(structured.get("generic", {}), DPMLM_GENERIC_KEYS)
    generic_cfg.setdefault("device", device)
    if seed is not None:
        generic_cfg["seed"] = seed
    if verbose:
        generic_cfg["verbose"] = True

    model_payload = prepare_dpmlm_model_config(
        structured.get("model", {}),
        generic_cfg,
    )
    runtime_payload = _filter_section(structured.get("runtime", {}), DPMLM_RUNTIME_KEYS)

    return DPMLMConfig(
        generic=generic_cfg,
        model=model_payload,
        runtime=runtime_payload,
    )


def initialise_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, items: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def build_adapter(dataset: str, args: argparse.Namespace) -> DatasetAdapter:
    kwargs: Dict[str, object] = {"max_records": args.max_records}
    key = dataset.lower()
    if key == "trustpilot":
        if args.dataset_path:
            kwargs["data_path"] = args.dataset_path
    elif key == "tab":
        if args.dataset_path:
            kwargs["data_path"] = args.dataset_path
    elif key in {"db_bio", "db-bio"}:
        if args.dataset_path:
            kwargs["root"] = args.dataset_path
        kwargs["split"] = args.split
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")
    return get_adapter(dataset, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic DPMLM benchmarking across datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (trustpilot, tab, db_bio).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override default dataset path/root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (if applicable).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit processing to the first N records.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON configuration file for DPMLM.",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        required=True,
        help="Comma-separated epsilon values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Directory for anonymised outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device preference (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--petre-ks",
        type=str,
        default="2,3,5,7,10",
        help="Comma-separated k values for PETRE variants.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    initialise_seeds(args.seed)

    epsilons = parse_epsilons(args.epsilons)
    logger.info("Running benchmark on dataset=%s with epsilons=%s", args.dataset, epsilons)

    petre_ks = [int(token.strip()) for token in args.petre_ks.split(",") if token.strip()]
    petre_ks.sort()

    adapter = build_adapter(args.dataset, args)
    logger.info("Loaded dataset adapter (%s records).", len(adapter))

    raw_config = load_config_file(args.config)
    dp_config = build_dpmlm_config(
        raw_config,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )

    mechanism = MultiEpsilonDPMLM(dp_config)
    output_root = Path(args.output_dir) / args.dataset.lower()
    if args.split:
        output_root = output_root / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    per_epsilon_outputs: Dict[float, List[Dict]] = {eps: [] for eps in epsilons}

    for record in adapter.iter_records():
        try:
            privacy_results = mechanism.privatize_many(record.text, epsilons=epsilons)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to privatize record %s: %s", record.uid, exc)
            for eps in epsilons:
                per_epsilon_outputs[eps].append(
                    {
                        "uid": record.uid,
                        "original": record.text,
                        "anonymized": "",
                        "name": record.name,
                        "annotations": record.annotations,
                        "utilities": record.utilities,
                        "metadata": record.metadata,
                        "error": str(exc),
                    }
                )
            continue

        for eps, privacy_result in privacy_results.items():
            per_epsilon_outputs[eps].append(
                {
                    "uid": record.uid,
                    "original": record.text,
                    "anonymized": privacy_result.private_text,
                    "name": record.name,
                    "annotations": record.annotations,
                    "utilities": record.utilities,
                    "metadata": record.metadata,
                    "privacy_metadata": privacy_result.metadata,
                }
            )

    for eps, rows in per_epsilon_outputs.items():
        output_path = output_root / f"results_{eps}.jsonl"
        write_jsonl(output_path, rows)
        logger.info("Wrote %d records to %s", len(rows), output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
