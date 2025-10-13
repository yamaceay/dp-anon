"""Benchmark multiple DP mechanisms on a single dataset/epsilon."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from dpmlm.config import DPMLMConfig
from dpmlm.config_utils import (
    DPMLM_GENERIC_KEYS,
    DPMLM_RUNTIME_KEYS,
    coerce_dpmlm_config,
    prepare_dpmlm_model_config,
)
from dpmlm.core import DPMLMMechanism
from dpmlm.factory import DPMechanismFactory
from dpmlm.interfaces import PrivacyResult
from loaders import DatasetAdapter, get_adapter
from petre import privatize_all

try:
    import torch
except Exception:  # pragma: no cover - torch optional at runtime
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

DPMLM_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("uniform", "dpmlm_uniform"),
    ("greedy", "dpmlm_greedy"),
    ("shap", "dpmlm_shap"),
)
OTHER_METHODS = ("dpbart", "dpparaphrase", "dpprompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple anonymization mechanisms on a dataset for a single epsilon."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (trustpilot, tab, db_bio).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Privacy epsilon value.",
    )
    parser.add_argument(
        "--dpmlm-config",
        required=True,
        help="Path to JSON config used for DPMLM variants.",
    )
    parser.add_argument(
        "--dpbart-config",
        default=None,
        help="Optional JSON config for DPBART.",
    )
    parser.add_argument(
        "--dpprompt-config",
        default=None,
        help="Optional JSON config for DPPrompt.",
    )
    parser.add_argument(
        "--dpparaphrase-config",
        default=None,
        help="Optional JSON config for DPParaphrase.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override dataset root/path.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (where applicable).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of records processed.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/method_comparison",
        help="Directory to write JSONL results.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device preference passed to DPMLM (auto, cpu, cuda, mps).",
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
    return parser.parse_args()


def initialise_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_json_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_section(data: Optional[Dict], allowed: Iterable[str]) -> Dict:
    return {key: value for key, value in (data or {}).items() if key in allowed}


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


def build_dpmlm_mechanisms(
    raw_config: Dict,
    device: str,
    seed: Optional[int],
    verbose: bool,
) -> Dict[str, Optional[DPMLMMechanism]]:
    structured = coerce_dpmlm_config(raw_config)
    generic_cfg = filter_section(structured.get("generic", {}), DPMLM_GENERIC_KEYS)
    generic_cfg.setdefault("device", device)
    if seed is not None:
        generic_cfg["seed"] = seed
    if verbose:
        generic_cfg["verbose"] = True

    base_model_payload = prepare_dpmlm_model_config(structured.get("model", {}), generic_cfg)
    runtime_payload = filter_section(structured.get("runtime", {}), DPMLM_RUNTIME_KEYS)

    mechanisms: Dict[str, Optional[DPMLMMechanism]] = {}
    for mode, label in DPMLM_VARIANTS:
        model_payload = dict(base_model_payload)
        model_payload["explainability_mode"] = mode
        try:
            config = DPMLMConfig(
                generic=copy.deepcopy(generic_cfg),
                model=model_payload,
                runtime=copy.deepcopy(runtime_payload),
            )
            mechanisms[label] = DPMLMMechanism(config=config)
            logger.info("Initialised DPMLM variant '%s'.", label)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to initialise DPMLM variant '%s': %s", label, exc)
            mechanisms[label] = None
    return mechanisms


def build_other_mechanisms(config_paths: Dict[str, Optional[str]]) -> Dict[str, Optional[object]]:
    factory = DPMechanismFactory()
    mechanisms: Dict[str, Optional[object]] = {}
    for method in OTHER_METHODS:
        config = load_json_config(config_paths.get(method))
        try:
            mech = factory.create_mechanism(method, config=config if config else None)
            mechanisms[method] = mech
            logger.info("Initialised mechanism '%s'.", method)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to initialise mechanism '%s': %s", method, exc)
            mechanisms[method] = None
    return mechanisms


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    initialise_seeds(args.seed)

    adapter = build_adapter(args.dataset, args)
    logger.info("Loaded dataset adapter (%s records).", len(adapter))

    dpmlm_config = load_json_config(args.dpmlm_config)
    dpmlm_mechanisms = build_dpmlm_mechanisms(
        raw_config=dpmlm_config,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )

    other_mechanisms = build_other_mechanisms(
        {
            "dpbart": args.dpbart_config,
            "dpparaphrase": args.dpparaphrase_config,
            "dpprompt": args.dpprompt_config,
        }
    )

    dataset_records = list(adapter.iter_records())
    output_rows: List[Dict[str, str]] = []
    epsilon = float(args.epsilon)

    for record in dataset_records:
        result = {
            "uid": str(record.uid),
            "original": record.text,
        }

        for label, mechanism in dpmlm_mechanisms.items():
            if mechanism is None:
                result[label] = ""
                continue
            try:
                privacy_result = mechanism.privatize(record.text, epsilon=epsilon)
                result[label] = privacy_result.private_text
            except Exception as exc:  # pragma: no cover
                logger.exception("DPMLM variant '%s' failed on %s: %s", label, record.uid, exc)
                result[label] = ""

        for method, mechanism in other_mechanisms.items():
            if mechanism is None:
                result[method] = ""
                continue
            try:
                privacy_result = mechanism.privatize(record.text, epsilon=epsilon)
                private_text = getattr(privacy_result, "private_text", "")
                result[method] = private_text
            except Exception as exc:  # pragma: no cover
                logger.exception("Mechanism '%s' failed on %s: %s", method, record.uid, exc)
                result[method] = ""

        output_rows.append(result)

    petre_pipeline = None
    for mechanism in dpmlm_mechanisms.values():
        if mechanism and getattr(mechanism.config.model, "risk_pipeline", None) is not None:
            petre_pipeline = mechanism.config.model.risk_pipeline
            break

    if petre_pipeline is not None:
        petre_results = privatize_all(
            dataset_records,
            petre_pipeline,
            dataset_name=args.dataset,
        )
        petre_map: Dict[str, Optional[PrivacyResult]] = {}
        for res in petre_results:
            uid = None
            if res.metadata:
                uid = res.metadata.get("uid")
            if uid is None:
                continue
            petre_map[str(uid)] = res
    else:
        logger.warning("No risk pipeline available; PETRE outputs will be empty.")
        petre_map = {str(record.uid): None for record in dataset_records}

    for row in output_rows:
        petre_result = petre_map.get(row["uid"])
        row["petre"] = petre_result.private_text if petre_result else ""

    output_path = Path(args.output_dir) / args.dataset.lower()
    if args.split:
        output_path = output_path / args.split
    output_file = output_path / f"epsilon_{epsilon}.jsonl"
    write_jsonl(output_file, output_rows)
    logger.info("Wrote %d records to %s", len(output_rows), output_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
