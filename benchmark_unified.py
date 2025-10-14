#!/usr/bin/env python3
"""Unified benchmarking harness covering annotations and anonymisation methods."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from annotation_utils import NormalisedAnnotation, apply_annotations
from annotate import StandaloneAnnotationGenerator
from benchmark import build_dpmlm_config  # reuse validated config builder
from dpmlm.multi import MultiEpsilonDPMLM
from loaders import DatasetAdapter, get_adapter
from loaders.base import DatasetRecord
from presets import get_petre_preset

logger = logging.getLogger("benchmark_unified")


DEFAULT_OUTPUT_ROOT = Path("outputs/unified_benchmark")
DEFAULT_TAB_TRAIN = Path("data/TAB/splitted/train.json")
DEFAULT_PETRE_KS = (2, 3, 5, 7, 10)
DEFAULT_METHODS = [
    "manual",
    "spacy",
    "presidio",
    "dpmlm_greedy_p095",
    "dpmlm_greedy_p098",
    "dpmlm_shap_p095",
    "dpmlm_shap_p098",
    "dpbart",
    "dpparaphrase",
    "dpprompt",
    "petre_k2",
    "petre_k3",
    "petre_k5",
    "petre_k7",
    "petre_k10",
]


class MethodUnavailable(RuntimeError):
    """Raised when a method cannot run due to missing dependencies."""


@dataclass
class ExecutionContext:
    """Shared context for method runners."""

    epsilon: float
    dataset: str
    dataset_path: Optional[str]
    split: Optional[str]
    device: str
    seed: Optional[int]
    preview_limit: int
    petre_ks: Sequence[int]
    output_dir: Path
    verbose: bool = False
    petre_starting_annotations: Optional[str] = None

    _annotation_generator: Optional[StandaloneAnnotationGenerator] = field(default=None, init=False, repr=False)
    _dpmlm_cache: Dict[str, MultiEpsilonDPMLM] = field(default_factory=dict, init=False, repr=False)
    _petre_annotations_path: Optional[Path] = field(default=None, init=False, repr=False)

    def get_annotation_generator(self) -> StandaloneAnnotationGenerator:
        if self._annotation_generator is None:
            logger.debug("Initialising annotation generator")
            self._annotation_generator = StandaloneAnnotationGenerator()
        return self._annotation_generator

    def get_dpmlm_mechanism(self, cache_key: str, config_payload: Dict[str, Any]) -> MultiEpsilonDPMLM:
        if cache_key not in self._dpmlm_cache:
            dp_config = build_dpmlm_config(
                config_payload,
                device=self.device,
                seed=self.seed,
                verbose=self.verbose,
            )
            self._dpmlm_cache[cache_key] = MultiEpsilonDPMLM(dp_config)
        return self._dpmlm_cache[cache_key]

    def get_petre_annotations_path(self, records: Sequence[DatasetRecord]) -> Path:
        if self.petre_starting_annotations:
            candidate = Path(self.petre_starting_annotations)
            if candidate.exists():
                return candidate
            logger.warning(
                "PETRE starting annotations %s not found; generating from dataset.",
                candidate,
            )

        if self._petre_annotations_path and self._petre_annotations_path.exists():
            return self._petre_annotations_path

        annotations_map: Dict[str, List[List[int]]] = {}
        for record in records:
            spans: List[List[int]] = []
            raw_annotations = record.annotations or {}
            if isinstance(raw_annotations, dict):
                for annot_data in raw_annotations.values():
                    entity_mentions = annot_data.get("entity_mentions", [])
                    for mention in entity_mentions:
                        start = mention.get("start_offset")
                        end = mention.get("end_offset")
                        if start is None or end is None:
                            continue
                        try:
                            start_i = int(start)
                            end_i = int(end)
                        except (TypeError, ValueError):
                            continue
                        if start_i < 0 or end_i <= start_i:
                            continue
                        spans.append([start_i, end_i])
            elif isinstance(raw_annotations, list):
                for mention in raw_annotations:
                    if isinstance(mention, dict):
                        start = mention.get("start") or mention.get("start_offset")
                        end = mention.get("end") or mention.get("end_offset")
                    elif isinstance(mention, (list, tuple)) and len(mention) >= 2:
                        start, end = mention[0], mention[1]
                    else:
                        continue
                    try:
                        start_i = int(start)
                        end_i = int(end)
                    except (TypeError, ValueError):
                        continue
                    if start_i < 0 or end_i <= start_i:
                        continue
                    spans.append([start_i, end_i])
            spans.sort(key=lambda item: item[0])
            if spans:
                annotations_map[str(record.uid)] = spans

        if not annotations_map:
            raise FileNotFoundError(
                "Unable to build PETRE starting annotations automatically. "
                "Provide --petre-starting-annotations pointing to a manual annotation file."
            )

        annotation_dir = self.output_dir / "petre" / self.dataset.lower()
        if self.split:
            annotation_dir = annotation_dir / self.split.lower()
        annotation_dir.mkdir(parents=True, exist_ok=True)
        output_path = annotation_dir / "starting_annotations.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(annotations_map, handle, ensure_ascii=False, indent=2)

        self._petre_annotations_path = output_path
        logger.info("Generated PETRE starting annotations at %s", output_path)
        return output_path


@dataclass
class MethodExecutionResult:
    status: str
    payload_by_record: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


MethodRunner = Callable[[Sequence[DatasetRecord], ExecutionContext], MethodExecutionResult]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a unified anonymisation benchmark across multiple mechanisms.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (trustpilot, tab, db_bio).")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override dataset root/path (defaults to TAB train split for dataset=tab).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split identifier where applicable.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=25.0,
        help="Differential privacy epsilon to use for supported mechanisms.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit processing to the first N records.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Subset of methods to execute. Use 'list' to show available methods.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device identifier for DP mechanisms (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to pass to DPMLM where applicable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory in which benchmark artefacts should be written.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=3,
        help="Number of records to preview in terminal output (0 to disable).",
    )
    parser.add_argument(
        "--petre-ks",
        default="2,3,5,7,10",
        help="Comma separated k values for PETRE.",
    )
    parser.add_argument(
        "--petre-starting-annotations",
        default=None,
        help="Path to manual annotations JSON for PETRE starting point (optional).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)
    if len(args.methods) == 1 and args.methods[0].lower() == "list":
        print("Available methods:")
        for method in sorted(DEFAULT_METHODS):
            print(f"  - {method}")
        raise SystemExit(0)
    if args.device == "auto":
        args.device = (
            "mps" if torch.backends.mps.is_available() else 
            "cuda" if torch.cuda.is_available() else 
            "cpu"
        )
    return args


def prepare_dataset_adapter(args: argparse.Namespace) -> DatasetAdapter:
    kwargs: Dict[str, Any] = {"max_records": args.max_records}
    dataset_lower = args.dataset.lower()
    if dataset_lower == "tab":
        dataset_path = args.dataset_path or str(DEFAULT_TAB_TRAIN)
        kwargs["data_path"] = dataset_path
        args.dataset_path = dataset_path
    elif dataset_lower in {"trustpilot"} and args.dataset_path:
        kwargs["data_path"] = args.dataset_path
    elif dataset_lower in {"db_bio", "db-bio"}:
        if args.dataset_path:
            kwargs["root"] = args.dataset_path
        kwargs["split"] = args.split
    adapter = get_adapter(args.dataset, **kwargs)
    logger.info("Loaded adapter for dataset %s (%s records).", args.dataset, len(adapter))
    return adapter


def method_manual(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    generator = ctx.get_annotation_generator()
    annotations, stats = generator.generate_manual_annotations(records)
    payload_by_record = {
        uid: {
            "annotations": [
                NormalisedAnnotation.from_mapping(span).to_dict()
                for span in (spans or [])
            ]
        }
        for uid, spans in annotations.items()
    }
    return MethodExecutionResult(status="ok", payload_by_record=payload_by_record, metadata=stats)


def method_spacy(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    generator = ctx.get_annotation_generator()
    available = generator.get_available_methods().get("spacy", {})
    if not available.get("available"):
        raise MethodUnavailable(available.get("error", "spaCy pipeline unavailable"))
    annotations, stats = generator.generate_spacy_annotations(records)
    payload_by_record = {
        uid: {
            "annotations": [
                NormalisedAnnotation.from_mapping(span).to_dict()
                for span in (spans or [])
            ]
        }
        for uid, spans in annotations.items()
    }
    return MethodExecutionResult(status="ok", payload_by_record=payload_by_record, metadata=stats)


def method_presidio(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    generator = ctx.get_annotation_generator()
    available = generator.get_available_methods().get("presidio", {})
    if not available.get("available"):
        raise MethodUnavailable(available.get("error", "Presidio pipeline unavailable"))
    annotations, stats = generator.generate_presidio_annotations(records)
    payload_by_record = {
        uid: {
            "annotations": [
                NormalisedAnnotation.from_mapping(span).to_dict()
                for span in (spans or [])
            ]
        }
        for uid, spans in annotations.items()
    }
    return MethodExecutionResult(status="ok", payload_by_record=payload_by_record, metadata=stats)


def _run_dp_llm_mechanism(
    factory: Callable[[], Any],
    records: Sequence[DatasetRecord],
    ctx: ExecutionContext,
    description: str,
) -> MethodExecutionResult:
    try:
        mechanism = factory()
    except ImportError as exc:  # pragma: no cover - optional deps
        raise MethodUnavailable(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to initialise %s mechanism", description)
        return MethodExecutionResult(status="error", error=str(exc))

    payload: Dict[str, Dict[str, Any]] = {}
    progress = tqdm(
        records,
        desc=f"{description} ({len(records)} records)",
        leave=False,
        dynamic_ncols=True,
    )
    for record in progress:
        try:
            result = mechanism.privatize(record.text, epsilon=ctx.epsilon)
        except Exception as exc:  # pragma: no cover - mechanism failure
            payload[str(record.uid)] = {"error": str(exc)}
            continue
        payload[str(record.uid)] = {
            "anonymized_text": {
                "original": record.text,
                "anonymized": result.private_text,
            },
            "metadata": result.metadata or {},
        }
    return MethodExecutionResult(status="ok", payload_by_record=payload)


def method_dpprompt(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    def factory():
        from dpmlm.llmdp import DPPromptPrivatizer

        return DPPromptPrivatizer(device=ctx.device)

    return _run_dp_llm_mechanism(factory, records, ctx, "DPPrompt")


def method_dpparaphrase(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    def factory():
        from dpmlm.llmdp import DPParaphrasePrivatizer

        return DPParaphrasePrivatizer(device=ctx.device)

    return _run_dp_llm_mechanism(factory, records, ctx, "DPParaphrase")


def method_dpbart(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    def factory():
        from dpmlm.llmdp import DPBartPrivatizer

        return DPBartPrivatizer(device=ctx.device)

    return _run_dp_llm_mechanism(factory, records, ctx, "DPBart")


def make_dpmlm_runner(name: str, explainability: str, pii_threshold: float) -> MethodRunner:
    risk_model_map = {
        "tab": "models/tri_pipelines/TAB/TRI_Pipeline",
        "trustpilot": "models/tri_pipelines/trustpilot/www.amazon.com/TRI_Pipeline",
    }
    annotator_map = {
        "tab": "models/pii_detectors/pii_model_20251004_184431",
        "trustpilot": "models/pii_detectors/pii_model_20251004_184431",
    }

    def runner(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
        dataset_key = ctx.dataset.lower()
        risk_model = risk_model_map.get(dataset_key)
        annotator_path = annotator_map.get(dataset_key)
        if not risk_model or not annotator_path:
            raise MethodUnavailable(f"No DPMLM resources configured for dataset '{ctx.dataset}'.")

        config_payload = {
            "model": {
                "explainability_mode": explainability,
                "risk_model": risk_model,
                "annotator_path": annotator_path,
                "pii_threshold": pii_threshold,
            }
        }
        mechanism = ctx.get_dpmlm_mechanism(
            cache_key=f"{name}:{ctx.dataset}:{ctx.device}:{pii_threshold}",
            config_payload=config_payload,
        )
        payload: Dict[str, Dict[str, Any]] = {}
        progress = tqdm(
            records,
            desc=f"DPMLM {name} ({len(records)} records)",
            leave=False,
            dynamic_ncols=True,
        )
        for record in progress:
            try:
                privacy_results = mechanism.privatize_many(record.text, epsilons=[ctx.epsilon])
                result = next(iter(privacy_results.values()))
            except Exception as exc:
                payload[str(record.uid)] = {"error": str(exc)}
                continue
            result_map = {
                "anonymized_text": {
                    "original": record.text,
                    "anonymized": result.private_text,
                },
                "metadata": result.metadata or {},
            }
            payload[str(record.uid)] = result_map
        metadata = {
            "explainability_mode": explainability,
            "pii_threshold": pii_threshold,
        }
        return MethodExecutionResult(status="ok", payload_by_record=payload, metadata=metadata)

    return runner


def make_petre_runner(k: int) -> MethodRunner:
    def runner(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional deps
            raise MethodUnavailable("PyTorch is required for PETRE") from exc

        preset = get_petre_preset(ctx.dataset)
        if ctx.dataset.lower() == "tab" and ctx.dataset_path:
            preset["data_file_path"] = ctx.dataset_path
        preset["ks"] = list(ctx.petre_ks)
        preset["mask_text"] = "[MASK]"
        try:
            starting_annotations_path = ctx.get_petre_annotations_path(records)
            preset["starting_anonymization_path"] = str(starting_annotations_path)
        except FileNotFoundError as exc:
            raise MethodUnavailable(str(exc)) from exc

        try:
            from petre import PETRE
        except ImportError as exc:  # pragma: no cover - optional deps
            raise MethodUnavailable("PETRE module not available") from exc

        petre_model = PETRE(**preset)
        petre_model.initialization(verbose=ctx.verbose)
        payload: Dict[str, Dict[str, Any]] = {}
        metadata: Dict[str, Any] = {"k": k}

        snapshot = petre_model.petre(k, verbose=ctx.verbose)
        annotations_map = petre_model.dataset.get_annotations()

        for uid, record_snapshot in snapshot.items():
            spans = annotations_map.get(uid, [])
            span_payload = [
                NormalisedAnnotation(start=int(start), end=int(end), label="MASK", replacement="[MASK]").to_dict()
                for start, end in spans
            ]
            anonymized_payload = {
                "original": record_snapshot.get("original"),
                "anonymized": record_snapshot.get("anonymized"),
            }
            payload[str(uid)] = {
                "annotations": span_payload,
                "anonymized_text": anonymized_payload,
                "perturbed_tokens": record_snapshot.get("perturbed_tokens"),
            }

        return MethodExecutionResult(status="ok", payload_by_record=payload, metadata=metadata)

    return runner


METHOD_REGISTRY: Dict[str, Tuple[str, MethodRunner]] = {
    "manual": ("annotations", method_manual),
    "spacy": ("annotations", method_spacy),
    "presidio": ("annotations", method_presidio),
    "dpbart": ("anonymized_text", method_dpbart),
    "dpparaphrase": ("anonymized_text", method_dpparaphrase),
    "dpprompt": ("anonymized_text", method_dpprompt),
    "dpmlm_greedy_p095": ("anonymized_text", make_dpmlm_runner("dpmlm_greedy_p095", "greedy", 0.95)),
    "dpmlm_greedy_p098": ("anonymized_text", make_dpmlm_runner("dpmlm_greedy_p098", "greedy", 0.98)),
    "dpmlm_shap_p095": ("anonymized_text", make_dpmlm_runner("dpmlm_shap_p095", "shap", 0.95)),
    "dpmlm_shap_p098": ("anonymized_text", make_dpmlm_runner("dpmlm_shap_p098", "shap", 0.98)),
    "petre_k2": ("annotations", make_petre_runner(2)),
    "petre_k3": ("annotations", make_petre_runner(3)),
    "petre_k5": ("annotations", make_petre_runner(5)),
    "petre_k7": ("annotations", make_petre_runner(7)),
    "petre_k10": ("annotations", make_petre_runner(10)),
}


def run_methods(
    records: Sequence[DatasetRecord],
    ctx: ExecutionContext,
    method_names: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    results_by_record: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    summary_by_method: Dict[str, Dict[str, Any]] = {}

    for method_name in method_names:
        method_key = method_name.lower()
        if method_key not in METHOD_REGISTRY:
            logger.warning("Unknown method '%s'; skipping.", method_name)
            summary_by_method[method_name] = {"status": "unknown"}
            continue

        kind, runner = METHOD_REGISTRY[method_key]
        logger.info("Running method %s (%s)", method_name, kind)
        try:
            outcome = runner(records, ctx)
        except MethodUnavailable as exc:
            logger.warning("Method '%s' unavailable: %s", method_name, exc)
            summary_by_method[method_name] = {"status": "unavailable", "reason": str(exc), "kind": kind}
            continue
        except Exception as exc:
            logger.exception("Method '%s' failed", method_name)
            summary_by_method[method_name] = {"status": "error", "reason": str(exc), "kind": kind}
            continue

        summary = {"status": outcome.status, "kind": kind}
        if outcome.metadata:
            summary["metadata"] = outcome.metadata
        if outcome.error:
            summary["reason"] = outcome.error
        summary_by_method[method_name] = summary

        if outcome.status != "ok":
            continue

        for uid, payload in outcome.payload_by_record.items():
            results_by_record[uid][method_name] = payload

    return results_by_record, summary_by_method


def build_output_payload(
    records: Sequence[DatasetRecord],
    results_by_record: Dict[str, Dict[str, Dict[str, Any]]],
    method_summary: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_records: List[Dict[str, Any]] = []
    for record in records:
        uid = str(record.uid)
        output_records.append(
            {
                "uid": uid,
                "original": record.text,
                "name": record.name,
                "metadata": record.metadata,
                "utilities": record.utilities,
                "methods": results_by_record.get(uid, {}),
            }
        )

    payload = {
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "epsilon": args.epsilon,
        "record_count": len(records),
        "methods": method_summary,
        "records": output_records,
    }
    return payload


def write_output(payload: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Path:
    dataset_part = args.dataset.lower()
    split_part = args.split.lower() if args.split else "default"
    epsilon_str = f"eps_{args.epsilon}".replace(".", "_")
    output_path = output_dir / dataset_part / split_part / f"benchmark_{epsilon_str}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def print_preview(
    records: Sequence[DatasetRecord],
    results_by_record: Dict[str, Dict[str, Dict[str, Any]]],
    method_summary: Dict[str, Dict[str, Any]],
    limit: int,
) -> None:
    if limit <= 0:
        return
    print("\nPreview of anonymised outputs:")
    shown = 0
    for record in records:
        if shown >= limit:
            break
        uid = str(record.uid)
        method_payloads = results_by_record.get(uid)
        if not method_payloads:
            continue
        print(f"\nUID: {uid}")
        print(f"Original: {record.text[:200]}{'...' if len(record.text) > 200 else ''}")
        for method_name, payload in method_payloads.items():
            summary = method_summary.get(method_name, {})
            status = summary.get("status", "unknown")
            print(f"  [{method_name}] status={status}")
            if "anonymized_text" in payload:
                anonymized_payload = payload["anonymized_text"]
                if isinstance(anonymized_payload, dict):
                    anonymized_text = anonymized_payload.get("anonymized") or ""
                else:
                    anonymized_text = str(anonymized_payload)
                if anonymized_text:
                    print(f"    Anonymized: {anonymized_text[:200]}{'...' if len(anonymized_text) > 200 else ''}")
            elif "annotations" in payload:
                preview_text = apply_annotations(record.text, payload["annotations"])
                print(f"    Applied annotations: {preview_text[:200]}{'...' if len(preview_text) > 200 else ''}")
        shown += 1
    if shown == 0:
        print("No method outputs available for preview.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    adapter = prepare_dataset_adapter(args)
    records = list(adapter.iter_records())
    if not records:
        logger.error("No records loaded from dataset.")
        return 1

    petre_ks = tuple(sorted({int(value) for value in args.petre_ks.split(",") if value.strip()}))
    ctx = ExecutionContext(
        epsilon=args.epsilon,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        device=args.device,
        seed=args.seed,
        preview_limit=args.preview_limit,
        petre_ks=petre_ks or DEFAULT_PETRE_KS,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
        petre_starting_annotations=args.petre_starting_annotations,
    )

    results_by_record, method_summary = run_methods(records, ctx, args.methods)
    payload = build_output_payload(records, results_by_record, method_summary, args)
    output_path = write_output(payload, args, ctx.output_dir)
    logger.info("Benchmark results written to %s", output_path)
    print_preview(records, results_by_record, method_summary, args.preview_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
