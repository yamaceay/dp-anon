#!/usr/bin/env python3
"""Unified benchmarking harness covering annotations and anonymisation methods."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from annotation_utils import NormalisedAnnotation, apply_annotations
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


@dataclass
class DatasetRecord:
    uid: str
    text: str
    name: str = ""
    annotations: Optional[Any] = None
    utilities: Dict[str, Optional[Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TabDatasetAdapter:
    def __init__(self, data_path: Optional[str] = None, max_records: Optional[int] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_TAB_TRAIN
        self.max_records = max_records
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                self._records: List[Dict[str, Any]] = json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TAB dataset from {self.data_path}") from exc

    def __len__(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._records):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("doc_id", idx))
            text = row.get("text", "")
            annotations = row.get("annotations")
            utilities = {
                "country": row.get("meta", {}).get("countries"),
                "years": row.get("meta", {}).get("years"),
            }
            name = row.get("meta", {}).get("applicant", "")
            metadata = {
                "quality_checked": row.get("quality_checked"),
                "task": row.get("task"),
                "dataset_type": row.get("dataset_type"),
                "meta": row.get("meta"),
            }

            yield DatasetRecord(
                uid=uid,
                text=text,
                name=name,
                annotations=annotations,
                utilities=utilities,
                metadata=metadata,
            )


def sanitize_identifier(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(ch if ch in allowed else "_" for ch in value)


def format_epsilon(epsilon: float) -> str:
    if isinstance(epsilon, int) or (isinstance(epsilon, float) and epsilon.is_integer()):
        return str(int(epsilon))
    return str(epsilon).replace("-", "neg").replace(".", "_")


def method_storage_name(method: str, kind: str, ctx: "ExecutionContext") -> str:
    base = method
    if kind != "annotations" and "eps" not in method.lower():
        base = f"{base}_eps{format_epsilon(ctx.epsilon)}"
    return sanitize_identifier(base)


def method_category(method: str, kind: str) -> str:
    lowered = method.lower()
    if lowered in {"manual", "spacy", "presidio"}:
        return "simple"
    if lowered.startswith("petre") or lowered.startswith("tri"):
        return "k_anonymity"
    if kind != "annotations":
        return "differential_privacy"
    return "differential_privacy"


def coalesce_annotations(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            continue
        span_copy = dict(span)
        span_copy["start"] = start_int
        span_copy["end"] = end_int
        prepared.append(span_copy)

    prepared.sort(key=lambda item: (item.get("start", 0), -(item.get("end", 0) - item.get("start", 0)), -item.get("end", 0)))

    result: List[Dict[str, Any]] = []
    seen_start: Optional[int] = None
    seen_pairs = set()
    for span in prepared:
        start = span["start"]
        end = span["end"]
        key = (start, end)
        if key in seen_pairs:
            continue
        if seen_start is not None and start == seen_start:
            continue
        result.append(span)
        seen_start = start
        seen_pairs.add(key)

    result.sort(key=lambda item: item.get("start", 0))
    return result


def load_existing_results(summary_path: Path, records_path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not summary_path.exists() or not records_path.exists():
        return {}, {}, {}

    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        summary_payload = {}

    record_info_map: Dict[str, Dict[str, Any]] = {}
    record_methods_map: Dict[str, Dict[str, Any]] = {}

    try:
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                uid = str(entry.get("uid"))
                if not uid:
                    continue
                record_info_map[uid] = {
                    key: entry.get(key)
                    for key in ("uid", "original", "name", "metadata", "utilities")
                    if key in entry
                }
                record_methods_map[uid] = entry.get("methods", {})
    except (OSError, json.JSONDecodeError):
        record_info_map = {}
        record_methods_map = {}

    return summary_payload, record_info_map, record_methods_map


def method_output_path(ctx: "ExecutionContext", method_name: str, kind: str) -> Path:
    category = method_category(method_name, kind)
    subdir = ctx.output_dir / "annotations" / category
    subdir.mkdir(parents=True, exist_ok=True)
    filename = f"{method_storage_name(method_name, kind, ctx)}.json"
    return subdir / filename


def relative_to_output(path: Path, ctx: "ExecutionContext") -> str:
    try:
        return os.path.relpath(path, ctx.output_dir)
    except ValueError:
        return str(path)


def ensure_record_entries(
    records: Sequence[DatasetRecord],
    record_info_map: Dict[str, Dict[str, Any]],
    record_methods_map: Dict[str, Dict[str, Any]],
) -> None:
    for record in records:
        uid = str(record.uid)
        info = record_info_map.setdefault(uid, {})
        info.update(
            {
                "uid": uid,
                "original": record.text,
                "name": record.name,
                "metadata": record.metadata,
                "utilities": record.utilities,
            }
        )
        record_methods_map.setdefault(uid, {})


def resolve_device(preference: str) -> str:
    if preference == "auto":
        return "cpu"
    return preference


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
    run_id: str = ""

    _annotation_generator: Optional[Any] = field(default=None, init=False, repr=False)
    _dpmlm_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _petre_annotations_path: Optional[Path] = field(default=None, init=False, repr=False)

    def get_annotation_generator(self):
        if self._annotation_generator is None:
            logger.debug("Initialising annotation generator")
            try:
                from annotate import StandaloneAnnotationGenerator  # type: ignore
            except Exception as exc:  # pragma: no cover - optional deps
                raise MethodUnavailable(f"Annotation generator unavailable: {exc}") from exc

            self._annotation_generator = StandaloneAnnotationGenerator()
        return self._annotation_generator

    def get_dpmlm_mechanism(self, cache_key: str, config_payload: Dict[str, Any]):
        if cache_key not in self._dpmlm_cache:
            try:
                from benchmark import build_dpmlm_config  # type: ignore
                from dpmlm.multi import MultiEpsilonDPMLM  # type: ignore
            except Exception as exc:  # pragma: no cover - optional deps
                raise MethodUnavailable(f"DPMLM dependencies unavailable: {exc}") from exc

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
        "--force",
        action="store_true",
        help="Recompute methods even if cached outputs exist.",
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
    return args


def prepare_dataset_adapter(args: argparse.Namespace):
    dataset_lower = args.dataset.lower()
    kwargs: Dict[str, Any] = {"max_records": args.max_records}

    if dataset_lower == "tab":
        dataset_path = args.dataset_path or str(DEFAULT_TAB_TRAIN)
        adapter = TabDatasetAdapter(data_path=dataset_path, max_records=args.max_records)
        args.dataset_path = dataset_path
    elif dataset_lower == "trustpilot":
        try:
            from loaders.trustpilot import TrustpilotDatasetAdapter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise MethodUnavailable(f"Failed to load Trustpilot adapter: {exc}") from exc

        if args.dataset_path:
            kwargs["data_path"] = args.dataset_path
        adapter = TrustpilotDatasetAdapter(**kwargs)
    elif dataset_lower in {"db_bio", "db-bio"}:
        try:
            from loaders.db_bio import DBBioDatasetAdapter  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise MethodUnavailable(f"Failed to load DB-Bio adapter: {exc}") from exc

        if args.dataset_path:
            kwargs["root"] = args.dataset_path
        kwargs["split"] = args.split
        adapter = DBBioDatasetAdapter(**kwargs)
    else:
        raise MethodUnavailable(f"Unsupported dataset '{args.dataset}'.")

    logger.info("Loaded adapter for dataset %s (%s records).", args.dataset, len(adapter))
    return adapter


def method_manual(records: Sequence[DatasetRecord], ctx: ExecutionContext) -> MethodExecutionResult:
    payload_by_record: Dict[str, Dict[str, Any]] = {}
    documents_with_annotations = 0
    total_spans = 0

    for record in records:
        uid = str(record.uid)
        annotations_normalised: List[Dict[str, Any]] = []
        raw_annotations = record.annotations or {}
        if isinstance(raw_annotations, dict):
            for annotator_payload in raw_annotations.values():
                entity_mentions = annotator_payload.get("entity_mentions", [])
                for mention in entity_mentions:
                    try:
                        annotations_normalised.append(
                            NormalisedAnnotation.from_mapping(mention).to_dict()
                        )
                    except ValueError:
                        continue
        elif isinstance(raw_annotations, list):
            for mention in raw_annotations:
                try:
                    annotations_normalised.append(
                        NormalisedAnnotation.from_mapping(mention).to_dict()
                    )
                except ValueError:
                    continue

        annotations_normalised = coalesce_annotations(annotations_normalised)

        if annotations_normalised:
            documents_with_annotations += 1
            total_spans += len(annotations_normalised)

        payload_by_record[uid] = {"annotations": annotations_normalised}

    stats = {
        "total_documents": len(records),
        "documents_with_annotations": documents_with_annotations,
        "total_spans": total_spans,
        "average_spans_per_document": (
            total_spans / documents_with_annotations if documents_with_annotations else 0
        ),
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
                for span in coalesce_annotations(spans or [])
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
                for span in coalesce_annotations(spans or [])
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
            "stats": {
                "perturbed_tokens": getattr(result, "perturbed_tokens", None),
                "total_tokens": getattr(result, "total_tokens", None),
                "added_tokens": getattr(result, "added_tokens", None),
                "deleted_tokens": getattr(result, "deleted_tokens", None),
            },
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
                "stats": {
                    "perturbed_tokens": result.perturbed_tokens,
                    "total_tokens": result.total_tokens,
                    "added_tokens": result.added_tokens,
                    "deleted_tokens": result.deleted_tokens,
                },
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
                "stats": {
                    "perturbed_tokens": record_snapshot.get("perturbed_tokens"),
                },
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
    record_info_map: Dict[str, Dict[str, Any]],
    record_methods_map: Dict[str, Dict[str, Any]],
    existing_summary: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    summary_existing = existing_summary.get("methods", {}) if existing_summary else {}
    summary_by_method: Dict[str, Dict[str, Any]] = dict(summary_existing)

    for method_name in method_names:
        method_key = method_name.lower()
        if method_key not in METHOD_REGISTRY:
            logger.warning("Unknown method '%s'; skipping.", method_name)
            summary_by_method[method_name] = {"status": "unknown"}
            continue

        kind, runner = METHOD_REGISTRY[method_key]
        existing_entry = summary_existing.get(method_name)
        already_ok = (
            not force
            and existing_entry
            and existing_entry.get("status") == "ok"
            and all(method_name in record_methods_map.get(str(record.uid), {}) for record in records)
        )
        if already_ok:
            summary_by_method[method_name] = existing_entry
            continue

        logger.info("Running method %s (%s)", method_name, kind)
        try:
            outcome = runner(records, ctx)
        except MethodUnavailable as exc:
            logger.warning("Method '%s' unavailable: %s", method_name, exc)
            summary_entry = {"status": "unavailable", "reason": str(exc), "kind": kind}
            summary_by_method[method_name] = summary_entry
            for record in records:
                uid = str(record.uid)
                record_methods_map.setdefault(uid, {}).setdefault(method_name, {
                    "type": kind,
                    "status": "unavailable",
                    "reason": str(exc),
                })
            continue
        except Exception as exc:
            logger.exception("Method '%s' failed", method_name)
            summary_entry = {"status": "error", "reason": str(exc), "kind": kind}
            summary_by_method[method_name] = summary_entry
            for record in records:
                uid = str(record.uid)
                record_methods_map.setdefault(uid, {}).setdefault(method_name, {
                    "type": kind,
                    "status": "error",
                    "error": str(exc),
                })
            continue

        if outcome.status != "ok":
            summary_entry = {
                "status": outcome.status,
                "kind": kind,
            }
            if outcome.error:
                summary_entry["reason"] = outcome.error
            if outcome.metadata:
                summary_entry["metadata"] = outcome.metadata
            summary_by_method[method_name] = summary_entry
            continue

        method_path = method_output_path(ctx, method_name, kind)
        method_path.parent.mkdir(parents=True, exist_ok=True)
        method_rel_path = relative_to_output(method_path, ctx)

        method_data: Dict[str, Any] = {}
        error_count = 0
        success_count = 0
        annotation_docs = 0
        total_spans = 0
        perturbed_total = 0
        total_tokens_total = 0
        added_total = 0
        deleted_total = 0

        for record in records:
            uid = str(record.uid)
            payload = outcome.payload_by_record.get(uid, {})
            if not payload:
                payload = {"error": "No output produced"}
            entry = {
                "type": kind,
                "file": method_rel_path,
                "key": uid,
            }

            if "error" in payload:
                error_count += 1
                entry["status"] = "error"
                entry["error"] = str(payload["error"])
                method_data[uid] = {"error": str(payload["error"])}
                record_methods_map[uid][method_name] = entry
                continue

            entry["status"] = "ok"
            if kind == "annotations":
                spans = payload.get("annotations", [])
                method_data[uid] = spans
                entry["span_count"] = len(spans)
                if spans:
                    annotation_docs += 1
                    total_spans += len(spans)
            else:
                anonymized = payload.get("anonymized_text", {})
                stats = payload.get("stats", {}) or {}
                method_data[uid] = {
                    "original": anonymized.get("original"),
                    "anonymized": anonymized.get("anonymized"),
                    "metadata": payload.get("metadata"),
                    "stats": stats,
                }
                preview_text = (anonymized.get("anonymized") or "")
                if preview_text:
                    entry["preview"] = preview_text[:120]
                perturbed_total += stats.get("perturbed_tokens", 0) or 0
                total_tokens_total += stats.get("total_tokens", 0) or 0
                added_total += stats.get("added_tokens", 0) or 0
                deleted_total += stats.get("deleted_tokens", 0) or 0

            record_methods_map[uid][method_name] = entry
            success_count += 1

        with method_path.open("w", encoding="utf-8") as handle:
            json.dump(method_data, handle, ensure_ascii=False, indent=2)

        summary_entry: Dict[str, Any] = {
            "status": "ok",
            "kind": kind,
            "file": method_rel_path,
            "records_processed": len(records),
            "records_successful": success_count,
            "records_with_error": error_count,
        }

        if kind == "annotations":
            summary_entry["metrics"] = {
                "documents_with_annotations": annotation_docs,
                "total_spans": total_spans,
                "average_spans_per_document": (total_spans / annotation_docs) if annotation_docs else 0,
            }
        else:
            average_perturbed = (perturbed_total / success_count) if success_count else 0
            summary_entry["metrics"] = {
                "total_perturbed_tokens": perturbed_total,
                "total_tokens": total_tokens_total,
                "average_perturbed_tokens": average_perturbed,
                "total_added_tokens": added_total,
                "total_deleted_tokens": deleted_total,
            }

        if outcome.metadata:
            summary_entry["metadata"] = outcome.metadata

        summary_by_method[method_name] = summary_entry

    return summary_by_method


def write_summary(summary_payload: Dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)


def write_records_jsonl(
    record_info_map: Dict[str, Dict[str, Any]],
    record_methods_map: Dict[str, Dict[str, Any]],
    records_path: Path,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for uid in sorted(record_info_map.keys()):
            info = record_info_map[uid]
            entry = {
                "uid": uid,
                "original": info.get("original"),
                "name": info.get("name"),
                "metadata": info.get("metadata"),
                "utilities": info.get("utilities"),
                "methods": record_methods_map.get(uid, {}),
            }
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def print_preview(
    records: Sequence[DatasetRecord],
    record_methods_map: Dict[str, Dict[str, Any]],
    ctx: ExecutionContext,
    summary_by_method: Dict[str, Dict[str, Any]],
    limit: int,
) -> None:
    if limit <= 0:
        return

    method_cache: Dict[str, Any] = {}

    def load_method_data(rel_path: str) -> Dict[str, Any]:
        if not rel_path:
            return {}
        if rel_path not in method_cache:
            file_path = ctx.output_dir / rel_path
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    method_cache[rel_path] = json.load(handle)
            except (OSError, json.JSONDecodeError):
                method_cache[rel_path] = {}
        return method_cache[rel_path]

    print("\nPreview of anonymised outputs:")
    shown = 0
    for record in records:
        if shown >= limit:
            break
        uid = str(record.uid)
        methods = record_methods_map.get(uid)
        if not methods:
            continue
        print(f"\nUID: {uid}")
        print(f"Original: {record.text[:200]}{'...' if len(record.text) > 200 else ''}")
        for method_name in summary_by_method.keys():
            if method_name not in methods:
                continue
            method_entry = methods[method_name]
            status = method_entry.get("status", "unknown")
            print(f"  [{method_name}] status={status}")
            if status != "ok":
                reason = method_entry.get("error") or method_entry.get("reason")
                if reason:
                    print(f"    Reason: {reason}")
                continue

            if method_entry.get("type") == "anonymized_text":
                preview = method_entry.get("preview")
                if not preview:
                    data = load_method_data(method_entry.get("file", ""))
                    record_data = data.get(uid, {}) if isinstance(data, dict) else {}
                    preview = record_data.get("anonymized") or record_data.get("anonymized_text", {}).get("anonymized")
                if preview:
                    snippet = preview[:200]
                    print(f"    Anonymized: {snippet}{'...' if len(preview) > 200 else ''}")
            elif method_entry.get("type") == "annotations":
                data = load_method_data(method_entry.get("file", ""))
                spans = data.get(uid, []) if isinstance(data, dict) else []
                try:
                    applied = apply_annotations(record.text, spans)
                except Exception:
                    applied = record.text
                print(f"    Applied annotations: {applied[:200]}{'...' if len(applied) > 200 else ''}")
        shown += 1

    if shown == 0:
        print("No method outputs available for preview.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    adapter = prepare_dataset_adapter(args)
    records = list(adapter.iter_records())
    if not records:
        logger.error("No records loaded from dataset.")
        return 1

    petre_ks = tuple(sorted({int(value) for value in args.petre_ks.split(",") if value.strip()})) or DEFAULT_PETRE_KS

    dataset_part = args.dataset.lower()
    split_part = (args.split or "default").lower()
    base_output_dir = Path(args.output_dir) / dataset_part / split_part
    base_output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_state_dir = base_output_dir / "_state" / f"epsilon_{format_epsilon(args.epsilon)}"
    epsilon_state_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    resolved_device = resolve_device(args.device)

    ctx = ExecutionContext(
        epsilon=args.epsilon,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        device=resolved_device,
        seed=args.seed,
        preview_limit=args.preview_limit,
        petre_ks=petre_ks,
        output_dir=base_output_dir,
        verbose=args.verbose,
        petre_starting_annotations=args.petre_starting_annotations,
        run_id=run_id,
    )

    summary_path = epsilon_state_dir / "summary.json"
    records_path = epsilon_state_dir / "records.jsonl"

    existing_summary, record_info_map, record_methods_map = load_existing_results(summary_path, records_path)
    if existing_summary:
        if existing_summary.get("dataset") != args.dataset or existing_summary.get("epsilon") != args.epsilon:
            existing_summary = {}
            record_info_map = {}
            record_methods_map = {}

    ensure_record_entries(records, record_info_map, record_methods_map)

    summary_by_method = run_methods(
        records,
        ctx,
        args.methods,
        record_info_map,
        record_methods_map,
        existing_summary,
        force=args.force,
    )

    active_methods = set(summary_by_method.keys())
    for methods in record_methods_map.values():
        for method_key in list(methods.keys()):
            if method_key not in active_methods:
                del methods[method_key]

    summary_payload = {
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "epsilon": args.epsilon,
        "record_count": len(record_info_map),
        "methods": summary_by_method,
        "run_id": ctx.run_id,
    }
    if ctx.seed is not None:
        summary_payload["seed"] = ctx.seed

    summary_payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    run_dir = base_output_dir / "texts" / ctx.run_id
    summary_run_path = run_dir / "summary.json"
    records_run_path = run_dir / "records.jsonl"

    write_summary(summary_payload, summary_path)
    write_records_jsonl(record_info_map, record_methods_map, records_path)
    write_summary(summary_payload, summary_run_path)
    write_records_jsonl(record_info_map, record_methods_map, records_run_path)

    logger.info("Benchmark summary written to %s", summary_run_path)
    logger.info("Benchmark records written to %s", records_run_path)

    print_preview(records, record_methods_map, ctx, summary_by_method, args.preview_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
