#!/usr/bin/env python3
"""Standalone annotation generator using loader adapters."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

from tqdm import tqdm

from loaders import DatasetAdapter, get_adapter
from loaders.base import DatasetRecord

try:
    import en_core_web_lg
    SPACY_MODEL_AVAILABLE = True
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_MODEL_AVAILABLE = False
    try:
        import spacy
        SPACY_AVAILABLE = True
        
    except ImportError:
        SPACY_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


class StandaloneAnnotationGenerator:
    def __init__(self):
        self.spacy_nlp = None
        if SPACY_AVAILABLE:
            if SPACY_MODEL_AVAILABLE:
                try:
                    self.spacy_nlp = en_core_web_lg.load()
                except Exception:
                    pass
            else:
                try:
                    self.spacy_nlp = spacy.load("en_core_web_lg")
                except OSError:
                    pass
        
        self.presidio_analyzer = None
        if PRESIDIO_AVAILABLE:
            try:
                self.presidio_analyzer = AnalyzerEngine()
            except Exception:
                pass

        self.manual_annotations: Optional[Dict[str, List[Dict[str, object]]]] = None

    def get_available_methods(self) -> Dict[str, Dict[str, Any]]:
        methods = {
            "spacy": {
                "available": SPACY_AVAILABLE and self.spacy_nlp is not None,
                "description": "spaCy Named Entity Recognition"
            },
            "presidio": {
                "available": PRESIDIO_AVAILABLE and self.presidio_analyzer is not None,
                "description": "Microsoft Presidio PII Detection"
            },
            "manual": {
                "available": True,
                "description": "User-provided manual annotations"
            },
        }
        
        if not methods["spacy"]["available"]:
            methods["spacy"]["error"] = "spaCy not available"
        if not methods["presidio"]["available"]:
            methods["presidio"]["error"] = "Presidio not available"
        return methods

    def generate_spacy_annotations(self, records: Iterable[DatasetRecord]) -> Tuple[Dict[str, List], Dict[str, Any]]:
        if not self.spacy_nlp:
            raise RuntimeError("spaCy not available")

        annotations = {}
        total_spans = 0
        documents_with_annotations = 0

        records_list = list(records)

        for record in tqdm(records_list, desc="Processing with spaCy", unit="docs"):
            key = str(record.uid)
            text = record.text or ""

            doc = self.spacy_nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": ent.label_,
                    "confidence": 1.0
                })

            if entities:
                annotations[key] = entities
                documents_with_annotations += 1
                total_spans += len(entities)

        stats = {
            "total_documents": len(records_list),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(records_list) if len(records_list) > 0 else 0,
            "average_spans_per_document": total_spans / documents_with_annotations if documents_with_annotations > 0 else 0
        }

        return annotations, stats

    def generate_presidio_annotations(self, records: Iterable[DatasetRecord]) -> Tuple[Dict[str, List], Dict[str, Any]]:
        if not self.presidio_analyzer:
            raise RuntimeError("Presidio not available")

        annotations = {}
        total_spans = 0
        documents_with_annotations = 0

        records_list = list(records)

        for record in tqdm(records_list, desc="Processing with Presidio", unit="docs"):
            key = str(record.uid)
            text = record.text or ""

            results = self.presidio_analyzer.analyze(text=text, language='en')
            entities = []
            for result in results:
                entities.append({
                    "start": result.start,
                    "end": result.end,
                    "text": text[result.start:result.end],
                    "label": result.entity_type,
                    "confidence": result.score
                })

            if entities:
                annotations[key] = entities
                documents_with_annotations += 1
                total_spans += len(entities)

        stats = {
            "total_documents": len(records_list),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(records_list) if len(records_list) > 0 else 0,
            "average_spans_per_document": total_spans / documents_with_annotations if documents_with_annotations > 0 else 0
        }

        return annotations, stats

    def generate_manual_annotations(
        self,
        records: Sequence[DatasetRecord],
    ) -> Tuple[Dict[str, List], Dict[str, float]]:
        final_records: Dict[str, List] = {}

        for record in records:
            final_annotations = []
            all_annotations = record.annotations
            for annotator_id, annotations in all_annotations.items():
                final_annotations.extend(
                    {
                        "start": annot.get("start_offset", 0),
                        "end":annot.get("end_offset", 0),
                        "text": annot.get("span_text"),
                        "label": annot.get("entity_type"),
                        "confidence": 1.0,
                        "metadata": {
                            "annotator_id": annotator_id,
                            "confidential_status": annot.get("confidential_status"),
                            "identifier_type": annot.get("identifier_type"),
                        }
                    }
                    for annot in annotations["entity_mentions"]
                )
            final_annotations_sorted = sorted(
                final_annotations,
                key=lambda x: x["start"],
            )
            final_records.update({str(record.uid): final_annotations_sorted})

        documents_with_annotations = sum(1 for anns in final_records if anns)
        total_spans = sum(len(anns) for anns in final_records.values())
        stats = {
            "total_documents": len(final_records),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(final_records) if final_records else 0,
            "average_spans_per_document": (total_spans / documents_with_annotations
                                            if documents_with_annotations > 0 else 0),
        }

        return final_records, stats

    def save_annotations(self, annotations: Dict[str, List], output_file: str, indent = None) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=indent, ensure_ascii=False)

def _build_adapter(dataset: str, args: argparse.Namespace) -> DatasetAdapter:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotation spans using loader datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name (trustpilot, tab, db_bio).")
    parser.add_argument("--dataset-path", default=None, help="Override dataset root/path.")
    parser.add_argument("--split", default="train", help="Dataset split (if applicable).")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of records.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("spacy",),
        help="Annotation methods to run (spacy, presidio, manual).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write annotation files.",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Store full span metadata rather than offsets only.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Overwrite existing annotation files.",
    )
    parser.add_argument(
        "--output-stats",
        action="store_true",
        help="Print method statistics.",
    )
    return parser.parse_args()


def filter_to_offsets(annotations: Dict[str, List[Dict[str, object]]]) -> Dict[str, List[Tuple[int, int]]]:
    filtered: Dict[str, List[Tuple[int, int]]] = {}
    for doc_id, entities in annotations.items():
        offsets: List[Tuple[int, int]] = []
        prev_start, prev_end = -1, -1
        for ent in entities:
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            if prev_start <= start < prev_end:
                continue
            offsets.append((start, end))
            prev_start, prev_end = start, end
        filtered[doc_id] = offsets
    return filtered


def main() -> None:
    args = parse_args()

    adapter = _build_adapter(args.dataset, args)
    records = list(adapter.iter_records())
    print(f"Loaded {len(records)} records from dataset '{args.dataset}'.")

    generator = StandaloneAnnotationGenerator()
    available_methods = generator.get_available_methods()

    requested_methods = [method.lower() for method in args.methods]
    results_generated = 0

    output_dir = Path(args.output_dir) / args.dataset.lower()
    if args.split:
        output_dir = output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing annotations to: {output_dir}")

    for method in requested_methods:
        if method not in available_methods:
            print(f"Unknown method '{method}'. Skipping.")
            continue

        info = available_methods[method]
        if not info["available"] and method != "manual":
            print(f"Method '{method}' unavailable: {info.get('error', 'dependency missing')}")
            continue

        output_file = output_dir / f"{method}.json"
        if output_file.exists() and not args.force_regenerate:
            print(f"Skipping {method}: {output_file} already exists.")
            continue

        print(f"\nGenerating {method} annotations...")
        try:
            if method == "spacy":
                annotations, stats = generator.generate_spacy_annotations(records)
            elif method == "presidio":
                annotations, stats = generator.generate_presidio_annotations(records)
            elif method == "manual":
                annotations, stats = generator.generate_manual_annotations(records)
            else:
                print(f"Method '{method}' not supported.")
                continue

            if not args.include_other:
                annotations = filter_to_offsets(annotations)

            generator.save_annotations(
                annotations,
                str(output_file),
                indent=2 if args.include_other else None,
            )
            results_generated += 1

            print(f"✓ Generated {method} annotations.")
            print(f"  Documents with annotations: {stats['documents_with_annotations']}")
            print(f"  Total spans: {stats['total_spans']}")
            print(f"  Coverage: {stats['coverage']:.2%}")
            if args.output_stats:
                print(f"  Avg spans per doc: {stats['average_spans_per_document']:.2f}")
        except Exception as exc:
            print(f"✗ Failed to generate {method} annotations: {exc}")

    print(f"\nGenerated {results_generated} annotation files.")


if __name__ == "__main__":
    main()
