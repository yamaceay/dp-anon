#!/usr/bin/env python3
"""Standalone annotation generator - no TRI dependencies."""

import json
import os
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

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

        self.manual_annotations = None
    
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
    
    def generate_spacy_annotations(self, data: list, text_column: str, id_column: str, name_column: str) -> Tuple[Dict[str, List], Dict[str, Any]]:
        if not self.spacy_nlp:
            raise RuntimeError("spaCy not available")
        
        annotations = {}
        total_spans = 0
        documents_with_annotations = 0
        
        
        for row in tqdm(data, desc="Processing with spaCy", unit="docs"):
            doc_id = str(row[id_column])
            text = str(row[text_column])
            name = get_name(row, name_column)
            key = name if name is not None else doc_id

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
            "total_documents": len(data),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(data) if len(data) > 0 else 0,
            "average_spans_per_document": total_spans / documents_with_annotations if documents_with_annotations > 0 else 0
        }
        
        return annotations, stats

    def generate_presidio_annotations(self, data: list, text_column: str, id_column: str, name_column: str) -> Tuple[Dict[str, List], Dict[str, Any]]:
        if not self.presidio_analyzer:
            raise RuntimeError("Presidio not available")
        
        annotations = {}
        total_spans = 0
        documents_with_annotations = 0
        
        
        for row in tqdm(data, desc="Processing with Presidio", unit="docs"):
            doc_id = str(row[id_column])
            text = str(row[text_column])
            name = get_name(row, name_column)
            key = name if name is not None else doc_id

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
            "total_documents": len(data),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(data) if len(data) > 0 else 0,
            "average_spans_per_document": total_spans / documents_with_annotations if documents_with_annotations > 0 else 0
        }
        
        return annotations, stats

    def generate_manual_annotations(self, data: list, id_column: str, name_column: str, annotation_column: str) -> None:
        """Load user-provided manual annotations."""
        manual_annotations_dict = {}
        for row in data:
            doc_id = str(row[id_column])
            name = get_name(row, name_column)
            key = name if name is not None else doc_id
            all_annotations = row[annotation_column]
            if all_annotations:
                found_entities = []
                for annotator, annotations in all_annotations.items():
                    for entity in annotations['entity_mentions']:
                        
                        span_text = entity.get("span_text") or entity.get("span_ext", "")
                        
                        entity_processed = {
                            "start": entity["start_offset"],
                            "end": entity["end_offset"],
                            "text": span_text,
                            "label": entity["entity_type"],
                            "confidence": 1.0,
                            "confidential_status": entity.get("confidential_status", "UNKNOWN"),
                            "identifier_type": entity.get("identifier_type", "UNKNOWN"),
                            "annotator": annotator,
                        }
                        found_entities.append(entity_processed)
            manual_annotations_dict[key] = sorted(found_entities, key=lambda x: x["start"])

        documents_with_annotations = sum(1 for anns in manual_annotations_dict.values() if anns)
        total_spans = sum(len(anns) for anns in manual_annotations_dict.values())
        stats = {
            "total_documents": len(data),
            "documents_with_annotations": documents_with_annotations,
            "total_spans": total_spans,
            "coverage": documents_with_annotations / len(data) if len(data) > 0 else 0,
            "average_spans_per_document": (total_spans / documents_with_annotations
                                            if documents_with_annotations > 0 else 0),
        }

        return manual_annotations_dict, stats

    def save_annotations(self, annotations: Dict[str, List], output_file: str, indent = None) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=indent, ensure_ascii=False)

def get_name(row, name_column):
    if not name_column:
        return None
    name_column_deep = name_column.split('.')
    val = row
    for col in name_column_deep:
        val = val.get(col, {})
    return val if isinstance(val, str) else None

def read_from_config_file(config_path: str) -> Dict[str, Any]:
    empty_config = {
        "input_file": None,
        "output_dir": None,
        "id_column": None,
        "text_column": None,
        "annotation_column": None,
        "name_column": None,
        "methods": [],
        "all_methods": False,
        "include_other": False,
        "force_regenerate": False,
        "output_stats": False,
    }
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    required_keys = ["input_file", "output_dir", "id_column", "text_column"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if "annotation_column" not in config and config.get("method") == "manual":
        raise ValueError("annotation_column is required for manual method")
    
    if "methods" not in config and not config.get("all_methods", False):
        raise ValueError("Either methods or all_methods must be specified")

    for key, value in empty_config.items():
        if key not in config:
            config[key] = value

    return config

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate annotations for dataset")
    parser.add_argument("config", type=str,
                        help="Path to JSON configuration file (not used in standalone mode)")
    
    args = parser.parse_args()
    config = read_from_config_file(args.config)
    args = argparse.Namespace(**config)

    try:
        if args.input_file.endswith('.json'):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print(f"Unsupported file format: {args.input_file}")
            sys.exit(1)

        print(f"Loaded {len(data)} documents from {args.input_file}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    columns = list(data[0].keys()) if data else []

    
    if args.id_column not in columns:
        print(f"Error: ID column '{args.id_column}' not found. Available: {list(columns)}")
        sys.exit(1)
    if args.text_column not in columns:
        print(f"Error: Text column '{args.text_column}' not found. Available: {list(columns)}")
        sys.exit(1)
    if "manual" in args.methods and args.annotation_column and args.annotation_column not in columns:
        print(f"Error: Annotation column '{args.annotation_column}' not found. Available: {list(columns)}")
        sys.exit(1)

    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using annotation folder: {args.output_dir}")
    
    
    if args.all_methods:
        methods_to_generate = ["spacy", "presidio", "manual"]
    elif args.methods:
        methods_to_generate = args.methods
    else:
        methods_to_generate = []
    
    print(f"Generating annotations for methods: {methods_to_generate}")
    
    
    generator = StandaloneAnnotationGenerator()
    
    
    available_methods = generator.get_available_methods()
    print("Available annotation methods:")
    for method, info in available_methods.items():
        status = "✓" if info["available"] else "✗"
        print(f"  {status} {method}: {info['description']}")
        if not info["available"] and "error" in info:
            print(f"    Error: {info['error']}")
    
    
    total_generated = 0
    
    
    method_progress = tqdm(methods_to_generate, desc="Generating annotations", unit="method")
    
    for method in method_progress:
        method_progress.set_description(f"Generating {method} annotations")
        annotation_file = os.path.join(args.output_dir, f"{method}.json")
        
        if os.path.exists(annotation_file) and not args.force_regenerate:
            print(f"Skipping {method} - annotations already exist")
            continue
            
        print(f"\nGenerating {method} annotations...")
        
        try:
            if method == "spacy":
                method_annotations, stats = generator.generate_spacy_annotations(
                    data, args.text_column, args.id_column, args.name_column
                )
            elif method == "presidio":
                method_annotations, stats = generator.generate_presidio_annotations(
                    data, args.text_column, args.id_column, args.name_column
                )
            elif method == "manual":
                method_annotations, stats = generator.generate_manual_annotations(
                    data, args.id_column, args.name_column, args.annotation_column
                )
            elif method == "ner7":
                print("NER7 is not supported in standalone mode.")
                continue
            else:
                print(f"Unknown method: {method}")
                continue

            if not args.include_other:
                print("Filtering to only offsets...")
                data_with_offsets = {}
                for doc_id, entities in method_annotations.items():
                    offsets = []
                    prev_offset = (-1, -1)
                    for ent in entities:
                        start, end = ent["start"], ent["end"]
                        prev_start, prev_end = prev_offset
                        if prev_start <= start < prev_end:
                            continue  
                        offsets.append((start, end))
                        prev_offset = (start, end)

                    data_with_offsets[doc_id] = offsets
                method_annotations = data_with_offsets
            
            print("Saving annotations...")
            generator.save_annotations(method_annotations, annotation_file, indent=2 if args.include_other else None)
            total_generated += 1
            
            print(f"✓ Generated {method} annotations:")
            print(f"  Documents with annotations: {stats['documents_with_annotations']}")
            print(f"  Total spans: {stats['total_spans']}")
            print(f"  Coverage: {stats['coverage']:.2%}")
            
            if args.output_stats:
                print(f"  Average spans per document: {stats['average_spans_per_document']:.2f}")
                
        except Exception as e:
            print(f"✗ Failed to generate {method} annotations: {e}")
    
    method_progress.close()
    
    print(f"\nGenerated {total_generated} new annotation files!")


if __name__ == "__main__":
    main()
