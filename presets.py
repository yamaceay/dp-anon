"""Dataset presets for commonly used configurations."""

from __future__ import annotations

from typing import Any, Dict

PETRE_PRESETS: Dict[str, Dict[str, Any]] = {
    "trustpilot": {
        "output_base_folder_path": "outputs/trustpilot/www.amazon.com",
        "data_file_path": "data/trustpilot/www.amazon.com/train.json",
        "individual_name_column": "review_id",
        "original_text_column": "review",
        "starting_anonymization_path": "outputs/trustpilot/www.amazon.com/annotations/spacy.json",
        "tri_pipeline_path": "models/tri_pipelines/trustpilot/www.amazon.com/TRI_Pipeline",
        "ks": [2, 3, 5, 7, 10],
        "mask_text": "[MASK]",
        "use_mask_all_instances": False,
        "explainability_mode": "Greedy",
        "use_chunking": True,
    },
    "tab": {
        "output_base_folder_path": "outputs/TAB",
        "data_file_path": "data/TAB/splitted/test.json",
        "individual_name_column": "doc_id",
        "original_text_column": "text",
        "starting_anonymization_path": "outputs/tab/annotations/manual.json",
        "tri_pipeline_path": "models/tri_pipelines/TAB/TRI_Pipeline",
        "ks": [2, 3, 5, 7, 10],
        "mask_text": "[MASK]",
        "use_mask_all_instances": False,
        "explainability_mode": "Greedy",
        "use_chunking": True,
    },
}


def get_petre_preset(name: str) -> Dict[str, Any]:
    """Return a deep copy of the PETRE preset for the given dataset."""
    key = (name or "").lower()
    if key not in PETRE_PRESETS:
        raise ValueError(f"Unknown PETRE preset '{name}'. Available: {sorted(PETRE_PRESETS)}")
    preset = PETRE_PRESETS[key]
    return {k: (list(v) if isinstance(v, list) else v) for k, v in preset.items()}


__all__ = ["PETRE_PRESETS", "get_petre_preset"]
