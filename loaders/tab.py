"""TAB (court cases) dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .base import DatasetAdapter, DatasetRecord

DEFAULT_TAB_PATH = Path("data/TAB/tab.json")


class TabDatasetAdapter(DatasetAdapter):
    """Adapter for the TAB anonymisation dataset."""

    def __init__(self, data_path: Optional[str] = None, max_records: Optional[int] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_TAB_PATH
        self.max_records = max_records
        try:
            with self.data_path.open("r", encoding="utf-8") as handle:
                self._records: List[dict] = json.load(handle)
        except Exception as exc:  # pragma: no cover - IO safety
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
