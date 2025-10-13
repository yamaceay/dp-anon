"""Trustpilot dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset

from .base import DatasetAdapter, DatasetRecord
from .utils import recode_text

DEFAULT_TRAIN_PATH = Path("data/trustpilot/www.amazon.com/train.json")


class TrustpilotDatasetAdapter(DatasetAdapter):
    """Adapter for Trustpilot review data."""

    def __init__(self, data_path: Optional[str] = None, max_records: Optional[int] = None):
        self.data_path = Path(data_path) if data_path else DEFAULT_TRAIN_PATH
        self.max_records = max_records

        data_files = {"train": str(self.data_path)}
        try:
            self._dataset = load_dataset("json", data_files=data_files)["train"]
        except Exception as exc:  # pragma: no cover - import/runtime safety
            raise RuntimeError(f"Failed to load Trustpilot dataset from {self.data_path}") from exc

    def __len__(self) -> int:
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for idx, row in enumerate(self._dataset):
            if self.max_records is not None and idx >= self.max_records:
                break

            uid = str(row.get("review_id", idx))
            text = recode_text(row.get("review", ""))
            utilities = {
                "category": row.get("category"),
                "stars": row.get("stars"),
            }
            metadata = dict(row)

            yield DatasetRecord(
                uid=uid,
                text=text,
                utilities=utilities,
                metadata=metadata,
            )
