"""DB-Bio dataset adapter with flexible split discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

from datasets import Dataset, load_from_disk

from .base import DatasetAdapter, DatasetRecord

DEFAULT_DB_BIO_ROOT = Path("data/db_bio")


class DBBioDatasetAdapter(DatasetAdapter):
    """Adapter for the DB-Bio legal dataset."""

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        max_records: Optional[int] = None,
    ):
        self.root = Path(root).expanduser() if root else DEFAULT_DB_BIO_ROOT
        self.split = split
        self.max_records = max_records

        split_path = self._find_split_path()
        self._dataset_path = split_path

        if split_path.is_dir():
            try:
                self._dataset: Union[Dataset, list] = load_from_disk(str(split_path))
            except Exception as exc:  # pragma: no cover - runtime safety
                raise RuntimeError(f"Failed to load DB-Bio dataset from {split_path}") from exc
        else:
            try:
                with split_path.open("r", encoding="utf-8") as handle:
                    self._dataset = json.load(handle)
            except json.JSONDecodeError:
                with split_path.open("r", encoding="utf-8") as handle:
                    self._dataset = [json.loads(line) for line in handle]
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"Failed to load DB-Bio dataset from {split_path}") from exc

    def __len__(self) -> int:
        if isinstance(self._dataset, Dataset):
            return len(self._dataset)
        return len(self._dataset)

    def iter_records(self) -> Iterable[DatasetRecord]:
        if isinstance(self._dataset, Dataset):
            iterator = self._dataset
        else:
            iterator = self._dataset

        for idx, row in enumerate(iterator):
            if self.max_records is not None and idx >= self.max_records:
                break

            if isinstance(row, dict):
                data = row
            else:  # Dataset row returns dict already; safeguard
                data = dict(row)

            text = data.get("text", "")
            uid = data.get("wiki_name") or data.get("label") or str(idx)
            name = data.get("people")
            annotations = None
            utilities = {
                "label": data.get("label"),
                "l1": data.get("l1"),
                "l2": data.get("l2"),
                "l3": data.get("l3"),
            }
            metadata = {
                "word_count": data.get("word_count"),
                "wiki_name": data.get("wiki_name"),
            }

            yield DatasetRecord(
                uid=str(uid),
                text=text,
                name=name,
                annotations=annotations,
                utilities=utilities,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_split_path(self) -> Path:
        """Locate the requested split directory/file."""
        candidates = []

        direct_dir = self.root / self.split
        if direct_dir.is_dir():
            if any(direct_dir.glob("data-*.arrow")) or (direct_dir / "dataset_info.json").exists():
                return direct_dir
            candidates.append(direct_dir)

        for suffix in (".jsonl", ".json"):
            direct_file = self.root / f"{self.split}{suffix}"
            if direct_file.is_file():
                return direct_file

        for dir_path in self.root.rglob(self.split):
            if not dir_path.is_dir():
                continue
            if any(dir_path.glob("data-*.arrow")) or (dir_path / "dataset_info.json").exists():
                return dir_path

        for suffix in (".jsonl", ".json"):
            matches = list(self.root.rglob(f"{self.split}{suffix}"))
            if matches:
                return matches[0]

        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            f"Unable to locate DB-Bio split '{self.split}' under {self.root.expanduser()}"
        )


__all__ = ["DBBioDatasetAdapter"]
