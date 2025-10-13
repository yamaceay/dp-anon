"""Base classes for dataset adapters used in benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass
class DatasetRecord:
    """Normalized representation of a dataset record."""

    uid: str
    text: str
    name: str = ""
    annotations: Optional[Any] = None
    utilities: Dict[str, Optional[Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base adapter providing a unified interface across datasets."""

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.iter_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        """Yield normalized dataset records."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the size of the dataset if known."""
        raise NotImplementedError
