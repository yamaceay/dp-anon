"""
Dataset adapter interfaces and helpers for DPMLM benchmarking.

Adapters provide a consistent way to access dataset records with a unique
identifier, raw text, optional annotations, and optional utility metadata.
"""

from .base import DatasetAdapter, DatasetRecord
from .trustpilot import TrustpilotDatasetAdapter
from .tab import TabDatasetAdapter
from .db_bio import DBBioDatasetAdapter


ADAPTER_REGISTRY = {
    "trustpilot": TrustpilotDatasetAdapter,
    "tab": TabDatasetAdapter,
    "db_bio": DBBioDatasetAdapter,
    "db-bio": DBBioDatasetAdapter,
}


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate a dataset adapter by name."""
    key = (name or "").lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset adapter '{name}'. "
            f"Available adapters: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    adapter_cls = ADAPTER_REGISTRY[key]
    return adapter_cls(**kwargs)


__all__ = [
    "DatasetAdapter",
    "DatasetRecord",
    "TrustpilotDatasetAdapter",
    "TabDatasetAdapter",
    "DBBioDatasetAdapter",
    "get_adapter",
]
