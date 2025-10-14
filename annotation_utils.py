"""Utility helpers for working with span annotations and anonymised text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


AnnotationMapping = Mapping[str, Any]


@dataclass(frozen=True)
class NormalisedAnnotation:
    """Simple container representing a span annotation."""

    start: int
    end: int
    text: str = ""
    label: Optional[str] = None
    replacement: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, mapping: AnnotationMapping) -> "NormalisedAnnotation":
        """Create a normalised annotation from a loosely defined mapping."""

        start = _safe_int(mapping, ("start", "start_offset", "begin"))
        end = _safe_int(mapping, ("end", "end_offset", "stop"))
        if start is None or end is None:
            raise ValueError(f"Annotation missing span offsets: {mapping}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid annotation offsets: start={start}, end={end}")

        label = _first_non_empty(mapping, ("label", "entity_type", "type", "tag"))
        replacement = _first_non_empty(mapping, ("replacement", "mask", "value"))
        text = str(mapping.get("text") or mapping.get("span_text") or "")
        confidence = _safe_float(mapping, ("confidence", "score", "probability"))
        source = _first_non_empty(mapping, ("source", "annotator", "annotator_id"))
        metadata = dict(mapping.get("metadata") or {})

        return cls(
            start=start,
            end=end,
            text=text,
            label=str(label) if label is not None else None,
            replacement=str(replacement) if replacement is not None else None,
            confidence=confidence,
            source=str(source) if source is not None else None,
            metadata=metadata or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serialisable dict while omitting empty fields."""
        payload: Dict[str, Any] = {"start": self.start, "end": self.end}
        if self.text:
            payload["text"] = self.text
        if self.label:
            payload["label"] = self.label
        if self.replacement:
            payload["replacement"] = self.replacement
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.source:
            payload["source"] = self.source
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


ReplacementFn = Callable[[NormalisedAnnotation], str]


def _safe_int(mapping: AnnotationMapping, keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return int(mapping[key])
            except (TypeError, ValueError):
                continue
    return None


def _safe_float(mapping: AnnotationMapping, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return None


def _first_non_empty(mapping: AnnotationMapping, keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


def normalise_annotations(
    annotations: Sequence[AnnotationMapping],
) -> List[NormalisedAnnotation]:
    """Normalise a heterogeneous set of annotation mappings."""
    normalised: List[NormalisedAnnotation] = []
    for annot in annotations:
        normalised.append(NormalisedAnnotation.from_mapping(annot))
    return normalised


def apply_annotations(
    text: str,
    annotations: Sequence[AnnotationMapping],
    *,
    replacement_fn: Optional[ReplacementFn] = None,
) -> str:
    """Apply annotations to text and return the anonymised result.

    Args:
        text: Source text to anonymise.
        annotations: Sequence of annotation mappings.  Each mapping is expected
            to contain at least `start` and `end` keys.  Additional metadata is
            ignored for replacement purposes but preserved for the replacement
            function.
        replacement_fn: Optional callable returning the replacement string for
            each annotation.  When omitted, the label of the annotation is used
            if available, otherwise the placeholder ``[REDACTED]``.

    Returns:
        The anonymised text with spans replaced according to the annotations.
    """

    if not annotations:
        return text

    normalised = normalise_annotations(annotations)
    sorted_spans = sorted(normalised, key=lambda span: span.start, reverse=True)

    def default_replacement(annotation: NormalisedAnnotation) -> str:
        if annotation.replacement:
            return annotation.replacement
        if annotation.label:
            return f"[{str(annotation.label).upper()}]"
        return "[REDACTED]"

    replacement_callable: ReplacementFn = replacement_fn or default_replacement

    result = text
    for annotation in sorted_spans:
        start, end = annotation.start, annotation.end
        if start >= len(result) or end > len(result):
            continue
        replacement = replacement_callable(annotation)
        result = result[:start] + replacement + result[end:]

    return result


__all__ = [
    "NormalisedAnnotation",
    "apply_annotations",
    "normalise_annotations",
    "ReplacementFn",
]
