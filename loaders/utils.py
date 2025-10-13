"""Utility helpers shared across dataset adapters."""

from __future__ import annotations


def recode_text(text: str) -> str:
    """Decode escaped characters while preserving Unicode."""
    if not isinstance(text, str):
        return text

    replacements = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        '\\"': '"',
        "\\'": "'",
    }
    for escaped, replacement in replacements.items():
        text = text.replace(escaped, replacement)
    return text.replace("\\\\", "\\")
