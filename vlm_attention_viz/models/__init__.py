"""Model adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ModelAdapter

# Lazy imports to avoid pulling in heavy dependencies at import time.
_REGISTRY: dict[str, tuple[str, str]] = {
    # key -> (module_path, class_name)
    "qwen3-vl": (".qwen3_vl", "Qwen3VLAdapter"),
}

# Patterns used for auto-detection from --model name.
_AUTO_DETECT: list[tuple[str, str]] = [
    ("qwen3-vl", "qwen3-vl"),
    ("qwen3_vl", "qwen3-vl"),
]


def list_model_types() -> list[str]:
    """Return all registered model type keys."""
    return sorted(_REGISTRY)


def get_adapter(model_type: str) -> ModelAdapter:
    """Instantiate and return the adapter for *model_type*."""
    if model_type not in _REGISTRY:
        available = ", ".join(list_model_types())
        raise ValueError(
            f"Unknown model type {model_type!r}. Available: {available}"
        )
    module_path, class_name = _REGISTRY[model_type]
    import importlib

    module = importlib.import_module(module_path, package=__name__)
    cls = getattr(module, class_name)
    return cls()


def detect_model_type(model_name: str) -> str | None:
    """Try to guess the model type from a HuggingFace model name."""
    lower = model_name.lower()
    for pattern, model_type in _AUTO_DETECT:
        if pattern in lower:
            return model_type
    return None
