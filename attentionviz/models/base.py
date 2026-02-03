"""Abstract base class for VLM model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class GridInfo:
    """Spatial grid metadata for image tokens."""

    __slots__ = ("grid_rows", "grid_cols", "start_idx", "end_idx", "positions")

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        start_idx: int,
        end_idx: int,
        positions: dict[int, tuple[int, int]],
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.positions = positions  # token_index -> (row, col)


class ModelAdapter(ABC):
    """Base class that each VLM must implement to support attention extraction."""

    @property
    @abstractmethod
    def default_model_name(self) -> str:
        """HuggingFace model identifier used when --model is not specified."""

    @abstractmethod
    def load(self, model_name: str, device: str) -> tuple[Any, Any]:
        """Load and return (model, processor).

        The model must be loaded with ``attn_implementation="eager"`` so that
        full attention matrices are returned.
        """

    @abstractmethod
    def build_messages(self, image_path: Path, prompt: str) -> list[dict]:
        """Return the chat-formatted message list expected by the processor."""

    @abstractmethod
    def preprocess(
        self, processor: Any, messages: list[dict], device: str
    ) -> dict[str, torch.Tensor]:
        """Tokenize and return model-ready inputs (on *device*)."""

    @abstractmethod
    def classify_token(self, token_id: int, token_text: str) -> str:
        """Classify a single token as ``"image"``, ``"special"``, or ``"text"``."""

    @abstractmethod
    def build_image_grid(
        self,
        input_ids: list[int],
        image_width: int,
        image_height: int,
    ) -> GridInfo:
        """Compute spatial grid positions for image tokens."""

    def get_num_layers(self, model: Any) -> int:
        """Return the number of transformer layers."""
        return model.config.text_config.num_hidden_layers

    def get_num_heads(self, model: Any) -> int:
        """Return the number of attention heads."""
        return model.config.text_config.num_attention_heads
