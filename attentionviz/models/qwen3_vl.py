"""Qwen3-VL model adapter for attention extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .base import GridInfo, ModelAdapter

# Qwen VL special token IDs (shared across the Qwen VL family).
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_TOKEN_ID = 151655


class Qwen3VLAdapter(ModelAdapter):
    """Adapter for Qwen3-VL models (2B, 8B, etc.)."""

    @property
    def default_model_name(self) -> str:
        return "Qwen/Qwen3-VL-2B-Thinking"

    def load(self, model_name: str, device: str) -> tuple[Any, Any]:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="float16",
            attn_implementation="eager",
        ).to(device)
        model.eval()
        return model, processor

    def build_messages(self, image_path: Path, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def preprocess(
        self, processor: Any, messages: list[dict], device: str
    ) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)
        return inputs

    def classify_token(self, token_id: int, token_text: str) -> str:
        if token_id == IMAGE_TOKEN_ID:
            return "image"
        if token_text.startswith("<|") and token_text.endswith("|>"):
            return "special"
        if token_text.startswith("<") and token_text.endswith(">"):
            return "special"
        return "text"

    def build_image_grid(
        self,
        input_ids: list[int],
        image_width: int,
        image_height: int,
    ) -> GridInfo:
        merge_size = 2
        patch_size = 16
        effective_patch = patch_size * merge_size  # 32 px per token
        grid_cols = image_width // effective_patch
        grid_rows = image_height // effective_patch

        start_idx: int | None = None
        end_idx: int | None = None
        for i, tid in enumerate(input_ids):
            if tid == IMAGE_TOKEN_ID:
                if start_idx is None:
                    start_idx = i
                end_idx = i

        if start_idx is None or end_idx is None:
            return GridInfo(0, 0, 0, 0, {})

        positions: dict[int, tuple[int, int]] = {}
        img_idx = 0
        for i in range(start_idx, end_idx + 1):
            if input_ids[i] == IMAGE_TOKEN_ID:
                row = img_idx // grid_cols
                col = img_idx % grid_cols
                positions[i] = (row, col)
                img_idx += 1

        return GridInfo(grid_rows, grid_cols, start_idx, end_idx, positions)
