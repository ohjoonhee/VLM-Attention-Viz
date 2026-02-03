"""CLI entry point for attention extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .models import detect_model_type, get_adapter, list_model_types


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vlm-attention-viz",
        description="Extract attention weights from vision-language models.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for output files"
    )
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=list_model_types(),
        help="Model family (auto-detected from --model when omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefill", "generate"],
        default="prefill",
        help="prefill: input-only attention; generate: full-sequence attention",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (generate mode only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve model type.
    model_type = args.model_type
    if model_type is None and args.model is not None:
        model_type = detect_model_type(args.model)
    if model_type is None and args.model is None:
        model_type = "qwen3-vl"  # default
    if model_type is None:
        available = ", ".join(list_model_types())
        print(
            f"Error: could not detect model type from {args.model!r}. "
            f"Use --model-type ({available}).",
            file=sys.stderr,
        )
        sys.exit(1)

    adapter = get_adapter(model_type)
    model_name = args.model or adapter.default_model_name

    from .extract import run_extraction

    run_extraction(
        adapter,
        image_path=Path(args.image),
        prompt=args.prompt,
        output_dir=Path(args.output_dir),
        model_name=model_name,
        device=args.device,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
    )
