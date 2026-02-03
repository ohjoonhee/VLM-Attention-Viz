"""
Extract attention weights from Qwen3-VL-2B-Thinking during inference.

Runs a forward pass on an image + text prompt, collects attention matrices
from every layer, and writes them to disk as per-layer float16 binary files
alongside a meta.json with token and grid information.

Usage:
    python extract.py \
        --image path/to/image.jpg \
        --prompt "Describe this image in detail." \
        --output-dir ./data/sample1
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Thinking"

# Qwen VL special token IDs (shared across Qwen VL family)
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_TOKEN_ID = 151655


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract attention weights from Qwen3-VL-2B-Thinking")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
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
        help="prefill: attention from input only; generate: run generation then attention on full sequence",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (only used in generate mode)",
    )
    return parser.parse_args()


def classify_token(token_id: int, token_text: str) -> str:
    """Return token type: 'image', 'special', or 'text'."""
    if token_id == IMAGE_TOKEN_ID:
        return "image"
    if token_text.startswith("<|") and token_text.endswith("|>"):
        return "special"
    if token_text.startswith("<") and token_text.endswith(">"):
        return "special"
    return "text"


def build_image_grid_positions(
    input_ids: list[int],
    image_width: int,
    image_height: int,
    merge_size: int = 2,
    patch_size: int = 16,
) -> dict:
    """Compute grid positions for image tokens and return grid metadata."""
    effective_patch = patch_size * merge_size  # 32 px per token
    grid_cols = image_width // effective_patch
    grid_rows = image_height // effective_patch

    # Find the image token block
    start_idx = None
    end_idx = None
    for i, tid in enumerate(input_ids):
        if tid == IMAGE_TOKEN_ID:
            if start_idx is None:
                start_idx = i
            end_idx = i

    if start_idx is None:
        return {"grid_rows": 0, "grid_cols": 0, "start_idx": 0, "end_idx": 0, "positions": {}}

    # Map each image token index to (row, col)
    positions = {}
    img_idx = 0
    for i in range(start_idx, end_idx + 1):
        if input_ids[i] == IMAGE_TOKEN_ID:
            row = img_idx // grid_cols
            col = img_idx % grid_cols
            positions[i] = (row, col)
            img_idx += 1

    return {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "positions": positions,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype="float16",
        attn_implementation="eager",  # required to get full attention matrices
    ).to(args.device)
    model.eval()

    text_config = model.config.text_config
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads

    print(f"Model loaded: {num_layers} layers, {num_heads} heads")

    # Build the chat message in Qwen VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": args.prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Okay let's look at the image. A girl is standing in front of a staircase."}],
        },
    ]

    # Apply the chat template and process vision inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(args.device)

    input_ids_list = inputs["input_ids"][0].tolist()
    print(f"Input sequence length: {len(input_ids_list)} tokens")

    if args.mode == "generate":
        # Step 1: Generate output tokens
        print(f"Generating (max_new_tokens={args.max_new_tokens})...")
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        # gen_output is (1, input_len + generated_len)
        full_ids = gen_output[0].tolist()
        num_generated = len(full_ids) - len(input_ids_list)
        print(f"Generated {num_generated} tokens, total sequence: {len(full_ids)}")

        # Step 2: Build forward-pass inputs using the full token sequence
        # Reuse original vision tensors (pixel_values, image_grid_thw) and just
        # replace input_ids / attention_mask with the full generated sequence.
        full_ids_tensor = gen_output  # already (1, full_len) on device
        forward_inputs = {
            "input_ids": full_ids_tensor,
            "attention_mask": torch.ones_like(full_ids_tensor),
        }
        # Carry over vision-related tensors from the original inputs
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in inputs:
                forward_inputs[key] = inputs[key]

        input_ids_list = full_ids
    else:
        forward_inputs = dict(inputs)

    seq_len = len(input_ids_list)
    print(f"Extracting attention for {seq_len} tokens (mode={args.mode})...")

    # Decode each token for metadata
    token_texts = []
    for tid in input_ids_list:
        decoded = processor.tokenizer.decode([tid], skip_special_tokens=False)
        token_texts.append(decoded)

    # Build image grid info using the processed image size
    # The processor resizes images; find the actual grid from token counts
    image_grid_info = build_image_grid_positions(input_ids_list, orig_w, orig_h)

    # Forward pass with attention output
    print("Running forward pass (output_attentions=True)...")
    with torch.no_grad():
        outputs = model(
            **forward_inputs,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # tuple of (batch, num_heads, seq_len, seq_len)
    print(f"Got attention from {len(attentions)} layers")

    # Save per-layer attention as float16 binary
    for layer_idx, attn_tensor in enumerate(attentions):
        # attn_tensor shape: (1, num_heads, seq_len, seq_len)
        attn_np = attn_tensor[0].cpu().to(torch.float16).numpy()  # (heads, seq, seq)
        out_path = output_dir / f"attn_layer_{layer_idx:02d}.bin"
        attn_np.tofile(str(out_path))
        print(f"  Saved {out_path.name} ({attn_np.nbytes / 1024 / 1024:.1f} MB)")

    # Build token metadata
    tokens_meta = []
    for i, (tid, text_str) in enumerate(zip(input_ids_list, token_texts)):
        tok_type = classify_token(tid, text_str)
        entry = {"id": i, "text": text_str, "type": tok_type}
        if i in image_grid_info["positions"]:
            entry["grid_pos"] = list(image_grid_info["positions"][i])
        tokens_meta.append(entry)

    # Write meta.json
    meta = {
        "model": args.model,
        "mode": args.mode,
        "prompt": args.prompt,
        "image_path": "image.jpg",
        "image_size": [orig_w, orig_h],
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "dtype": "float16",
        "tokens": tokens_meta,
        "image_grid": {
            "rows": image_grid_info["grid_rows"],
            "cols": image_grid_info["grid_cols"],
            "start_idx": image_grid_info["start_idx"],
            "end_idx": image_grid_info["end_idx"],
        },
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {meta_path}")

    # Copy the input image
    dest_image = output_dir / "image.jpg"
    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(image_path, dest_image)
    else:
        image.save(dest_image, "JPEG", quality=95)
    print(f"Saved {dest_image}")

    print(f"\nDone. Output in {output_dir}/")
    print(f"  - meta.json ({meta_path.stat().st_size / 1024:.0f} KB)")
    total_attn_mb = sum((output_dir / f"attn_layer_{i:02d}.bin").stat().st_size for i in range(num_layers)) / 1024 / 1024
    print(f"  - {num_layers} attention files ({total_attn_mb:.1f} MB total)")


if __name__ == "__main__":
    main()
