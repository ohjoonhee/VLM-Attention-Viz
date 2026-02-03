"""Model-agnostic attention extraction orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models.base import ModelAdapter


def run_extraction(
    adapter: ModelAdapter,
    *,
    image_path: Path,
    prompt: str,
    output_dir: Path,
    model_name: str,
    device: str,
    mode: str = "prefill",
    max_new_tokens: int = 256,
) -> None:
    """Run the full extraction pipeline using *adapter*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # --- Load model ---
    print(f"Loading model: {model_name}")
    model, processor = adapter.load(model_name, device)
    num_layers = adapter.get_num_layers(model)
    num_heads = adapter.get_num_heads(model)
    print(f"Model loaded: {num_layers} layers, {num_heads} heads")

    # --- Preprocess ---
    messages = adapter.build_messages(image_path, prompt)
    inputs = adapter.preprocess(processor, messages, device)

    input_ids_list: list[int] = inputs["input_ids"][0].tolist()
    print(f"Input sequence length: {len(input_ids_list)} tokens")

    # --- Optionally generate first ---
    if mode == "generate":
        print(f"Generating (max_new_tokens={max_new_tokens})...")
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        full_ids = gen_output[0].tolist()
        num_generated = len(full_ids) - len(input_ids_list)
        print(f"Generated {num_generated} tokens, total sequence: {len(full_ids)}")

        forward_inputs: dict = {
            "input_ids": gen_output,
            "attention_mask": torch.ones_like(gen_output),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in inputs:
                forward_inputs[key] = inputs[key]

        input_ids_list = full_ids
    else:
        forward_inputs = dict(inputs)

    seq_len = len(input_ids_list)
    print(f"Extracting attention for {seq_len} tokens (mode={mode})...")

    # --- Decode tokens ---
    token_texts: list[str] = []
    for tid in input_ids_list:
        decoded = processor.tokenizer.decode([tid], skip_special_tokens=False)
        token_texts.append(decoded)

    # --- Image grid ---
    image_grid_info = adapter.build_image_grid(input_ids_list, orig_w, orig_h)

    # --- Forward pass ---
    print("Running forward pass (output_attentions=True)...")
    with torch.no_grad():
        outputs = model(
            **forward_inputs,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    print(f"Got attention from {len(attentions)} layers")

    # --- Save per-layer attention ---
    for layer_idx, attn_tensor in enumerate(attentions):
        attn_np = attn_tensor[0].cpu().to(torch.float16).numpy()
        out_path = output_dir / f"attn_layer_{layer_idx:02d}.bin"
        attn_np.tofile(str(out_path))
        print(f"  Saved {out_path.name} ({attn_np.nbytes / 1024 / 1024:.1f} MB)")

    # --- Build metadata ---
    tokens_meta = []
    for i, (tid, text_str) in enumerate(zip(input_ids_list, token_texts)):
        tok_type = adapter.classify_token(tid, text_str)
        entry: dict = {"id": i, "text": text_str, "type": tok_type}
        if i in image_grid_info.positions:
            entry["grid_pos"] = list(image_grid_info.positions[i])
        tokens_meta.append(entry)

    meta = {
        "model": model_name,
        "mode": mode,
        "prompt": prompt,
        "image_path": "image.jpg",
        "image_size": [orig_w, orig_h],
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "dtype": "float16",
        "tokens": tokens_meta,
        "image_grid": {
            "rows": image_grid_info.grid_rows,
            "cols": image_grid_info.grid_cols,
            "start_idx": image_grid_info.start_idx,
            "end_idx": image_grid_info.end_idx,
        },
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {meta_path}")

    # --- Copy image ---
    dest_image = output_dir / "image.jpg"
    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(image_path, dest_image)
    else:
        image.save(dest_image, "JPEG", quality=95)
    print(f"Saved {dest_image}")

    print(f"\nDone. Output in {output_dir}/")
    print(f"  - meta.json ({meta_path.stat().st_size / 1024:.0f} KB)")
    total_attn_mb = (
        sum(
            (output_dir / f"attn_layer_{i:02d}.bin").stat().st_size
            for i in range(num_layers)
        )
        / 1024
        / 1024
    )
    print(f"  - {num_layers} attention files ({total_attn_mb:.1f} MB total)")
