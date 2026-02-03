# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python extraction
```bash
uv sync                              # install core deps
uv pip install -e ".[qwen]"          # install model-specific deps
uv run python -m vlm_attention_viz --help # run CLI
uv run python -m vlm_attention_viz --image <path> --output-dir <dir>  # extract attention
```

### Web frontend
```bash
cd web
npm install
npm run dev                           # dev server (serves data from ../output/)
DATA_DIR=../data npm run dev          # override data directory
npm run build                         # tsc && vite build
```

## Architecture

Two independent components connected by a file-based data contract:

**Python package** (`vlm_attention_viz/`) extracts attention weights from VLMs and writes per-layer float16 binary files + `meta.json`. **Web frontend** (`web/`) reads those files and renders interactive Canvas-based heatmaps.

### Python: Model adapter pattern

`cli.py` → `extract.py` → `models/`

- `models/base.py` defines `ModelAdapter` ABC with 6 abstract methods: `load`, `build_messages`, `preprocess`, `classify_token`, `build_image_grid`, plus `default_model_name` property.
- `models/__init__.py` is a lazy-loading registry (`_REGISTRY` dict mapping string keys to `(module_path, class_name)` tuples). It also auto-detects model type from `--model` name via `_AUTO_DETECT` patterns.
- `extract.py` is model-agnostic: receives an adapter instance and calls its methods to load the model, preprocess inputs, classify tokens, and build the image grid. The forward pass, attention saving, and metadata generation are all generic.
- To add a new model family: create `models/<name>.py` implementing `ModelAdapter`, add entries to `_REGISTRY` and `_AUTO_DETECT` in `models/__init__.py`, and add model-specific deps to `[project.optional-dependencies]` in `pyproject.toml`.

### Web: Vanilla TypeScript + Canvas

`main.ts` orchestrates everything. No framework — rendering is done directly on HTML Canvas for performance with large attention matrices.

- `data-loader.ts` fetches binary attention files via a Web Worker (`worker.ts` decodes float16→float32), maintains an LRU cache of 3 layers, and prefetches adjacent ±1 layers.
- `canvas-renderer.ts` has two renderers: `ImageRenderer` (image + heatmap overlay on dual canvas) and `TokenRenderer` (token sequence as wrapped text flow on single canvas with custom scrolling).
- `controls.ts` wires up UI controls and keyboard shortcuts, returns sync functions so code can update the UI.
- `color.ts` provides precomputed 256-entry RGBA lookup tables for color scales (blues, viridis, hot) and per-view min/max normalization.

### Data contract (Python → Web)

Output directory contains:
- `meta.json`: model info, token metadata (id, text, type, optional grid_pos), image grid dimensions, seq_len, num_layers, num_heads
- `attn_layer_XX.bin`: float16 binary, shape `(num_heads, seq_len, seq_len)`, C-contiguous
- `image.jpg`: copy of input image

The Vite dev server plugin (`vite.config.ts`) serves these files under `/data/` and lists available datasets at `/data/__list__`.

Token types are `"text"`, `"image"`, or `"special"`. Image tokens carry `grid_pos: [row, col]` for spatial mapping.
