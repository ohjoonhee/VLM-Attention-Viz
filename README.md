# VLM Attention Viz

Web-based visualization of attention patterns in vision-language models.

**Two components:**

1. **Python extraction package** -- runs VLM inference on an image + prompt, extracts attention weights from every layer, and saves them as per-layer float16 binary files. Supports multiple model families via an adapter pattern.
2. **Web frontend** -- loads pre-extracted data and provides interactive heatmap visualization across all layers and heads.

See [SPEC.md](SPEC.md) for the full design document.

## Supported models

| Model family | `--model-type` | Default `--model` |
|---|---|---|
| Qwen3-VL | `qwen3-vl` | `Qwen/Qwen3-VL-2B-Thinking` |

Adding a new model family requires creating a single adapter file in `vlm_attention_viz/models/`.

## Setup

### Python (extraction)

Requires Python 3.10+ and a CUDA GPU. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv pip install -e ".[qwen]"   # install Qwen-specific dependencies
```

### Web (visualization)

```bash
cd web
npm install
```

## Usage

### 1. Extract attention data

```bash
uv run python -m vlm_attention_viz \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --output-dir ./output/sample1
```

Options:
- `--model-type qwen3-vl` -- model family (auto-detected from `--model` when omitted)
- `--model Qwen/Qwen3-VL-2B-Thinking` -- HuggingFace model name
- `--mode prefill` (default) -- attention from the input prompt only
- `--mode generate` -- run generation first, then extract attention over the full sequence
- `--max-new-tokens 256` -- max tokens to generate (generate mode only)

### 2. Launch the visualizer

```bash
cd web
npm run dev
```

By default the dev server serves data from `../output/`. Override with:

```bash
DATA_DIR=../data npm run dev
```

## Project structure

```
vlm_attention_viz/           # Python package
  __init__.py
  __main__.py                # python -m vlm_attention_viz entry point
  cli.py                     # CLI argument parsing
  extract.py                 # Model-agnostic extraction orchestration
  models/
    __init__.py              # Model adapter registry
    base.py                  # Abstract base class for adapters
    qwen3_vl.py              # Qwen3-VL adapter
pyproject.toml               # Python project config + dependencies
web/
  index.html                 # Entry HTML
  src/
    main.ts                  # App entry point
    canvas-renderer.ts       # Canvas heatmap rendering
    data-loader.ts           # Binary data fetching + parsing
    worker.ts                # Web Worker for off-thread parsing
    controls.ts              # UI controls
    color.ts                 # Color scale utilities
    types.ts                 # TypeScript types
  vite.config.ts             # Vite config (includes data-serving plugin)
```
