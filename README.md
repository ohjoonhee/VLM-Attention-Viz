# AttentionViz

Web-based visualization of attention patterns in the Qwen3-VL vision-language model.

**Two components:**

1. **Python extraction script** -- runs Qwen3-VL inference on an image + prompt, extracts attention weights from every layer, and saves them as per-layer float16 binary files.
2. **Web frontend** -- loads pre-extracted data and provides interactive heatmap visualization across all layers and heads.

See [SPEC.md](SPEC.md) for the full design document.

## Setup

### Python (extraction)

Requires Python 3.10+ and a CUDA GPU. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### Web (visualization)

```bash
cd web
npm install
```

## Usage

### 1. Extract attention data

```bash
uv run python extract.py \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --output-dir ./output/sample1
```

Options:
- `--mode prefill` (default) -- attention from the input prompt only
- `--mode generate` -- run generation first, then extract attention over the full sequence
- `--max-new-tokens 256` -- max tokens to generate (generate mode only)
- `--model Qwen/Qwen3-VL-2B-Thinking` -- model name (default)

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
extract.py           # Attention extraction script
pyproject.toml       # Python dependencies
web/
  index.html         # Entry HTML
  src/
    main.ts          # App entry point
    canvas-renderer.ts  # Canvas heatmap rendering
    data-loader.ts   # Binary data fetching + parsing
    worker.ts        # Web Worker for off-thread parsing
    controls.ts      # UI controls
    color.ts         # Color scale utilities
    types.ts         # TypeScript types
  vite.config.ts     # Vite config (includes data-serving plugin)
```
