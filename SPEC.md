# VLM Attention Viz — Qwen3-VL Attention Visualization

## 1. Overview

A web-based tool for visualizing attention patterns in the **Qwen3-VL-8B-Instruct** vision-language model. The system has two parts:

1. **Python extraction script** — runs Qwen3-VL inference on an image+prompt, extracts all attention weights, and saves them to disk in an efficient binary format.
2. **Static web page** — loads the pre-extracted data and provides interactive visualization of attention patterns across all layers and heads.

### Design Priorities (ordered)

1. **Performance** — fast rendering and low memory usage; the data is large
2. **Clarity** — attention patterns should be immediately readable
3. **Simplicity** — minimal dependencies, easy to run locally

---

## 2. Model Architecture Reference (Qwen3-VL-8B)

| Component | Value |
|---|---|
| LLM layers | 36 |
| Attention heads (query) | 32 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| Hidden size | 4,096 |
| ViT patch size | 16×16 px |
| Spatial merge size | 2 (→ effective 32×32 px per token) |
| Vision start/end token IDs | 151652 / 151653 |
| Image token ID | 151655 |

### Token Sequence Structure

For a single-image input, the token sequence looks like:

```
[system tokens] [vision_start] [img_tok_0] [img_tok_1] ... [img_tok_N] [vision_end] [text tokens...]
```

Image tokens are arranged in a **row-major grid**. For an image resized to W×H pixels, the grid is `(W/32) × (H/32)` tokens. For example, a 672×448 image produces a 21×14 = 294 image token grid.

### Scale Estimate

For a typical input with ~500 total tokens:
- Each attention matrix: 500 × 500 = 250K values
- Per layer (32 heads): 8M values
- All 36 layers: ~288M values
- At float16: **~576 MB**
- At uint8 (quantized): **~288 MB**

---

## 3. Data Extraction Pipeline

### 3.1 Script: `extract.py`

A CLI script that:
1. Loads `Qwen3-VL-8B-Instruct` via HuggingFace Transformers
2. Processes a user-provided image + text prompt
3. Runs a forward pass with `output_attentions=True`
4. Saves attention data and metadata to an output directory

### 3.2 CLI Interface

```bash
python extract.py \
  --image path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --output-dir ./data/sample1 \
  --model Qwen/Qwen3-VL-8B-Instruct   # optional, default
```

### 3.3 Output Format

The output directory contains:

```
sample1/
  meta.json          # metadata + token info
  attn_layer_00.bin  # attention data for layer 0
  attn_layer_01.bin  # attention data for layer 1
  ...
  attn_layer_35.bin
  image.jpg          # copy of the input image (for display)
```

#### `meta.json`

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "prompt": "Describe this image in detail.",
  "image_path": "image.jpg",
  "image_size": [672, 448],
  "num_layers": 36,
  "num_heads": 32,
  "seq_len": 512,
  "dtype": "float16",
  "tokens": [
    { "id": 0, "text": "<|system|>", "type": "special" },
    { "id": 1, "text": "<|vision_start|>", "type": "special" },
    { "id": 2, "text": "[IMG]", "type": "image", "grid_pos": [0, 0] },
    { "id": 3, "text": "[IMG]", "type": "image", "grid_pos": [0, 1] },
    ...
    { "id": 296, "text": "<|vision_end|>", "type": "special" },
    { "id": 297, "text": "Describe", "type": "text" },
    ...
  ],
  "image_grid": {
    "rows": 14,
    "cols": 21,
    "start_idx": 2,
    "end_idx": 295
  }
}
```

- `tokens[].type`: one of `"text"`, `"image"`, `"special"`
- `tokens[].grid_pos`: `[row, col]` — only present for `type: "image"`
- `image_grid`: summary of the image token block for quick access

#### `attn_layer_XX.bin`

Raw binary file containing a flat array of **float16** values, shape `(num_heads, seq_len, seq_len)`, stored in C-contiguous (row-major) order. Each file is `32 × seq_len × seq_len × 2` bytes.

**Why per-layer files**: The web frontend loads only the layer currently being viewed, keeping memory usage bounded. Switching layers triggers a fetch for the next layer file (with prefetching for adjacent layers).

---

## 4. Web Frontend

### 4.1 Tech Stack

- **Vanilla TypeScript** — no framework; a single `index.html` + bundled JS
- **HTML Canvas** — all heatmap rendering via 2D Canvas API (no DOM-per-token)
- **Web Workers** — parse binary attention data off the main thread
- **Bundler**: Vite (for dev server + TypeScript + production build)

Rationale: For rendering thousands of cells with frequent updates, Canvas avoids DOM overhead entirely. No framework reduces bundle size and avoids reactivity overhead for what is fundamentally a rendering-heavy app.

### 4.2 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Controls Bar                                               │
│  [Layer ◂ 0 ▸] [Head: Aggregated ▾] [Agg: Mean | Max]     │
│  [Direction: Source→Target | Target←Source]                  │
│  [Color Scale ▾] [Opacity slider]                           │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│   Image Panel            │   Token Sequence Panel           │
│                          │                                  │
│   ┌──────────────────┐   │   ┌────────────────────────────┐ │
│   │                  │   │   │ <system> <vis_start>       │ │
│   │  Original image  │   │   │ [IMG grid region]          │ │
│   │  with heatmap    │   │   │ <vis_end> Describe this    │ │
│   │  overlay         │   │   │ image in detail .          │ │
│   │                  │   │   │                            │ │
│   └──────────────────┘   │   │ (text tokens with colored  │ │
│                          │   │  backgrounds)              │ │
│                          │   └────────────────────────────┘ │
│                          │                                  │
├──────────────────────────┴──────────────────────────────────┤
│  Status: Layer 12, Head Avg (mean), 512 tokens, src: #297  │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Panels

#### Image Panel (left)

- Displays the input image at its natural aspect ratio
- When a source token is selected, overlays a **heatmap** on the image region
- Each image patch maps to a grid cell; the cell's opacity/color encodes the attention weight from (or to) the selected source token
- Heatmap is rendered on a separate Canvas layer on top of the image
- Supports hover to show exact attention value and grid position

#### Token Sequence Panel (right)

- Displays the full token sequence as a **wrapped text flow**
- Each token is a small rectangular cell rendered on Canvas
- Text tokens show their decoded text, with **background color** encoding attention weight
- Image tokens in this panel are shown as small placeholder cells (labeled `[IMG]`) or collapsed into a single `[IMAGE 21×14]` summary block (toggle)
- Special tokens (`<|vision_start|>`, etc.) are displayed dimmed
- Clicking a token selects it as the source/target for attention display
- The currently selected token has a visible border/highlight

### 4.4 Interaction Model

1. **Click a token** (in either panel) to select it
2. All other tokens update their heatmap color to show the attention relationship:
   - **Source→Target mode**: colors show `attention[selected, :]` — "where does this token look?"
   - **Target←Source mode**: colors show `attention[:, selected]` — "what looks at this token?"
3. **Hover** any token to see a tooltip with: token text, position index, attention weight value
4. **Layer navigation**: slider or `←`/`→` keys to move between layers
5. **Head selection**:
   - Default: "Aggregated" (mean or max across all 32 heads)
   - Dropdown to pick a specific head (0–31)
   - Toggle between mean and max aggregation
6. **Image panel click**: clicking on the image selects the corresponding image patch token

### 4.5 Color Scale

- Default: **sequential blue** (low=transparent, high=saturated blue) — works well overlaid on both images and text
- The scale is normalized **per-view** (0 = min attention in current row/column, 1 = max) so that patterns are always visible regardless of absolute magnitude
- Alternative palette option: viridis, hot

### 4.6 Performance Strategy

| Concern | Solution |
|---|---|
| Large binary files | Load one layer at a time; prefetch adjacent ±1 layers |
| Parsing float16 | Web Worker decodes `.bin` → `Float32Array`, posts back to main thread |
| Rendering many tokens | Single Canvas; batch-draw all token rectangles in one pass |
| Aggregation compute | Compute mean/max across heads in the Worker when "Aggregated" mode is active |
| Heatmap on image | Separate overlay Canvas; only re-render on token selection change |
| Smooth layer switching | Cache decoded attention data for recently visited layers (LRU, max ~3 layers in memory) |
| Color mapping | Pre-compute a 256-entry lookup table; map normalized attention → RGBA |

### 4.7 Keyboard Shortcuts

| Key | Action |
|---|---|
| `←` / `→` | Previous / next layer |
| `↑` / `↓` | Previous / next head (or cycle aggregation modes) |
| `Tab` | Switch direction (Source→Target ↔ Target←Source) |
| `Escape` | Deselect token |

---

## 5. File Structure

```
vlm_attention_viz/
  extract.py              # data extraction script
  pyproject.toml          # Python dependencies (torch, transformers, etc.)
  web/
    index.html            # single HTML entry point
    src/
      main.ts             # entry point, orchestration
      canvas-renderer.ts  # Canvas drawing logic
      data-loader.ts      # fetch + parse binary attention data
      worker.ts           # Web Worker for binary parsing + aggregation
      types.ts            # TypeScript type definitions
      controls.ts         # UI controls (layer/head/direction pickers)
      color.ts            # color scale utilities
    vite.config.ts
    package.json
    tsconfig.json
  data/                   # extracted attention data (gitignored)
    sample1/
      meta.json
      attn_layer_00.bin
      ...
      image.jpg
```

---

## 6. Non-Goals (v1)

- Multi-image or video input
- Live inference from the web UI
- Cross-attention between ViT and LLM (only LLM self-attention is visualized)
- Mobile/responsive layout
- Exporting visualizations as images/video
- Head-importance ranking or pruning analysis

---

## 7. Dependencies

### Python (`pyproject.toml`)

- `torch` (existing)
- `transformers` (HuggingFace, for Qwen3-VL)
- `qwen-vl-utils` (Qwen's image preprocessing utilities)
- `Pillow` (image handling)

### Web (`package.json`)

- `vite` (dev/build)
- `typescript`
- No runtime dependencies