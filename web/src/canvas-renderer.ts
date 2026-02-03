import type { Metadata, TokenRect, ColorScaleName } from "./types";
import { getColorLUT, normalizeAndMap } from "./color";

const TOKEN_HEIGHT = 24;
const TOKEN_PAD = 2;
const FONT_SIZE = 12;
export class ImageRenderer {
  private container: HTMLElement;
  private baseCanvas: HTMLCanvasElement;
  private overlayCanvas: HTMLCanvasElement;
  private baseCtx: CanvasRenderingContext2D;
  private overlayCtx: CanvasRenderingContext2D;
  private img: HTMLImageElement | null = null;
  private meta: Metadata;
  private displayW = 0;
  private displayH = 0;
  private onTokenSelect: (idx: number) => void;
  private attnVector: Float32Array | null = null;

  constructor(container: HTMLElement, meta: Metadata, onTokenSelect: (idx: number) => void) {
    this.container = container;
    this.meta = meta;
    this.onTokenSelect = onTokenSelect;

    this.baseCanvas = document.createElement("canvas");
    this.overlayCanvas = document.createElement("canvas");
    this.baseCanvas.style.position = "absolute";
    this.baseCanvas.style.top = "0";
    this.baseCanvas.style.left = "0";
    this.overlayCanvas.style.position = "absolute";
    this.overlayCanvas.style.top = "0";
    this.overlayCanvas.style.left = "0";
    this.overlayCanvas.style.cursor = "crosshair";
    container.style.position = "relative";
    container.appendChild(this.baseCanvas);
    container.appendChild(this.overlayCanvas);

    this.baseCtx = this.baseCanvas.getContext("2d")!;
    this.overlayCtx = this.overlayCanvas.getContext("2d")!;

    this.overlayCanvas.addEventListener("click", (e) => this.handleClick(e));
    this.overlayCanvas.addEventListener("mousemove", (e) => this.handleHover(e));
    this.overlayCanvas.addEventListener("mouseleave", () => this.hideTooltip());
  }

  async loadImage(url: string) {
    return new Promise<void>((resolve) => {
      this.img = new Image();
      this.img.onload = () => {
        this.fitToContainer();
        this.drawBase();
        resolve();
      };
      this.img.src = url;
    });
  }

  private fitToContainer() {
    if (!this.img) return;
    const maxW = this.container.clientWidth;
    const maxH = this.container.clientHeight;
    const aspect = this.img.naturalWidth / this.img.naturalHeight;
    if (maxW / maxH > aspect) {
      this.displayH = maxH;
      this.displayW = Math.round(maxH * aspect);
    } else {
      this.displayW = maxW;
      this.displayH = Math.round(maxW / aspect);
    }
    for (const c of [this.baseCanvas, this.overlayCanvas]) {
      c.width = this.displayW;
      c.height = this.displayH;
      c.style.width = this.displayW + "px";
      c.style.height = this.displayH + "px";
    }
  }

  private drawBase() {
    if (!this.img) return;
    this.baseCtx.drawImage(this.img, 0, 0, this.displayW, this.displayH);
  }

  drawOverlay(attnVector: Float32Array | null, colorScale: ColorScaleName, opacity: number) {
    this.attnVector = attnVector;
    this.overlayCtx.clearRect(0, 0, this.displayW, this.displayH);
    if (!attnVector) return;

    const grid = this.meta.image_grid;
    const cellW = this.displayW / grid.cols;
    const cellH = this.displayH / grid.rows;

    // Extract just the image token attention values
    const imgValues = new Float32Array(grid.rows * grid.cols);
    for (let i = grid.start_idx; i <= grid.end_idx; i++) {
      const tok = this.meta.tokens[i];
      if (tok.type === "image" && tok.grid_pos) {
        const [r, c] = tok.grid_pos;
        imgValues[r * grid.cols + c] = attnVector[i];
      }
    }

    const lut = getColorLUT(colorScale);
    const colors = normalizeAndMap(imgValues, lut, opacity);

    for (let r = 0; r < grid.rows; r++) {
      for (let c = 0; c < grid.cols; c++) {
        const ci = (r * grid.cols + c) * 4;
        this.overlayCtx.fillStyle = `rgba(${colors[ci]},${colors[ci + 1]},${colors[ci + 2]},${colors[ci + 3] / 255})`;
        this.overlayCtx.fillRect(
          Math.floor(c * cellW),
          Math.floor(r * cellH),
          Math.ceil(cellW),
          Math.ceil(cellH),
        );
      }
    }
  }

  private gridPosFromMouse(e: MouseEvent): [number, number] | null {
    const rect = this.overlayCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const grid = this.meta.image_grid;
    const c = Math.floor((x / this.displayW) * grid.cols);
    const r = Math.floor((y / this.displayH) * grid.rows);
    if (r >= 0 && r < grid.rows && c >= 0 && c < grid.cols) return [r, c];
    return null;
  }

  private handleClick(e: MouseEvent) {
    const pos = this.gridPosFromMouse(e);
    if (!pos) return;
    const [r, c] = pos;
    // Find token with this grid pos
    for (const tok of this.meta.tokens) {
      if (tok.grid_pos && tok.grid_pos[0] === r && tok.grid_pos[1] === c) {
        this.onTokenSelect(tok.id);
        return;
      }
    }
  }

  private handleHover(e: MouseEvent) {
    const pos = this.gridPosFromMouse(e);
    if (!pos) {
      this.hideTooltip();
      return;
    }
    const [r, c] = pos;
    let tooltip = document.getElementById("img-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "img-tooltip";
      tooltip.className = "tooltip";
      document.body.appendChild(tooltip);
    }
    let attnVal = "—";
    if (this.attnVector) {
      for (const tok of this.meta.tokens) {
        if (tok.grid_pos && tok.grid_pos[0] === r && tok.grid_pos[1] === c) {
          attnVal = `${(this.attnVector[tok.id] * 100).toFixed(1)}%`;
          break;
        }
      }
    }
    tooltip.textContent = `Patch [${r}, ${c}] attn: ${attnVal}`;
    tooltip.style.display = "block";
    tooltip.style.left = e.clientX + 12 + "px";
    tooltip.style.top = e.clientY + 12 + "px";
  }

  private hideTooltip() {
    const tooltip = document.getElementById("img-tooltip");
    if (tooltip) tooltip.style.display = "none";
  }

  resize() {
    this.fitToContainer();
    this.drawBase();
  }
}

export class TokenRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private meta: Metadata;
  private rects: TokenRect[] = [];
  private collapseImages: boolean;
  private onTokenSelect: (idx: number) => void;
  private selectedToken: number | null = null;
  private hoveredToken: number | null = null;
  private attnColors: Uint8ClampedArray | null = null;
  private scrollOffset = 0;
  private totalHeight = 0;

  constructor(
    canvas: HTMLCanvasElement,
    meta: Metadata,
    onTokenSelect: (idx: number) => void,
    collapseImages = true,
  ) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.meta = meta;
    this.collapseImages = collapseImages;
    this.onTokenSelect = onTokenSelect;

    canvas.addEventListener("click", (e) => this.handleClick(e));
    canvas.addEventListener("mousemove", (e) => this.handleHover(e));
    canvas.addEventListener("mouseleave", () => this.handleLeave());
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.scrollOffset = Math.max(
        0,
        Math.min(this.totalHeight - this.canvas.height, this.scrollOffset + e.deltaY),
      );
      this.draw();
    }, { passive: false });

    this.layoutTokens();
  }

  private layoutTokens() {
    this.rects = [];
    const W = this.canvas.width;
    let x = TOKEN_PAD;
    let y = TOKEN_PAD;
    const grid = this.meta.image_grid;

    this.ctx.font = `${FONT_SIZE}px monospace`;

    for (let i = 0; i < this.meta.tokens.length; i++) {
      const tok = this.meta.tokens[i];

      // Collapsed image block
      if (this.collapseImages && tok.type === "image" && i === grid.start_idx) {
        const label = `[IMAGE ${grid.cols}x${grid.rows}]`;
        const w = this.ctx.measureText(label).width + 8;
        if (x + w > W - TOKEN_PAD) {
          x = TOKEN_PAD;
          y += TOKEN_HEIGHT + TOKEN_PAD;
        }
        // Store one rect for the whole image block
        this.rects.push({ x, y, w, h: TOKEN_HEIGHT, tokenIdx: -1 });
        x += w + TOKEN_PAD;
        // Skip remaining image tokens
        i = grid.end_idx;
        continue;
      }

      if (this.collapseImages && tok.type === "image") continue;

      const label = this.tokenLabel(tok.text, tok.type);
      const w = Math.max(this.ctx.measureText(label).width + 8, 20);
      if (x + w > W - TOKEN_PAD) {
        x = TOKEN_PAD;
        y += TOKEN_HEIGHT + TOKEN_PAD;
      }
      this.rects.push({ x, y, w, h: TOKEN_HEIGHT, tokenIdx: i });
      x += w + TOKEN_PAD;
    }
    this.totalHeight = y + TOKEN_HEIGHT + TOKEN_PAD * 2;
  }

  private tokenLabel(text: string, type: string): string {
    if (type === "special") {
      // Shorten special tokens
      return text.replace(/<\|?/g, "<").replace(/\|?>/g, ">");
    }
    if (text === "\n") return "\\n";
    return text;
  }

  setSelected(idx: number | null) {
    this.selectedToken = idx;
  }

  updateColors(attnVector: Float32Array | null, colorScale: ColorScaleName, opacity: number) {
    if (!attnVector) {
      this.attnColors = null;
    } else {
      const lut = getColorLUT(colorScale);
      this.attnColors = normalizeAndMap(attnVector, lut, opacity);
    }
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = `${FONT_SIZE}px monospace`;
    ctx.textBaseline = "middle";

    const grid = this.meta.image_grid;

    for (const rect of this.rects) {
      const dy = rect.y - this.scrollOffset;
      if (dy + rect.h < 0 || dy > H) continue;

      const tokIdx = rect.tokenIdx;
      const isImageBlock = tokIdx === -1;

      // Background color from attention
      if (this.attnColors && tokIdx >= 0) {
        const ci = tokIdx * 4;
        ctx.fillStyle = `rgba(${this.attnColors[ci]},${this.attnColors[ci + 1]},${this.attnColors[ci + 2]},${this.attnColors[ci + 3] / 255})`;
        ctx.fillRect(rect.x, dy, rect.w, rect.h);
      }

      // Border for selected
      if (tokIdx === this.selectedToken) {
        ctx.strokeStyle = "#ff4444";
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, dy, rect.w, rect.h);
      } else if (tokIdx === this.hoveredToken) {
        ctx.strokeStyle = "#888";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x + 0.5, dy + 0.5, rect.w - 1, rect.h - 1);
      } else {
        ctx.strokeStyle = "#ddd";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(rect.x + 0.5, dy + 0.5, rect.w - 1, rect.h - 1);
      }

      // Text
      let label: string;
      let textColor: string;
      if (isImageBlock) {
        label = `[IMAGE ${grid.cols}x${grid.rows}]`;
        textColor = "#666";
      } else {
        const tok = this.meta.tokens[tokIdx];
        label = this.tokenLabel(tok.text, tok.type);
        textColor = tok.type === "special" ? "#999" : "#222";
      }
      ctx.fillStyle = textColor;
      ctx.fillText(label, rect.x + 4, dy + rect.h / 2);
    }
  }

  private hitTest(e: MouseEvent): TokenRect | null {
    const canvasRect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - canvasRect.left;
    const my = e.clientY - canvasRect.top + this.scrollOffset;
    for (const r of this.rects) {
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) {
        return r;
      }
    }
    return null;
  }

  private handleClick(e: MouseEvent) {
    const rect = this.hitTest(e);
    if (rect && rect.tokenIdx >= 0) {
      this.onTokenSelect(rect.tokenIdx);
    }
  }

  private handleHover(e: MouseEvent) {
    const rect = this.hitTest(e);
    const newHover = rect ? rect.tokenIdx : null;
    if (newHover !== this.hoveredToken) {
      this.hoveredToken = newHover;
      this.draw();
    }

    // Tooltip
    let tooltip = document.getElementById("tok-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "tok-tooltip";
      tooltip.className = "tooltip";
      document.body.appendChild(tooltip);
    }
    if (rect && rect.tokenIdx >= 0) {
      const tok = this.meta.tokens[rect.tokenIdx];
      const attnVal =
        this.attnColors
          ? `${((this.attnColors[rect.tokenIdx * 4 + 3] / 255) * 100).toFixed(1)}%`
          : "—";
      tooltip.textContent = `#${tok.id} "${tok.text}" (${tok.type}) attn: ${attnVal}`;
      tooltip.style.display = "block";
      tooltip.style.left = e.clientX + 12 + "px";
      tooltip.style.top = e.clientY + 12 + "px";
    } else {
      tooltip.style.display = "none";
    }
  }

  private handleLeave() {
    this.hoveredToken = null;
    this.draw();
    const tooltip = document.getElementById("tok-tooltip");
    if (tooltip) tooltip.style.display = "none";
  }

  resize(width: number, height: number) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.layoutTokens();
    this.draw();
  }
}
