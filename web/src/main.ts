import type { AppState, Metadata, AggMode, Direction, ColorScaleName } from "./types";
import { DataLoader } from "./data-loader";
import { ImageRenderer, TokenRenderer } from "./canvas-renderer";
import { initControls } from "./controls";

const DATA_BASE = "/data";

function getDatasetFromURL(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("dataset") || ".";
}

function setDatasetURL(dataset: string) {
  const url = new URL(window.location.href);
  if (dataset === ".") {
    url.searchParams.delete("dataset");
  } else {
    url.searchParams.set("dataset", dataset);
  }
  window.history.replaceState(null, "", url.toString());
}

function dataPrefix(dataset: string): string {
  return dataset === "." ? DATA_BASE : `${DATA_BASE}/${dataset}`;
}

async function initDatasetPicker(currentDataset: string, onSwitch: (ds: string) => void) {
  const select = document.getElementById("dataset-select") as HTMLSelectElement;
  const resp = await fetch(`${DATA_BASE}/__list__`);
  const datasets: string[] = await resp.json();

  select.innerHTML = "";
  for (const ds of datasets) {
    const opt = document.createElement("option");
    opt.value = ds;
    opt.textContent = ds === "." ? "(root)" : ds;
    if (ds === currentDataset) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener("change", () => {
    onSwitch(select.value);
  });
}

async function loadDataset(dataset: string) {
  const prefix = dataPrefix(dataset);
  const metaResp = await fetch(`${prefix}/meta.json`);
  const meta: Metadata = await metaResp.json();

  const state: AppState = {
    meta,
    layer: 0,
    head: -1,
    aggMode: "mean",
    direction: "source",
    colorScale: "blues",
    opacity: 0.7,
    selectedToken: null,
  };

  const loader = new DataLoader(meta.num_heads, meta.seq_len, prefix);

  // Clear existing canvases in image panel
  const imageContainer = document.getElementById("image-panel")!;
  imageContainer.innerHTML = "";

  const imageRenderer = new ImageRenderer(imageContainer, meta, onTokenSelect);
  await imageRenderer.loadImage(`${prefix}/${meta.image_path}`);

  const tokenCanvas = document.getElementById("token-canvas") as HTMLCanvasElement;
  const tokenPanel = document.getElementById("token-panel")!;
  tokenCanvas.width = tokenPanel.clientWidth;
  tokenCanvas.height = tokenPanel.clientHeight;
  const tokenRenderer = new TokenRenderer(tokenCanvas, meta, onTokenSelect);
  tokenRenderer.draw();

  const statusEl = document.getElementById("status-bar")!;

  function updateStatus() {
    const headStr = state.head === -1 ? `Agg (${state.aggMode})` : `Head ${state.head}`;
    const selStr = state.selectedToken !== null ? `selected: #${state.selectedToken}` : "none selected";
    statusEl.textContent = `Layer ${state.layer}, ${headStr}, ${meta.seq_len} tokens, ${selStr}`;
  }

  let currentLayerData: Float32Array | null = null;

  async function loadCurrentLayer() {
    currentLayerData = await loader.loadLayer(state.layer);
    loader.prefetch(state.layer, meta.num_layers);
    updateView();
  }

  function updateView() {
    if (!currentLayerData || state.selectedToken === null) {
      imageRenderer.drawOverlay(null, state.colorScale, state.opacity);
      tokenRenderer.updateColors(null, state.colorScale, state.opacity);
      tokenRenderer.setSelected(state.selectedToken);
      tokenRenderer.draw();
      updateStatus();
      return;
    }

    const attnVector = loader.getAttentionVector(
      currentLayerData,
      state.selectedToken,
      state.direction,
      state.head,
      state.aggMode,
    );

    imageRenderer.drawOverlay(attnVector, state.colorScale, state.opacity);
    tokenRenderer.setSelected(state.selectedToken);
    tokenRenderer.updateColors(attnVector, state.colorScale, state.opacity);
    updateStatus();
  }

  function onTokenSelect(idx: number) {
    state.selectedToken = idx;
    updateView();
  }

  const controlSync = initControls(state, {
    onLayerChange(layer) {
      state.layer = layer;
      controlSync.syncLayer(layer);
      loadCurrentLayer();
    },
    onHeadChange(head) {
      state.head = head;
      controlSync.syncHead(head);
      updateView();
    },
    onAggModeChange(mode: AggMode) {
      state.aggMode = mode;
      updateView();
    },
    onDirectionChange(dir: Direction) {
      state.direction = dir;
      updateView();
    },
    onColorScaleChange(scale: ColorScaleName) {
      state.colorScale = scale;
      updateView();
    },
    onOpacityChange(opacity: number) {
      state.opacity = opacity;
      updateView();
    },
    onDeselect() {
      state.selectedToken = null;
      updateView();
    },
  });

  window.addEventListener("resize", () => {
    imageRenderer.resize();
    tokenCanvas.width = tokenPanel.clientWidth;
    tokenCanvas.height = tokenPanel.clientHeight;
    tokenRenderer.resize(tokenPanel.clientWidth, tokenPanel.clientHeight);
    updateView();
  });

  updateStatus();
  await loadCurrentLayer();
}

async function main() {
  const currentDataset = getDatasetFromURL();

  await initDatasetPicker(currentDataset, (ds) => {
    setDatasetURL(ds);
    loadDataset(ds);
  });

  await loadDataset(currentDataset);
}

main().catch(console.error);
