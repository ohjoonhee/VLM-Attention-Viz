import type { AppState, AggMode, Direction, ColorScaleName } from "./types";

export interface ControlCallbacks {
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
  onAggModeChange: (mode: AggMode) => void;
  onDirectionChange: (dir: Direction) => void;
  onColorScaleChange: (scale: ColorScaleName) => void;
  onOpacityChange: (opacity: number) => void;
  onDeselect: () => void;
}

export function initControls(state: AppState, cb: ControlCallbacks) {
  const layerSlider = document.getElementById("layer-slider") as HTMLInputElement;
  const layerLabel = document.getElementById("layer-label")!;
  const layerPrev = document.getElementById("layer-prev")!;
  const layerNext = document.getElementById("layer-next")!;
  const headSelect = document.getElementById("head-select") as HTMLSelectElement;
  const aggToggle = document.getElementById("agg-toggle") as HTMLButtonElement;
  const dirToggle = document.getElementById("dir-toggle") as HTMLButtonElement;
  const colorSelect = document.getElementById("color-select") as HTMLSelectElement;
  const opacitySlider = document.getElementById("opacity-slider") as HTMLInputElement;
  const opacityLabel = document.getElementById("opacity-label")!;

  // Layer
  layerSlider.min = "0";
  layerSlider.max = String(state.meta.num_layers - 1);
  layerSlider.value = String(state.layer);
  layerLabel.textContent = String(state.layer);

  layerSlider.addEventListener("input", () => {
    const v = parseInt(layerSlider.value);
    layerLabel.textContent = String(v);
    cb.onLayerChange(v);
  });
  layerPrev.addEventListener("click", () => {
    if (state.layer > 0) cb.onLayerChange(state.layer - 1);
  });
  layerNext.addEventListener("click", () => {
    if (state.layer < state.meta.num_layers - 1) cb.onLayerChange(state.layer + 1);
  });

  // Head
  headSelect.innerHTML = '<option value="-1">Aggregated</option>';
  for (let i = 0; i < state.meta.num_heads; i++) {
    headSelect.innerHTML += `<option value="${i}">Head ${i}</option>`;
  }
  headSelect.value = String(state.head);
  headSelect.addEventListener("change", () => {
    cb.onHeadChange(parseInt(headSelect.value));
  });

  // Agg mode
  aggToggle.textContent = state.aggMode;
  aggToggle.addEventListener("click", () => {
    const next: AggMode = state.aggMode === "mean" ? "max" : "mean";
    aggToggle.textContent = next;
    cb.onAggModeChange(next);
  });

  // Direction
  dirToggle.textContent = state.direction === "source" ? "Source → Target" : "Target ← Source";
  dirToggle.addEventListener("click", () => {
    const next: Direction = state.direction === "source" ? "target" : "source";
    dirToggle.textContent = next === "source" ? "Source → Target" : "Target ← Source";
    cb.onDirectionChange(next);
  });

  // Color scale
  colorSelect.value = state.colorScale;
  colorSelect.addEventListener("change", () => {
    cb.onColorScaleChange(colorSelect.value as ColorScaleName);
  });

  // Opacity
  opacitySlider.value = String(state.opacity);
  opacityLabel.textContent = `${Math.round(state.opacity * 100)}%`;
  opacitySlider.addEventListener("input", () => {
    const v = parseFloat(opacitySlider.value);
    opacityLabel.textContent = `${Math.round(v * 100)}%`;
    cb.onOpacityChange(v);
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        if (state.layer > 0) cb.onLayerChange(state.layer - 1);
        break;
      case "ArrowRight":
        e.preventDefault();
        if (state.layer < state.meta.num_layers - 1) cb.onLayerChange(state.layer + 1);
        break;
      case "ArrowUp":
        e.preventDefault();
        if (state.head === -1) {
          cb.onHeadChange(0);
        } else if (state.head > 0) {
          cb.onHeadChange(state.head - 1);
        } else {
          // Cycle agg mode when at head 0
          const next: AggMode = state.aggMode === "mean" ? "max" : "mean";
          aggToggle.textContent = next;
          cb.onAggModeChange(next);
        }
        break;
      case "ArrowDown":
        e.preventDefault();
        if (state.head < state.meta.num_heads - 1 && state.head >= 0) {
          cb.onHeadChange(state.head + 1);
        } else {
          cb.onHeadChange(-1);
        }
        break;
      case "Tab":
        e.preventDefault();
        const next: Direction = state.direction === "source" ? "target" : "source";
        dirToggle.textContent = next === "source" ? "Source → Target" : "Target ← Source";
        cb.onDirectionChange(next);
        break;
      case "Escape":
        cb.onDeselect();
        break;
    }
  });

  // Return updater for syncing UI with state
  return {
    syncLayer(layer: number) {
      layerSlider.value = String(layer);
      layerLabel.textContent = String(layer);
    },
    syncHead(head: number) {
      headSelect.value = String(head);
    },
  };
}
