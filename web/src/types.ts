export interface TokenMeta {
  id: number;
  text: string;
  type: "text" | "image" | "special";
  grid_pos?: [number, number];
}

export interface ImageGrid {
  rows: number;
  cols: number;
  start_idx: number;
  end_idx: number;
}

export interface Metadata {
  model: string;
  prompt: string;
  image_path: string;
  image_size: [number, number];
  num_layers: number;
  num_heads: number;
  seq_len: number;
  dtype: string;
  tokens: TokenMeta[];
  image_grid: ImageGrid;
}

export type Direction = "source" | "target";
export type AggMode = "mean" | "max";
export type ColorScaleName = "blues" | "viridis" | "hot";

export interface AppState {
  meta: Metadata;
  layer: number;
  head: number; // -1 = aggregated
  aggMode: AggMode;
  direction: Direction;
  colorScale: ColorScaleName;
  opacity: number;
  selectedToken: number | null;
}

export interface TokenRect {
  x: number;
  y: number;
  w: number;
  h: number;
  tokenIdx: number;
}

// Worker messages
export interface WorkerRequest {
  type: "load";
  url: string;
  numHeads: number;
  seqLen: number;
}

export interface WorkerResponse {
  type: "loaded";
  // Full attention data: numHeads * seqLen * seqLen float32 values
  data: Float32Array;
  numHeads: number;
  seqLen: number;
}
