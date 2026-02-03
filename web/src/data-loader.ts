import type { AggMode } from "./types";

interface CacheEntry {
  layer: number;
  data: Float32Array; // numHeads * seqLen * seqLen
  accessTime: number;
}

export class DataLoader {
  private worker: Worker;
  private cache: CacheEntry[] = [];
  private maxCache = 3;
  private pending = new Map<number, Promise<Float32Array>>();
  private numHeads: number;
  private seqLen: number;
  private dataPrefix: string;

  constructor(numHeads: number, seqLen: number, dataPrefix: string) {
    this.numHeads = numHeads;
    this.seqLen = seqLen;
    this.dataPrefix = dataPrefix;
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
  }

  private layerUrl(layer: number): string {
    return `${this.dataPrefix}/attn_layer_${String(layer).padStart(2, "0")}.bin`;
  }

  async loadLayer(layer: number): Promise<Float32Array> {
    // Check cache
    const cached = this.cache.find((e) => e.layer === layer);
    if (cached) {
      cached.accessTime = Date.now();
      return cached.data;
    }

    // Check pending
    const pending = this.pending.get(layer);
    if (pending) return pending;

    const promise = new Promise<Float32Array>((resolve, reject) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === "loaded") {
          this.worker.removeEventListener("message", handler);
          const data = e.data.data as Float32Array;
          this.addToCache(layer, data);
          this.pending.delete(layer);
          resolve(data);
        }
      };
      this.worker.addEventListener("message", handler);
      this.worker.addEventListener("error", (err) => {
        this.pending.delete(layer);
        reject(err);
      }, { once: true });

      this.worker.postMessage({
        type: "load",
        url: this.layerUrl(layer),
        numHeads: this.numHeads,
        seqLen: this.seqLen,
      });
    });

    this.pending.set(layer, promise);
    return promise;
  }

  private addToCache(layer: number, data: Float32Array) {
    if (this.cache.length >= this.maxCache) {
      // Evict LRU
      let oldest = 0;
      for (let i = 1; i < this.cache.length; i++) {
        if (this.cache[i].accessTime < this.cache[oldest].accessTime) oldest = i;
      }
      this.cache.splice(oldest, 1);
    }
    this.cache.push({ layer, data, accessTime: Date.now() });
  }

  prefetch(layer: number, numLayers: number) {
    if (layer + 1 < numLayers) this.loadLayer(layer + 1);
    if (layer - 1 >= 0) this.loadLayer(layer - 1);
  }

  /**
   * Get attention vector for a selected token.
   * Returns Float32Array of length seqLen.
   */
  getAttentionVector(
    layerData: Float32Array,
    selectedToken: number,
    direction: "source" | "target",
    head: number, // -1 = aggregated
    aggMode: AggMode,
  ): Float32Array {
    const S = this.seqLen;
    const H = this.numHeads;
    const result = new Float32Array(S);

    if (head >= 0) {
      // Single head
      const headOffset = head * S * S;
      if (direction === "source") {
        // attention[selected, :] — where does this token look?
        const rowStart = headOffset + selectedToken * S;
        result.set(layerData.subarray(rowStart, rowStart + S));
      } else {
        // attention[:, selected] — what looks at this token?
        for (let i = 0; i < S; i++) {
          result[i] = layerData[headOffset + i * S + selectedToken];
        }
      }
    } else {
      // Aggregated across heads
      for (let h = 0; h < H; h++) {
        const headOffset = h * S * S;
        if (direction === "source") {
          const rowStart = headOffset + selectedToken * S;
          for (let j = 0; j < S; j++) {
            const v = layerData[rowStart + j];
            if (aggMode === "mean") {
              result[j] += v / H;
            } else {
              if (v > result[j]) result[j] = v;
            }
          }
        } else {
          for (let i = 0; i < S; i++) {
            const v = layerData[headOffset + i * S + selectedToken];
            if (aggMode === "mean") {
              result[i] += v / H;
            } else {
              if (v > result[i]) result[i] = v;
            }
          }
        }
      }
    }
    return result;
  }
}
