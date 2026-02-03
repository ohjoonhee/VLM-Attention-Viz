import type { ColorScaleName } from "./types";

export type RGBA = [number, number, number, number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(a: RGBA, b: RGBA, t: number): RGBA {
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
    Math.round(lerp(a[3], b[3], t)),
  ];
}

function buildLUT(stops: { t: number; color: RGBA }[]): RGBA[] {
  const lut: RGBA[] = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s].t && t <= stops[s + 1].t) {
        lo = stops[s];
        hi = stops[s + 1];
        break;
      }
    }
    const range = hi.t - lo.t;
    const local = range > 0 ? (t - lo.t) / range : 0;
    lut[i] = lerpColor(lo.color, hi.color, local);
  }
  return lut;
}

const PALETTES: Record<ColorScaleName, { t: number; color: RGBA }[]> = {
  blues: [
    { t: 0, color: [255, 255, 255, 0] },
    { t: 0.25, color: [198, 219, 239, 100] },
    { t: 0.5, color: [107, 174, 214, 180] },
    { t: 0.75, color: [33, 113, 181, 220] },
    { t: 1, color: [8, 48, 107, 255] },
  ],
  viridis: [
    { t: 0, color: [68, 1, 84, 0] },
    { t: 0.25, color: [59, 82, 139, 100] },
    { t: 0.5, color: [33, 145, 140, 180] },
    { t: 0.75, color: [94, 201, 98, 220] },
    { t: 1, color: [253, 231, 37, 255] },
  ],
  hot: [
    { t: 0, color: [0, 0, 0, 0] },
    { t: 0.33, color: [200, 20, 0, 140] },
    { t: 0.66, color: [255, 165, 0, 210] },
    { t: 1, color: [255, 255, 100, 255] },
  ],
};

const lutCache = new Map<ColorScaleName, RGBA[]>();

export function getColorLUT(name: ColorScaleName): RGBA[] {
  let lut = lutCache.get(name);
  if (!lut) {
    lut = buildLUT(PALETTES[name]);
    lutCache.set(name, lut);
  }
  return lut;
}

export function normalizeAndMap(
  values: Float32Array,
  lut: RGBA[],
  opacity: number,
): Uint8ClampedArray {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const out = new Uint8ClampedArray(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    const idx = Math.round(((values[i] - min) / range) * 255);
    const c = lut[idx];
    const o = i * 4;
    out[o] = c[0];
    out[o + 1] = c[1];
    out[o + 2] = c[2];
    out[o + 3] = Math.round(c[3] * opacity);
  }
  return out;
}
