// Web Worker: fetches binary attention data, decodes float16 → float32

function decodeFloat16(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;

  if (exp === 0) {
    // Subnormal or zero
    return (sign ? -1 : 1) * (2 ** -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    // Inf or NaN
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
}

self.onmessage = async (e: MessageEvent) => {
  const { type, url, numHeads, seqLen } = e.data;
  if (type !== "load") return;

  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const totalValues = numHeads * seqLen * seqLen;
  const out = new Float32Array(totalValues);

  for (let i = 0; i < totalValues; i++) {
    out[i] = decodeFloat16(view.getUint16(i * 2, true));
  }

  const msg = { type: "loaded", data: out, numHeads, seqLen };
  (self as unknown as Worker).postMessage(msg, { transfer: [out.buffer] });
};
