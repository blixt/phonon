// The CPU backend's worker (dist/phonon-cpu.js), in one of two roles. The engine's worker loads
// the module, lays out its memory (cpu.ts) and runs the engine (engine.ts) on it, as the page
// asks (remote.ts). With shared memory it also starts pool workers, more copies of this file,
// each a thread that takes part of every op.
import { unpackModel, type PhononModel } from "./checkpoint";
import { CpuGraph, instantiate, simdLevel, STACK, TILES, type CpuExports, type Variant } from "./cpu";
import { MODEL_CACHE } from "./files";
import { PhononEngine } from "./engine";
import type { ToWorker } from "./remote";

const scope = self as unknown as {
  postMessage(m: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
  crossOriginIsolated: boolean;
};
interface Pool {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
}

/** Starts pool threads 1..n-1, thread i's stack ending at `stacks` + i * STACK, and answers how
 *  many threads came up in order (with this one). */
async function startPool(pool: Pool, url: string, n: number, stacks: number): Promise<number> {
  const workers = Array.from({ length: n - 1 }, () => new Worker(url, { type: "module" }));
  const up = await Promise.all(
    workers.map(
      (w, k) =>
        new Promise<boolean>((done) => {
          const timer = setTimeout(() => done(false), 10000);
          w.onmessage = () => {
            clearTimeout(timer);
            done(true);
          };
          w.onerror = () => {
            clearTimeout(timer);
            done(false);
          };
          w.postMessage({ role: "pool", ...pool, index: k + 1, stack: stacks + (k + 1) * STACK });
        }),
    ),
  );
  const count = up.indexOf(false) < 0 ? up.length : up.indexOf(false);
  workers.slice(count).forEach((w) => w.terminate());
  return 1 + count;
}

/** The fastest GEMM tile on this device, timed on first use and kept in the cache. */
async function tile(g: CpuGraph, key: string): Promise<Variant["tile"]> {
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const hit = await cache?.match(key).catch(() => undefined);
  if (hit) {
    const known = (await hit.json()) as Variant["tile"];
    if (TILES.some((t) => JSON.stringify(t) === JSON.stringify(known))) return known;
  }
  const best = g.fastestTile();
  await cache?.put(key, new Response(JSON.stringify(best))).catch(() => {});
  return best;
}

let engine: PhononEngine<unknown, unknown> | null = null;
let graph: CpuGraph | null = null;
const cancelled = new Set<number>();

async function load(m: Extract<ToWorker, { type: "load" }>) {
  const level = simdLevel();
  if (!level) throw new Error("This browser has no WebAssembly SIMD.");
  const shared = scope.crossOriginIsolated && m.threads > 1;
  const url = new URL(`./phonon-cpu-${level}.wasm`, import.meta.url);
  const model = unpackModel(m.packed);
  const wasm = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const { module, memory, exports: x } = await instantiate(wasm, shared, model.blob.byteLength + 64 * 2 ** 20);
  // Pool threads' stacks first, then everything cpu.ts lays out.
  const base = Math.ceil((x.__heap_base.value as number) / 64) * 64;
  const threads = shared ? await startPool({ module, memory }, import.meta.url, m.threads, base) : 1;
  const g = new CpuGraph(x, model, { threads, heap: base + (shared ? m.threads - 1 : 0) * STACK });
  // A new build of the module times the tiles again.
  g.tune({ tile: await tile(g, `${url}?bytes=${wasm.byteLength}&threads=${threads}`), gemv: "int16" });
  graph = g;
  const light: PhononModel = { tensors: model.tensors, pace: model.pace, blob: new Uint8Array() };
  engine = new PhononEngine(g, light, m.align);
  await engine.ready();
  return { threads, simd: level, memory: memory.buffer.byteLength };
}

scope.onmessage = async (e: MessageEvent) => {
  const m = e.data as ToWorker | { role: "pool"; module: WebAssembly.Module; memory: WebAssembly.Memory; index: number; stack: number };
  if ("role" in m) {
    const instance = await WebAssembly.instantiate(m.module, { env: { memory: m.memory } });
    const x = instance.exports as unknown as CpuExports;
    x.__stack_pointer.value = m.stack;
    scope.postMessage({ up: true });
    x.worker(m.index);
    return;
  }
  if (m.type === "load") {
    try {
      scope.postMessage({ type: "loaded", ...(await load(m)) });
    } catch (err) {
      scope.postMessage({ type: "error", message: String((err as Error)?.message ?? err) });
    }
    return;
  }
  if (m.type === "cancel") {
    cancelled.add(m.id);
    return;
  }
  if (m.type === "speak") {
    const { id } = m;
    try {
      const out = await engine!.synthesize(m.ids, m.voice, {
        ...m.options,
        latents: m.latents,
        cancelled: () => cancelled.has(id),
        onRows: (rows) => scope.postMessage({ type: "rows", id, rows }, [rows.buffer as ArrayBuffer]),
        onChunk: (samples) => scope.postMessage({ type: "chunk", id, samples }, [samples.buffer as ArrayBuffer]),
      });
      if (graph?.times.size) {
        console.log(JSON.stringify([...graph.times].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, +v.toFixed(2)])));
        graph.times.clear();
      }
      scope.postMessage({ type: "done", id, cancelled: !out });
    } catch (err) {
      scope.postMessage({ type: "error", id, message: String((err as Error)?.message ?? err) });
    }
    cancelled.delete(id);
  }
};
