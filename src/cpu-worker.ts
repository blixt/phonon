// The CPU backend's worker (dist/phonon-cpu.js), in one of two roles. The engine's worker loads
// the module, lays out its memory (cpu.ts) and runs the engine (engine.ts) on it, as the page
// asks (remote.ts). With shared memory it also starts pool workers, more copies of this file,
// each a thread that takes part of every op.
import { unpackModel, type PhononModel } from "./checkpoint";
import { CpuGraph, STACK, type CpuExports } from "./cpu";
import { PhononEngine } from "./engine";
import type { ToWorker } from "./remote";

const scope = self as unknown as {
  postMessage(m: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
  crossOriginIsolated: boolean;
};
const PAGE = 65536;
/** The memory's largest size, in pages: the module allows 1 GiB. */
const MAX_PAGES = 16384;

/** Modules that validate only with relaxed SIMD (i8x16.relaxed_swizzle), or with SIMD128. */
const RELAXED = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 0, 253, 15, 65, 0, 253, 15, 253,
  128, 2, 11,
]);
const SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
export function simdLevel(): "relaxed" | "simd" | null {
  if (WebAssembly.validate(RELAXED)) return "relaxed";
  if (WebAssembly.validate(SIMD)) return "simd";
  return null;
}

/** The module with its memory import made unshared, for a page that is not cross-origin
 *  isolated: one flag byte in the import section. */
function unshare(bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  let at = 8;
  const leb = () => {
    let v = 0;
    let shift = 0;
    for (;;) {
      const b = bytes[at++];
      v |= (b & 0x7f) << shift;
      if (!(b & 0x80)) return v >>> 0;
      shift += 7;
    }
  };
  while (at < bytes.length) {
    const id = bytes[at++];
    const size = leb();
    const end = at + size;
    if (id === 2) {
      const skip = () => {
        const n = leb();
        at += n;
      };
      for (let n = leb(); n > 0; n--) {
        skip();
        skip();
        const kind = bytes[at++];
        if (kind === 2) {
          if (bytes[at] & 2) bytes[at] &= ~2;
          return bytes;
        }
        if (kind === 0) leb();
        else if (kind === 1) {
          at++;
          const flags = bytes[at++];
          leb();
          if (flags & 1) leb();
        } else if (kind === 3) at += 2;
        else throw new Error("Unexpected import");
      }
    }
    at = end;
  }
  return bytes;
}

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

let engine: PhononEngine<unknown, unknown> | null = null;
const cancelled = new Set<number>();

async function load(m: Extract<ToWorker, { type: "load" }>) {
  const level = simdLevel();
  if (!level) throw new Error("This browser has no WebAssembly SIMD.");
  const shared = scope.crossOriginIsolated && m.threads > 1;
  const url = new URL(`./phonon-cpu-${level}.wasm`, import.meta.url);
  let bytes: Uint8Array<ArrayBuffer> = new Uint8Array(await (await fetch(url)).arrayBuffer());
  if (!shared) bytes = unshare(bytes);
  const module = await WebAssembly.compile(bytes);
  const model = unpackModel(m.packed);
  const need = Math.ceil((model.blob.byteLength + 96 * 2 ** 20) / PAGE);
  let memory: WebAssembly.Memory | null = null;
  for (const maximum of [MAX_PAGES, 8192, 6144, 4096]) {
    try {
      memory = new WebAssembly.Memory({ initial: Math.min(need, maximum), maximum, shared } as WebAssembly.MemoryDescriptor);
      break;
    } catch {
      /* a smaller reservation */
    }
  }
  if (!memory) throw new Error("No memory for the CPU engine.");
  const instance = await WebAssembly.instantiate(module, { env: { memory } });
  const x = instance.exports as unknown as CpuExports;
  // Pool threads' stacks first, then everything cpu.ts lays out.
  const base = Math.ceil((x.__heap_base.value as number) / 64) * 64;
  const threads = shared ? await startPool({ module, memory }, import.meta.url, m.threads, base) : 1;
  const g = new CpuGraph({ ...x, memory }, model, { threads, heap: base + (shared ? m.threads - 1 : 0) * STACK });
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
      scope.postMessage({ type: "done", id, cancelled: !out });
    } catch (err) {
      scope.postMessage({ type: "error", id, message: String((err as Error)?.message ?? err) });
    }
    cancelled.delete(id);
  }
};
