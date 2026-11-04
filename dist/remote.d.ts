import type { Engine, Synthesis, SynthesizeOptions } from "./engine";
import type { Align } from "./files";
export type ToWorker = {
    type: "load";
    packed: ArrayBuffer;
    align: Align;
    threads: number;
} | {
    type: "speak";
    id: number;
    ids: number[];
    voice: string;
    options: {
        seed?: number;
        pace?: number;
        words?: number;
    };
    latents?: Float32Array<ArrayBuffer>;
} | {
    type: "cancel";
    id: number;
} | {
    type: "close";
};
export declare class CpuEngine implements Engine {
    /** Threads that take part in each op, this worker's included. */
    readonly threads: number;
    /** The WebAssembly build: "relaxed" (relaxed SIMD) or "simd". */
    readonly simd: "relaxed" | "simd";
    private readonly worker;
    private readonly voices;
    private readonly jobs;
    private next;
    private constructor();
    private dead;
    /** Settles when the worker stops: load again to go on. */
    readonly lost: Promise<Error>;
    /** Starts the worker with a model's packed weights (moved to it, not copied). */
    static start(packed: ArrayBuffer, align: Align, threads: number): Promise<CpuEngine>;
    /** Ends the worker and its pool threads. Texts in progress fail, and so do later ones. */
    close(): Promise<void>;
    hasVoice(name: string): boolean;
    synthesize(ids: number[], voice: string, options?: SynthesizeOptions): Promise<Synthesis | null>;
    private hear;
}
