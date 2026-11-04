import type { Engine, Synthesis, SynthesizeOptions } from "./engine";
export type ToWorker = {
    type: "load";
    packed: ArrayBuffer;
    align: {
        layer: number;
        head: number;
    };
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
};
export declare class CpuEngine implements Engine {
    /** Threads that take part in each op, this worker's included. */
    readonly threads: number;
    /** The WebAssembly build: "relaxed" (relaxed SIMD) or "simd". */
    readonly simd: string;
    private readonly worker;
    private readonly voices;
    private readonly jobs;
    private next;
    private constructor();
    private dead;
    /** Settles when the worker stops: load again to go on. */
    readonly lost: Promise<Error>;
    /** Starts the worker with a model's packed weights (moved to it, not copied). */
    static start(packed: ArrayBuffer, align: {
        layer: number;
        head: number;
    }, threads: number): Promise<CpuEngine>;
    hasVoice(name: string): boolean;
    synthesize(ids: number[], voice: string, options?: SynthesizeOptions): Promise<Synthesis | null>;
    private hear;
}
