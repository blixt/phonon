import { type PhononModel } from "./checkpoint";
import { TEXT_MAX } from "./lm";
export { TEXT_MAX };
export declare class PhononEngine {
    private readonly g;
    private readonly model;
    private readonly lm;
    private readonly codec;
    /** Each voice's prompt keys and values (lm.ts). */
    private readonly voices;
    private readonly stages;
    private pace;
    constructor(device: GPUDevice, model: PhononModel, { subgroups }: {
        subgroups: boolean;
    });
    /** Settles when every shader has compiled. */
    ready(): Promise<void>;
    /** Registers a voice by its speaker latents, [125][512]. */
    addVoice(name: string, latents: Float32Array): void;
    hasVoice(name: string): boolean;
    /** The LM's pace (pace.ts): its condition and BOS row are written into the blob. */
    setPace(delta: number): void;
    /** Text ids to 24 kHz audio in one voice, in chunks as they are made. Null when cancelled. */
    synthesize(ids: number[], voice: string, { seed, words, cancelled, onChunk, }?: {
        seed?: number;
        /** The text's words: a short text keeps more frames after the end of speech. */
        words?: number;
        cancelled?: () => boolean;
        /** Each chunk of audio as it is made. */
        onChunk?: (samples: Float32Array) => void;
    }): Promise<Synthesis | null>;
    /** The alignment head's scores for `frames` frames over `n` text tokens. */
    private alignment;
}
/** A text's audio, in the chunks it was made in, and the alignment head's scores over its
 *  tokens: one row per 80 ms frame. */
export interface Synthesis {
    chunks: Float32Array[];
    align: Float32Array;
}
