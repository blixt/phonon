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
    private readonly rowsAt;
    private pace;
    constructor(device: GPUDevice, model: PhononModel, { subgroups, align }: {
        subgroups: boolean;
        align: {
            layer: number;
            head: number;
        };
    });
    /** Settles when every shader has compiled. */
    ready(): Promise<void>;
    /** Registers a voice by its speaker latents, [125][512]. */
    addVoice(name: string, latents: Float32Array): void;
    hasVoice(name: string): boolean;
    /** The LM's pace (pace.ts): its condition and BOS row are written into the blob. */
    setPace(delta: number): void;
    /** Text ids to 24 kHz audio in one voice, in chunks as they are made, with the alignment
     *  head's scores of each kept frame over the text as they are made. Null when cancelled. */
    synthesize(ids: number[], voice: string, { seed, words, cancelled, onChunk, onRows, }?: {
        seed?: number;
        /** The text's words: a short text keeps more frames after the end of speech. */
        words?: number;
        cancelled?: () => boolean;
        /** Each chunk of audio as it is made. */
        onChunk?: (samples: Float32Array) => void;
        /** The alignment rows of the next kept frames, one score per text token each. */
        onRows?: (rows: Float32Array) => void;
    }): Promise<Synthesis | null>;
}
/** A text's audio, in the chunks it was made in. */
export interface Synthesis {
    chunks: Float32Array[];
}
