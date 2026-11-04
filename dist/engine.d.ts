import { type PhononModel } from "./checkpoint";
import { type Graph } from "./graph";
import { TEXT_MAX } from "./lm";
export { TEXT_MAX };
export declare class PhononEngine<B, O> implements Engine {
    private readonly g;
    private readonly model;
    private readonly lm;
    private readonly codec;
    /** Each voice's prompt keys and values (lm.ts). */
    private readonly voices;
    private readonly stages;
    private readonly rowsAt;
    private pace;
    /** Settles when the text in progress has drained. */
    private turn;
    /** `g` holds the model's weights. */
    constructor(g: Graph<B, O>, model: PhononModel, align: {
        layer: number;
        head: number;
    });
    /** Settles when every shader has compiled. */
    ready(): Promise<void>;
    /** Registers a voice by its speaker latents, [125][512]. Its prefill uses the LM's caches, so
     *  it runs in the queue, before a text (`synthesize`'s `latents`). */
    private addVoice;
    hasVoice(name: string): boolean;
    /** The LM's pace (pace.ts): its condition and BOS row are written into the blob, so it is set
     *  in the queue, before a text (`synthesize`'s `pace`). */
    private setPace;
    /** Text ids to 24 kHz audio in one voice, in chunks as they are made, with the alignment
     *  head's scores of each kept frame over the text as they are made. Null when cancelled. One
     *  text at a time: a call waits until the one before has drained, cancelled or not. */
    synthesize(ids: number[], voice: string, options?: SynthesizeOptions): Promise<Synthesis | null>;
    private run;
}
/** A text's audio, in the chunks it was made in. */
export interface Synthesis {
    chunks: Float32Array[];
}
export interface SynthesizeOptions {
    seed?: number;
    /** The LM's pace (pace.ts). */
    pace?: number;
    /** The voice's speaker latents, [125][512], the first time it is used. */
    latents?: Float32Array<ArrayBuffer>;
    /** The text's words: a short text keeps more frames after the end of speech. */
    words?: number;
    cancelled?: () => boolean;
    /** Each chunk of audio as it is made. */
    onChunk?: (samples: Float32Array) => void;
    /** The alignment rows of the next kept frames, one score per text token each. */
    onRows?: (rows: Float32Array) => void;
}
/** What the page speaks through: an engine here, or one in a worker (cpu.ts). */
export interface Engine {
    hasVoice(name: string): boolean;
    synthesize(ids: number[], voice: string, options?: SynthesizeOptions): Promise<Synthesis | null>;
}
