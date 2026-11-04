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
    /** Settles when the text in progress has drained. */
    private turn;
    constructor(device: GPUDevice, model: PhononModel, { subgroups, align }: {
        subgroups: boolean;
        align: {
            layer: number;
            head: number;
        };
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
    synthesize(...args: Parameters<PhononEngine["run"]>): ReturnType<PhononEngine["run"]>;
    private run;
}
/** A text's audio, in the chunks it was made in. */
export interface Synthesis {
    chunks: Float32Array[];
}
