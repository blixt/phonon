import { type Language, type ModelId } from "./files";
import { type Span } from "./timing";
/** A voice of Gradium's catalog. */
export interface Voice {
    name: string;
    id: string;
    language: Language;
    locale: string;
    gender: string;
    description: string;
}
/** Speech for a text: its 24 kHz samples and when each word and token is heard. */
export interface Speech {
    audio: Float32Array;
    sampleRate: number;
    words: Span[];
    tokens: Span[];
}
export interface SpeakOptions {
    /** A catalog voice's name ("Delia") or id. */
    voice?: string;
    /** Speaking speed, from about 0.77 to 1.3. */
    speed?: number;
    /** The sampling seed: the same seed and text give the same audio. */
    seed?: number;
    /** Each chunk of audio as soon as it is made: the first comes within milliseconds. */
    onAudio?: (samples: Float32Array) => void;
    signal?: AbortSignal;
}
/** The model, ready to speak. */
export declare class Phonon {
    private readonly engine;
    private readonly tokenizer;
    private readonly catalog;
    /** Settles when the GPU device is lost (a reset): load again to go on. */
    readonly lost: Promise<unknown>;
    private constructor();
    /** Loads a model (files.ts MODELS, Phonon 31M by default): a download the first time, then
     *  the browser's cache. `progress` hears the download in percent. */
    static load({ model, progress, }?: {
        model?: ModelId;
        progress?: (percent: number) => void;
    }): Promise<Phonon>;
    /** The catalog's voices in the model's languages. */
    get voices(): readonly Voice[];
    /** Text to speech, in parts that fit the model, joined. */
    speak(text: string, options?: SpeakOptions): Promise<Speech>;
}
export { TEXT_MAX } from "./engine";
export { MODELS, type Language, type ModelId } from "./files";
export type { Span } from "./timing";
