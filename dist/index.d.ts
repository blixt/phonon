import { type Language, type ModelId } from "./files";
import { type Span, type Word } from "./timing";
/** Where a model runs. */
export interface Backend {
    name: "webgpu" | "cpu";
    /** On the CPU: threads that share each op's work, and the WebAssembly build ("relaxed" for
     *  relaxed SIMD, else "simd"). */
    threads?: number;
    simd?: string;
}
export interface LoadOptions {
    /** files.ts MODELS: Phonon 31M by default. */
    model?: ModelId;
    /** Hears the download in percent. */
    progress?: (percent: number) => void;
    /** "auto" (the default) runs on WebGPU when the browser gives an adapter, else on the CPU. */
    backend?: "auto" | "webgpu" | "cpu";
    /** CPU threads, when the page is cross-origin isolated: by default one per core, at most 8. */
    threads?: number;
}
/** A voice of Gradium's catalog. */
export interface Voice {
    name: string;
    id: string;
    language: Language;
    locale: string;
    gender: string;
    description: string;
}
/** Speech for a text: its 24 kHz samples and when each word, and each token in it, is heard. */
export interface Speech {
    audio: Float32Array;
    sampleRate: number;
    words: Word[];
    tokens: Span[];
}
export interface SpeakOptions {
    /** A catalog voice's name ("Delia") or id. The model's `defaultVoice` when not given. */
    voice?: string;
    /** Speaking speed, from about 0.77 to 1.3. */
    speed?: number;
    /** The sampling seed: the same seed and text give the same audio. */
    seed?: number;
    /** Each chunk of audio as soon as it is made: the first comes within milliseconds. */
    onAudio?: (samples: Float32Array) => void;
    /** Each word with its times, in order, as soon as nothing still to come can change them: once
     *  the token after it has its start, or the text has ended. They are the same as the result's
     *  `words`. With audio made much faster than real time, that is before the word is heard. A
     *  long word, or a slow device, can make a word arrive after its sound has begun. */
    onWord?: (word: Word) => void;
    signal?: AbortSignal;
}
/** The model, ready to speak. */
export declare class Phonon {
    private readonly engine;
    private readonly tokenizer;
    private readonly catalog;
    /** Settles when the GPU device is lost (a reset): load again to go on. */
    readonly lost: Promise<unknown>;
    readonly backend: Backend;
    /** The voice `speak` uses when none is given: the default of the model's first language. */
    readonly defaultVoice: string;
    private readonly lag;
    private constructor();
    /** Loads a model: a download the first time, then the browser's cache. It runs on WebGPU
     *  when the browser has an adapter, else on the CPU, in a worker. */
    static load({ model, progress, backend, threads }?: LoadOptions): Promise<Phonon>;
    /** The catalog's voices in the model's languages. */
    get voices(): readonly Voice[];
    /** The words `speak` gives for a text, in order, each with its tokens' text, as the tokenizer
     *  writes them: a page can lay them out before the times arrive. */
    words(text: string): {
        text: string;
        tokens: string[];
    }[];
    private parts;
    /** Text to speech, in parts that fit the model, joined. */
    speak(text: string, options?: SpeakOptions): Promise<Speech>;
}
export { TEXT_MAX } from "./engine";
export { DEFAULT_VOICE, MODELS, type Language, type ModelId } from "./files";
export type { Span, Word } from "./timing";
