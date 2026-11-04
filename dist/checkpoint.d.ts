import { type PaceWeights } from "./pace";
export interface Tensor {
    /** Byte offset of its values: f16, or int8 when it has scales. */
    offset: number;
    count: number;
    /** Byte offset of its f16 scales, one per Q8_GROUP values. */
    scales?: number;
}
export interface PhononModel {
    /** The weights, each tensor's values and scales 256-byte aligned. */
    blob: Uint8Array<ArrayBuffer>;
    tensors: Record<string, Tensor>;
    /** What setPace needs: the condition the LM adds for a pace. */
    pace: PaceWeights;
}
/** The LM's layers: 6 in Phonon 31M, 9 in the 40M models. */
export declare const lmLayers: (m: PhononModel) => number;
/** The safetensors tensors of `buffer`, read as f32. */
export declare function readSafetensors(buffer: ArrayBuffer): Map<string, {
    shape: number[];
    data: Float32Array;
}>;
/** f32 to f16 bits, rounding to nearest even. */
export declare function toF16(values: Float32Array): Uint16Array<ArrayBuffer>;
/** Symmetric int8 with one f16 scale per group, from the values rounded to f16 first:
 *  scale = max |v| / 127, q = v / scale rounded half to even. */
export declare function quantize(values: Float32Array): {
    q: Int8Array;
    scales: Uint16Array;
};
export declare function convertCheckpoint(buffer: ArrayBuffer): PhononModel;
/** A model as one buffer, for the cache: a JSON header (the tensors, the pace vectors'
 *  lengths), the pace vectors and the blob. */
export declare function packModel(model: PhononModel): Uint8Array<ArrayBuffer>;
export declare function unpackModel(buffer: ArrayBuffer): PhononModel;
/** A model without its weights, once a backend holds them: the tensors' places and copies of the
 *  pace's vectors, which are otherwise views into the whole packed file. */
export declare function withoutWeights(m: PhononModel): PhononModel;
/** A catalog voice file: its speaker latents as [125 frames][512]. */
export declare function readVoice(buffer: ArrayBuffer): Float32Array<ArrayBuffer>;
