export interface PaceWeights {
    speakers: Float32Array;
    speakersProj: Float32Array;
    durationProj: Float32Array;
    latentProj: Float32Array;
    bos: Float32Array;
}
/** The condition at a pace, and the BOS row's input, which carries it too. */
export declare function paceVectors(p: PaceWeights, delta: number): {
    cond: Float32Array;
    bos: Float32Array;
};
export declare function paceForSpeed(speed: number): number;
