// Builds the engine's weights file from Gradium's checkpoint, the same way the converter in
// src/checkpoint.ts did in the browser: the large matrices in int8 with one f16 scale per 32
// values, everything else f16, in the cache's form (packModel). The file is hosted on Hugging
// Face (src/files.ts) so a browser downloads 41 MB, not the 103 MB checkpoint.
// Usage: npm run build-weights -- CHECKPOINT.safetensors OUT.bin
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { convertCheckpoint, packModel } from "../src/checkpoint";

const [input, output] = process.argv.slice(2);
const bytes = readFileSync(input);
const packed = packModel(convertCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
writeFileSync(output, packed);
console.log(`${output}: ${packed.byteLength} bytes, sha256 ${createHash("sha256").update(packed).digest("hex")}`);
