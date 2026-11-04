#!/bin/sh
# Builds the CPU backend's two WebAssembly modules into dist/: SIMD128, and SIMD128 with relaxed
# SIMD. Each imports a memory that can be shared, for threads. Needs Rust with the
# wasm32-unknown-unknown target and rust-src (core is built with atomics).
set -e
cd "$(dirname "$0")"
out=../dist
mkdir -p "$out"
for variant in simd relaxed; do
  features="+simd128,+atomics,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext"
  [ "$variant" = relaxed ] && features="$features,+relaxed-simd"
  RUSTC_BOOTSTRAP=1 CARGO_TARGET_DIR="target/$variant" RUSTFLAGS="-C target-feature=$features -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=1073741824 -C link-arg=--export=__stack_pointer -C link-arg=--export=__heap_base -C link-arg=-zstack-size=262144" \
    cargo build --quiet --release --target wasm32-unknown-unknown -Z build-std=core,panic_abort
  cp "target/$variant/wasm32-unknown-unknown/release/phonon_cpu.wasm" "$out/phonon-cpu-$variant.wasm"
done
ls -l "$out"/phonon-cpu-*.wasm
