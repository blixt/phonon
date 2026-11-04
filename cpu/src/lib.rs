//! Phonon's CPU backend in WebAssembly: one kernel per op of the engine's graph (src/graph.ts),
//! run from op descriptors that src/cpu.ts writes into this module's memory. There is no
//! allocator: cpu.ts lays out every buffer. SIMD128 is required, relaxed SIMD makes a second
//! build (fused multiply-add), and with shared memory the work of each op is cut over threads.
#![no_std]
#![feature(stdarch_wasm_atomic_wait, wasm_numeric_instr)]
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

mod attn;
mod gemm;
mod gemv;
mod math;
mod ops;
mod pool;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
