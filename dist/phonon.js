// src/pace.ts
var H = 512;
function matvec(w, x, rows) {
  const cols = x.length;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += w[r * cols + c] * x[c];
    y[r] = s;
  }
  return y;
}
function paceVectors(p, delta) {
  const half = 64;
  const phases = new Float32Array(2 * half);
  for (let i = 0; i < half; i++) {
    const a = 1e3 * delta / 1e4 ** (i / (half - 1));
    phases[i] = Math.cos(a);
    phases[half + i] = Math.sin(a);
  }
  const cond = matvec(p.speakersProj, p.speakers, H);
  const dur = matvec(p.durationProj, phases, H);
  for (let i = 0; i < H; i++) cond[i] += dur[i];
  const bos = matvec(p.latentProj, p.bos, H);
  for (let i = 0; i < H; i++) bos[i] += cond[i];
  return { cond, bos };
}
var PACE = [
  [0.77, 0.6],
  [0.8, 0.45],
  [0.84, 0.3],
  [0.93, 0.15],
  [1, 0],
  [1.07, -0.15],
  [1.22, -0.3],
  [1.3, -0.45]
];
function paceForSpeed(speed) {
  const at = PACE.findIndex(([s]) => speed <= s);
  if (at === 0) return PACE[0][1];
  if (at < 0) return PACE.at(-1)[1];
  const [s0, d0] = PACE[at - 1];
  const [s1, d1] = PACE[at];
  return d0 + (speed - s0) / (s1 - s0) * (d1 - d0);
}

// src/kernels.ts
var STATE = { pos: 0, frame: 1, end: 2, chunk: 3, text: 4, after: 5, seed: 6, scale: 7 };
var UNIFORMS = {
  layernorm: ["go", "bo"],
  rows: ["rows", "pos0"],
  embedRows: ["eo", "eso", "bo"],
  upsample: ["T", "wo", "wso", "mo", "so"],
  shiftRows: ["src", "count"],
  gemm: ["M", "yo", "ro", "rows", "wo", "bo", "wso"],
  gemmReduce: ["rows", "yo", "ro", "stride", "bo"],
  gemv: ["xo", "yo", "ro", "go", "so", "ho", "wo", "bo", "vo", "lgo", "lbo", "wso"],
  finishStep: ["ewo", "ebo", "ngo", "nbo"]
};
function struct(fields, types = {}) {
  const pad = Array.from({ length: (4 - fields.length % 4) % 4 }, (_, i) => `p${i}`);
  return `struct P { ${[...fields, ...pad].map((f) => `${f}: ${types[f] ?? "u32"}`).join(", ")} }`;
}
var Q8_GROUP = 32;
var { pos: POS, frame: FRAME, end: END, chunk: CHUNK, text: TEXT, after: AFTER, seed: SEED, scale: SCALE } = STATE;
var GATES = {
  none: "",
  step: `if (st[${END}] != 0u && st[${FRAME}] >= st[${END}]) { return; }`,
  chunk: `if (st[${END}] != 0u && st[${CHUNK}] / 16u >= st[${END}]) { return; }`
};
var BLOB = `
fn hf(i: u32) -> f32 { let f = unpack2x16float(wv[i >> 3u][(i >> 1u) & 3u]); return select(f.x, f.y, (i & 1u) == 1u); }
fn h4(i: u32) -> vec4<f32> { let q = wv[i >> 3u]; return select(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), (i & 4u) != 0u); }`;
var int8At = (o, so, e) => `(f32(unpack4xI8(wv[(${o}) / 4u + (${e}) / 16u][((${e}) / 4u) % 4u])[(${e}) % 4u]) * hf(${so} + (${e}) / ${Q8_GROUP}u))`;
var ACT = `
fn gelu(v: f32) -> f32 {
  // erf via Abramowitz-Stegun 7.1.26 is too coarse; use a high-precision rational erf.
  let z = v * 0.70710678118654752;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  let erfz = select(r - 1.0, 1.0 - r, z >= 0.0);
  return 0.5 * v * (1.0 + erfz);
}
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }`;
function actCall(act, v) {
  if (act === "gelu") return `gelu(${v})`;
  if (act === "silu") return `silu(${v})`;
  return v;
}
function residual(res, ri) {
  return {
    none: "",
    add: `v = r[${ri}] + v;`,
    add_y: "v = y[o] + v;",
    gate: `v = r[${ri}] + gv[p.go + n] * v;`,
    gate_y: "v = y[o] + gv[p.go + n] * v;",
    vec: "v = v + hf(p.vo + n);"
  }[res];
}
function subgroupSum(W) {
  if (W === 32) return "fn sumSeg(v: f32, li: u32) -> f32 { return subgroupAdd(v); }";
  let body = "var t = v;";
  for (let s = W / 2; s >= 1; s /= 2) body += ` t += subgroupShuffleXor(t, ${s}u);`;
  return `fn sumSeg(v: f32, li: u32) -> f32 { ${body} return t; }`;
}
function layernorm({ eps, gate }) {
  return `
${struct(UNIFORMS.layernorm)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
var<workgroup> red: array<f32, 128>;
${BLOB}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &y; _ = &wv; _ = &st;
  ${GATES[gate]}
  let base = wid.x * 512u;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = x[base + li + i * 128u]; s += v[i]; }
  red[li] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let mean = red[0] / 512.0;
  workgroupBarrier();
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  red[li] = q; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 512.0 + ${eps});
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 128u;
    y[base + c] = (v[i] - mean) * inv * hf(p.go + c) + hf(p.bo + c);
  }
}`;
}
function cacheParts(cache) {
  const codec = cache.kind === "codec";
  return {
    kv16: !codec,
    /** Row 0's position past `pos0`. */
    at: codec ? `st[${CHUNK}]` : "0u",
    gate: GATES[codec ? "chunk" : "none"],
    slot: (e) => codec ? `((${e}) % ${cache.ring}u)` : `(${e})`,
    window: codec ? cache.window : 0
  };
}
function ropeKV({ theta, cache }) {
  const { kv16, at, gate, slot: slotOf } = cacheParts(cache);
  const slot = slotOf("pos");
  const store = kv16 ? `let kw = (${slot} * 512u + h * 64u) / 2u + t;
  kc[kw] = pack2x16float(vec2<f32>(rot(base + 512u, d0, c0, s0), rot(base + 512u, d1, c1, s1)));
  vc[kw] = pack2x16float(vec2<f32>(qkv[base + 1024u + d0], qkv[base + 1024u + d1]));` : `let kb = ${slot} * 512u + h * 64u;
  kc[kb + d0] = rot(base + 512u, d0, c0, s0); kc[kb + d1] = rot(base + 512u, d1, c1, s1);
  vc[kb + d0] = qkv[base + 1024u + d0]; vc[kb + d1] = qkv[base + 1024u + d1];`;
  return `
${struct(UNIFORMS.rows)}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<${kv16 ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> vc: array<${kv16 ? "u32" : "f32"}>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
fn rot(base: u32, d: u32, c: f32, s: f32) -> f32 {
  if (d < 32u) { return qkv[base + d] * c - qkv[base + d + 32u] * s; }
  return qkv[base + d] * c + qkv[base + d - 32u] * s;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &qkv; _ = &qb; _ = &kc; _ = &vc; _ = &st;
  ${gate}
  let t = gid.x % 32u;
  let h = (gid.x / 32u) % 8u;
  let m = gid.x / 256u;
  if (m >= p.rows) { return; }
  let pos = p.pos0 + ${at} + m;
  let base = m * 1536u + h * 64u;
  let d0 = 2u * t; let d1 = d0 + 1u;
  let a0 = f32(pos) * pow(${theta}, -f32(d0 % 32u) / 32.0);
  let a1 = f32(pos) * pow(${theta}, -f32(d1 % 32u) / 32.0);
  let c0 = cos(a0); let s0 = sin(a0); let c1 = cos(a1); let s1 = sin(a1);
  qb[m * 512u + h * 64u + d0] = rot(base, d0, c0, s0);
  qb[m * 512u + h * 64u + d1] = rot(base, d1, c1, s1);
  ${store}
}`;
}
function attention({ cache }) {
  const { kv16, at, gate, slot, window } = cacheParts(cache);
  const kT = kv16 ? "vec2<u32>" : "vec4<f32>";
  const rd = (b, i) => kv16 ? `kv(${b}[${i}])` : `${b}[${i}]`;
  return `
${struct(UNIFORMS.rows)}
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<${kT}>;
@group(0) @binding(2) var<storage, read> vc: array<${kT}>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> ks: array<array<vec4<f32>, 16>, 16>;
var<workgroup> vs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> sc: array<array<f32, 17>, 16>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn lo(pos: u32) -> u32 { return ${window ? `select(0u, pos + 1u - ${window}u, pos + 1u > ${window}u)` : "0u"}; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &vc; _ = &o; _ = &st;
  ${gate}
  let h = wid.y; let m0 = wid.x * 16u;
  let r = li / 16u; let c = li % 16u;
  let m = m0 + r;
  let last = min(m0 + 15u, p.rows - 1u);
  let base = p.pos0 + ${at};
  let pos = base + m;
  let rowOk = m < p.rows;
  let mylo = lo(pos);
  qs[r][c] = select(vec4<f32>(0.0), qb[m * 128u + h * 16u + c] * 0.125, rowOk);
  var mx = -1e30; var l = 0.0; var acc = vec4<f32>(0.0);
  let k0 = lo(base + m0); let k1 = base + last + 1u;   // keys [k0, k1) cover the 16 rows
  for (var t = k0; t < k1; t += 16u) {
    workgroupBarrier();
    let kp = t + r;   // this thread loads key t + r, dims 4c..4c+3
    if (kp < k1) {
      let s = ${slot("kp")} * 128u + h * 16u + c;
      ks[r][c] = ${rd("kc", "s")}; vs[r][c] = ${rd("vc", "s")};
    } else { ks[r][c] = vec4<f32>(0.0); vs[r][c] = vec4<f32>(0.0); }
    workgroupBarrier();
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qs[r][e], ks[c][e]); }
    let kc2 = t + c;
    sc[r][c] = select(-1e30, d, rowOk && kc2 <= pos && kc2 >= mylo && kc2 < k1);
    workgroupBarrier();
    var tm = mx;
    for (var j = 0u; j < 16u; j++) { tm = max(tm, sc[r][j]); }
    let scale = exp(mx - tm);
    l *= scale; acc *= scale;
    for (var j = 0u; j < 16u; j++) {
      let s = sc[r][j];
      let pj = select(0.0, exp(s - tm), s > -1e29);
      l += pj; acc += pj * vs[j][c];
    }
    mx = tm;
  }
  if (rowOk) { o[m * 128u + h * 16u + c] = acc / l; }
}`;
}
function embedRows({ bos, pad }) {
  return `
${struct(UNIFORMS.embedRows)}
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &ids; _ = &wv; _ = &x;
  let c = gid.x % 512u; let i = gid.x / 512u;
  let id = ids[i];
  var v = 0.0;
  if (id == ${bos}u) { v = hf(p.bo + c); } else if (id != ${pad}u) { v = ${int8At("p.eo", "p.eso", "id * 512u + c")}; }
  x[i * 512u + c] = v;
}`;
}
function noise() {
  return `
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<u32>;
fn pcg(v: u32) -> u32 { var s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) i: u32) {
  _ = &z; _ = &st;
  ${GATES.step}
  let f = st[${FRAME}];
  let a = pcg(st[${SEED}] ^ pcg(f * 64u + i * 2u)); let b = pcg(a ^ 0x9e3779b9u);
  let u1 = (f32(a >> 8u) + 1.0) / 16777217.0; let u2 = f32(b >> 8u) / 16777216.0;
  z[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2) * bitcast<f32>(st[${SCALE}]);
}`;
}
function upsample() {
  return `
${struct(UNIFORMS.upsample)}
@group(0) @binding(0) var<storage, read> lats: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &lats; _ = &wv; _ = &y; _ = &st;
  ${GATES.chunk}
  let c = gid.x % 512u; let tl = gid.x / 512u;
  if (tl >= p.T) { return; }
  let t = st[${CHUNK}] + tl; let ph = t % 16u; let q = t / 16u;
  var acc = 0.0;
  for (var j = 0u; j < 2u; j++) {
    if (q < j) { continue; }
    let src = q - j;
    for (var i = 0u; i < 32u; i++) {
      let v = lats[src * 32u + i] * hf(p.so + i) + hf(p.mo + i);
      acc += ${int8At("p.wo", "p.wso", "((c * 16u + ph) * 2u + j) * 32u + i")} * v;
    }
  }
  y[tl * 512u + c] = acc;
}`;
}
function shiftRows() {
  return `
${struct(UNIFORMS.shiftRows)}
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &b;
  if (gid.x >= p.count) { return; }
  b[gid.x] = b[p.src + gid.x];
}`;
}
function gemm({
  K,
  N,
  act = "none",
  bias = false,
  res = "none",
  load = "linear",
  conv = null,
  splits = 1,
  q8,
  gate
}) {
  const isT = load === "convT";
  const shape = { cin: 0, k: 1, dil: 1, elu: false, stride: 1, ...conv };
  const stride = isT ? shape.stride : 1;
  const cin = shape.cin;
  const elu4 = (v) => `select(exp(${v}) - vec4<f32>(1.0), ${v}, ${v} > vec4<f32>(0.0))`;
  const row = load === "linear" ? `m * ${K}u + k` : load === "conv" ? `(m + (k / ${cin}u) * ${shape.dil}u) * ${cin}u + k % ${cin}u` : `(m + 1u - k / ${cin}u) * ${cin}u + k % ${cin}u`;
  const aLoad = `fn loadA(m: u32, k: u32) -> vec4<f32> {
  if (m >= p.M) { return vec4<f32>(0.0); }
  let v = x[(${row}) / 4u];
  return ${shape.elu && load !== "linear" ? elu4("v") : "v"};
}`;
  const Ks = Math.ceil(K / splits / 32) * 32;
  const wrow = isT ? stride * K : K;
  const bLoad = q8 ? `
    if (li < 128u) {
      let row = li / 2u; let kq = (li % 2u) * 16u;
      let n = tn + row;
      var q = vec4<u32>(0u); var sc = 0.0;
      if (n < ${N}u) {
        let col = wbase + kt + kq;
        q = wv[p.wo / 4u + (n * ${wrow}u + col) / 16u];
        sc = hf(p.wso + n * ${wrow / Q8_GROUP}u + col / ${Q8_GROUP}u);
      }
      let k4 = kq / 4u;
      Bs[row][k4] = vec4<f32>(unpack4xI8(q.x)) * sc; Bs[row][k4 + 1u] = vec4<f32>(unpack4xI8(q.y)) * sc;
      Bs[row][k4 + 2u] = vec4<f32>(unpack4xI8(q.z)) * sc; Bs[row][k4 + 3u] = vec4<f32>(unpack4xI8(q.w)) * sc;
    }` : `
    {
      let row = li / 4u; let kq = (li % 4u) * 8u;
      let n = tn + row;
      var q = vec4<u32>(0u);
      if (n < ${N}u) { q = wv[(p.wo * 2u + n * ${wrow}u + wbase + kt + kq) / 8u]; }
      Bs[row][kq / 4u] = vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y));
      Bs[row][kq / 4u + 1u] = vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w));
    }`;
  const epilogue = splits > 1 ? `y[(sp * p.rows + orow) * ${N}u + n] = v;` : `${bias ? "v += hf(p.bo + n);" : ""}
      v = ${actCall(act, "v")};
      let o = p.yo + orow * ${N}u + n;
      ${residual(res, `p.ro + orow * ${N}u + n`)}
      y[o] = v;`;
  return `
${struct(UNIFORMS.gemm)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
var<workgroup> As: array<array<vec4<f32>, 8>, 64>;
var<workgroup> Bs: array<array<vec4<f32>, 8>, 64>;
${BLOB}
${ACT}
${aLoad}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &st;
  ${GATES[gate]}
  let tm = wid.y * 64u;
  let tn = wid.x * 64u;
  let ph = wid.z / ${splits}u;
  let sp = wid.z % ${splits}u;
  let wbase = ${isT ? `ph * ${2 * cin}u` : "0u"};
  let li = lid.y * 16u + lid.x;
  let k0 = sp * ${Ks}u;
  let k1 = min(${K}u, k0 + ${Ks}u);
  var acc: array<vec4<f32>, 4>;
  for (var kt = k0; kt < k1; kt += 32u) {
    for (var e = 0u; e < 2u; e++) {
      let idx = li + e * 256u;        // 512 vec4 = 64 rows x 8 vec4 (32 k)
      let row = idx / 8u; let k4 = idx % 8u;
      As[row][k4] = loadA(tm + row, kt + k4 * 4u);
    }
    ${bLoad}
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4++) {
      let a0 = As[lid.y * 4u][k4]; let a1 = As[lid.y * 4u + 1u][k4]; let a2 = As[lid.y * 4u + 2u][k4]; let a3 = As[lid.y * 4u + 3u][k4];
      let b0 = Bs[lid.x * 4u][k4]; let b1 = Bs[lid.x * 4u + 1u][k4]; let b2 = Bs[lid.x * 4u + 2u][k4]; let b3 = Bs[lid.x * 4u + 3u][k4];
      acc[0] += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc[1] += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc[2] += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc[3] += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = tm + lid.y * 4u + i;
    if (m >= p.M) { continue; }
    let orow = ${isT ? `(m * ${stride}u + ph)` : "m"};
    for (var j = 0u; j < 4u; j++) {
      let n = tn + lid.x * 4u + j;
      if (n >= ${N}u) { continue; }
      var v = acc[i][j];
      ${epilogue}
    }
  }
}`;
}
function gemmReduce({
  N,
  splits,
  act = "none",
  bias = false,
  res = "none",
  gate
}) {
  return `
${struct(UNIFORMS.gemmReduce)}
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
${BLOB}
${ACT}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &part; _ = &y; _ = &wv; _ = &r; _ = &st;
  ${GATES[gate]}
  let i = gid.x;
  if (i >= p.rows * ${N}u) { return; }
  let n = i % ${N}u; let row = i / ${N}u;
  var v = 0.0;
  for (var s = 0u; s < ${splits}u; s++) { v += part[(s * p.stride + row) * ${N}u + n]; }
  ${bias ? "v += hf(p.bo + n);" : ""}
  v = ${actCall(act, "v")};
  let o = p.yo + i;
  ${residual(res, "p.ro + i")}
  y[o] = v;
}`;
}
function gemvShape(K) {
  const WG = 128;
  const R = 2;
  const U = K / 16;
  const LPR = Math.min(32, U);
  const RPP = 32 / LPR;
  const RS = R * RPP;
  return { WG, R, U, LPR, RPP, RS, rowsPerGroup: WG / 32 * RS };
}
function gemv({
  K,
  N,
  act = "none",
  bias = false,
  res = "none",
  ln = null,
  combine = 0,
  sg
}) {
  const { WG, R, U, LPR, RPP, RS } = gemvShape(K);
  const K4 = K / 4;
  const xo = "p.xo / 4u";
  const pre = ln ? `
  // one pass: sum and sum of squares (f32 is ample for 512 values of this size)
  var s1 = 0.0; var s2 = 0.0;
  for (var i = li; i < 128u; i += ${WG}u) { let v = x[${xo} + i]; s1 += v.x + v.y + v.z + v.w; s2 += dot(v, v); }
  ${sg ? `s1 = subgroupAdd(s1); s2 = subgroupAdd(s2);
  if ((li & 31u) == 0u) { lnr[li >> 5u] = vec2<f32>(s1, s2); }
  workgroupBarrier();` : `lnr[li] = vec2<f32>(s1, s2);
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) { if (li < s) { lnr[li] += lnr[li + s]; } workgroupBarrier(); }`}
  var t = vec2<f32>(0.0);
  ${sg ? `for (var i = 0u; i < ${WG / 32}u; i++) { t += lnr[i]; }` : "t = lnr[0];"}
  let mean = t.x / 512.0;
  let inv = inverseSqrt(max(t.y / 512.0 - mean * mean, 0.0) + ${ln.eps});
  for (var i = li; i < 128u; i += ${WG}u) {
    var o = (x[${xo} + i] - vec4<f32>(mean)) * inv;
    ${ln.affine ? "o = o * h4(p.lgo + i * 4u) + h4(p.lbo + i * 4u);" : ""}
    ${ln.modulate ? "o = o * (vec4<f32>(1.0) + md[p.so / 4u + i]) + md[p.ho / 4u + i];" : ""}
    xs[i] = o;
  }
  workgroupBarrier();` : combine ? `
  // part: [split][head] blocks of 66 floats: 64 dims, max, sum
  for (var i = li; i < 128u; i += ${WG}u) {
    let h = i / 16u; let d = (i % 16u) * 4u;
    var mx = -1e30;
    for (var s = 0u; s < ${combine}u; s++) { mx = max(mx, r[(s * 8u + h) * 66u + 64u]); }
    var num = vec4<f32>(0.0); var den = 0.0;
    for (var s = 0u; s < ${combine}u; s++) {
      let b = (s * 8u + h) * 66u;
      let f = exp(r[b + 64u] - mx);
      num += vec4<f32>(r[b + d], r[b + d + 1u], r[b + d + 2u], r[b + d + 3u]) * f;
      den += r[b + 65u] * f;
    }
    xs[i] = num / den;
  }
  workgroupBarrier();` : `
  for (var i = li; i < ${K4}u; i += ${WG}u) { xs[i] = x[${xo} + i]; }
  workgroupBarrier();`;
  return `${sg ? "enable subgroups;" : ""}
${struct(UNIFORMS.gemv)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> gv: array<f32>;
@group(0) @binding(6) var<storage, read> md: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> st: array<u32>;
var<workgroup> xs: array<vec4<f32>, ${Math.max(K4, 128)}>;
${ln ? `var<workgroup> lnr: array<vec2<f32>, ${sg ? WG / 32 : WG}>;` : ""}
${BLOB}
${ACT}
${sg ? subgroupSum(LPR) : `var<workgroup> r32: array<vec2<f32>, ${WG}>;`}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &gv; _ = &md; _ = &st;
  ${GATES.step}
  let lane = li & 31u;
  let ul = lane % ${LPR}u; let sub = lane / ${LPR}u;
  let n0 = (wid.x * ${WG / 32}u + (li >> 5u)) * ${RS}u + sub;
  ${pre}
  var acc: array<f32, ${R}>;
  for (var j = 0u; j < ${R}u; j++) {
    let n = n0 + j * ${RPP}u;
    if (n < ${N}u) {
      let wb = p.wo / 4u + n * ${U}u;
      for (var i = 0u; i < ${U / LPR}u; i++) {
        let u = ul + i * ${LPR}u;
        let q = wv[wb + u];
        let sc = hf(p.wso + n * ${K / Q8_GROUP}u + (u * 16u) / ${Q8_GROUP}u);
        let a = dot(xs[u * 4u], vec4<f32>(unpack4xI8(q.x))) + dot(xs[u * 4u + 1u], vec4<f32>(unpack4xI8(q.y)))
              + dot(xs[u * 4u + 2u], vec4<f32>(unpack4xI8(q.z))) + dot(xs[u * 4u + 3u], vec4<f32>(unpack4xI8(q.w)));
        acc[j] += a * sc;
      }
    }
  }
  ${sg ? "" : `r32[li] = vec2<f32>(acc[0], acc[1]); workgroupBarrier();
  for (var s = ${LPR / 2}u; s > 0u; s >>= 1u) { if (ul < s) { r32[li] += r32[li + s]; } workgroupBarrier(); }
  `}for (var j = 0u; j < ${R}u; j++) {
    let tot = ${sg ? "sumSeg(acc[j], li)" : "r32[li][j]"};
    let n = n0 + j * ${RPP}u;
    if (ul == 0u && n < ${N}u) {
      var v = tot${bias ? " + hf(p.bo + n)" : ""};
      v = ${actCall(act, "v")};
      let o = p.yo + n;
      ${residual(res, "p.ro + n")}
      y[o] = v;
    }
  }
}`;
}
function finishStep({ eps, threshold }) {
  return `
${struct(UNIFORMS.finishStep)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> lat: array<f32>;
@group(0) @binding(3) var<storage, read_write> lats: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
var<workgroup> ended: u32;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &lat; _ = &lats; _ = &st;
  // st is read_write here, so the gate goes through workgroup memory to stay uniform.
  if (li == 0u) { ended = select(0u, 1u, st[${END}] != 0u && st[${FRAME}] >= st[${END}]); }
  if (workgroupUniformLoad(&ended) == 1u) { return; }
  let a = x[li]; let b2 = x[li + 256u];
  red[li] = a + b2; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / 512.0; workgroupBarrier();
  red[li] = (a - mean) * (a - mean) + (b2 - mean) * (b2 - mean); workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 512.0 + ${eps}); workgroupBarrier();
  let c2 = li + 256u;
  red[li] = ((a - mean) * inv * hf(p.ngo + li) + hf(p.nbo + li)) * hf(p.ewo + li)
          + ((b2 - mean) * inv * hf(p.ngo + c2) + hf(p.nbo + c2)) * hf(p.ewo + c2);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let f = st[${FRAME}];
  if (li < 32u) { lats[f * 32u + li] = lat[li]; }
  let logit = red[0] + hf(p.ebo);
  workgroupBarrier();
  storageBarrier();
  if (li == 0u) {
    st[${POS}] = st[${POS}] + 1u; st[${FRAME}] = f + 1u;
    if (st[${END}] == 0u && logit > ${threshold}) { st[${END}] = f + st[${AFTER}] + 1u; }
  }
}`;
}
function attnSplit({
  splits,
  theta,
  sg,
  align
}) {
  const pack = (v) => `vec2<u32>(pack2x16float(${v}.xy), pack2x16float(${v}.zw))`;
  return `${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${align ? "@group(0) @binding(5) var<storage, read_write> al: array<f32>;" : ""}
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
var<workgroup> qf: array<f32, 64>;
var<workgroup> kf: array<f32, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> pr: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> acc4: array<vec4<f32>, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qkv; _ = &kc; _ = &vc; _ = &part; _ = &st;${align ? " _ = &al;" : ""}
  ${GATES.step}
  let sp = wid.x; let h = wid.y;
  let pos = st[${POS}];
  let hb = h * 64u;
  let n = pos + 1u;
  let per = (n + ${splits - 1}u) / ${splits}u;
  let j0 = sp * per; let j1 = min(n, j0 + per);
  let own = pos >= j0 && pos < j1;
  if (li < 32u) {
    let ang = f32(pos) * pow(${theta}, -f32(li) / 32.0);
    let c = cos(ang); let s = sin(ang);
    let q1 = qkv[hb + li]; let q2 = qkv[hb + li + 32u];
    qf[li] = (q1 * c - q2 * s) * 0.125; qf[li + 32u] = (q2 * c + q1 * s) * 0.125;
    if (own) {
      let k1 = qkv[512u + hb + li]; let k2 = qkv[512u + hb + li + 32u];
      kf[li] = k1 * c - k2 * s; kf[li + 32u] = k2 * c + k1 * s;
    }
  } else if (own && li < 48u) {
    let d = (li - 32u) * 4u;
    let vv = vec4<f32>(qkv[1024u + hb + d], qkv[1025u + hb + d], qkv[1026u + hb + d], qkv[1027u + hb + d]);
    vc[(pos * 512u + hb + d) / 4u] = ${pack("vv")};
  }
  workgroupBarrier();
  if (own && li < 16u) { let k4 = vec4<f32>(kf[li * 4u], kf[li * 4u + 1u], kf[li * 4u + 2u], kf[li * 4u + 3u]); kc[(pos * 512u + hb) / 4u + li] = ${pack("k4")}; }
  storageBarrier();
  workgroupBarrier();
  // scores: a lane group of 8 per key, each lane 8 dims
  let kk = li / 8u; let part8 = li % 8u;
  let qa = vec4<f32>(qf[part8 * 8u], qf[part8 * 8u + 1u], qf[part8 * 8u + 2u], qf[part8 * 8u + 3u]);
  let qb = vec4<f32>(qf[part8 * 8u + 4u], qf[part8 * 8u + 5u], qf[part8 * 8u + 6u], qf[part8 * 8u + 7u]);
  let cnt = j1 - min(j0, j1);
  for (var base = 0u; base < cnt; base += 16u) {
    let j = j0 + base + kk;
    var d = 0.0;
    if (base + kk < cnt) {
      let kb = (j * 512u + hb) / 4u + part8 * 2u;
      d = dot(qa, kv(kc[kb])) + dot(qb, kv(kc[kb + 1u]));
    }
    ${sg ? `d += subgroupShuffleXor(d, 4u); d += subgroupShuffleXor(d, 2u); d += subgroupShuffleXor(d, 1u);
    if (part8 == 0u && base + kk < cnt) { sc[base + kk] = d; }` : `pr[li] = d; workgroupBarrier();
    if (li < 16u && base + li < cnt) { var t = 0.0; for (var e = 0u; e < 8u; e++) { t += pr[li * 8u + e]; } sc[base + li] = t; }
    workgroupBarrier();`}
  }
  workgroupBarrier();
  var mx = -1e30;
  for (var j = li; j < cnt; j += 128u) { mx = max(mx, sc[j]); }
  red[li] = mx; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] = max(red[li], red[li + s]); } workgroupBarrier(); }
  mx = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var j = li; j < cnt; j += 128u) {
    ${align ? `let k = j0 + j - ${align.lo}u;
    if (h == ${align.head}u && j0 + j >= ${align.lo}u && k < st[${TEXT}]) { al[st[${FRAME}] * ${align.width}u + k] = sc[j]; }` : ""}
    let e = exp(sc[j] - mx); sc[j] = e; sum += e;
  }
  red[li] = sum; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  sum = red[0];
  // values: 16 lanes over the head's 64 dims (vec4), 8 key groups
  let dv = li % 16u; let g = li / 16u;
  var a = vec4<f32>(0.0);
  for (var j = g; j < cnt; j += 8u) { a += sc[j] * kv(vc[((j0 + j) * 512u + hb) / 4u + dv]); }
  acc4[li] = a; workgroupBarrier();
  let pb = (sp * 8u + h) * 66u;
  if (li < 16u) {
    var t = vec4<f32>(0.0);
    for (var e = 0u; e < 8u; e++) { t += acc4[e * 16u + li]; }
    part[pb + li * 4u] = t.x; part[pb + li * 4u + 1u] = t.y; part[pb + li * 4u + 2u] = t.z; part[pb + li * 4u + 3u] = t.w;
  }
  if (li == 0u) { part[pb + 64u] = select(mx, -1e30, j1 <= j0); part[pb + 65u] = sum; }
}`;
}
function alignRow({ lo, head }) {
  return `
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> al: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &al; _ = &st;
  let m = st[${POS}] - ${lo}u;
  for (var i = li; i < st[${TEXT}]; i += 256u) {
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qb[m * 128u + ${head * 16}u + e] * 0.125, kv(kc[(${lo}u + i) * 128u + ${head * 16}u + e])); }
    al[i] = d;
  }
}`;
}

// src/checkpoint.ts
var lmLayers = (m) => Object.keys(m.tensors).filter((n) => /^l\d+\.qkv$/.test(n)).length;
function readSafetensors(buffer) {
  const view = new DataView(buffer);
  const headerSize = Number(view.getBigUint64(0, true));
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize))
  );
  const base = 8 + headerSize;
  const out = /* @__PURE__ */ new Map();
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [start, end] = t.data_offsets;
    const bytes = buffer.slice(base + start, base + end);
    let data;
    if (t.dtype === "F32") data = new Float32Array(bytes);
    else if (t.dtype === "BF16") {
      const src = new Uint16Array(bytes);
      const bits = new Uint32Array(src.length);
      for (let i = 0; i < src.length; i++) bits[i] = src[i] << 16;
      data = new Float32Array(bits.buffer);
    } else throw new Error(`Unexpected tensor type ${t.dtype} for ${name}`);
    out.set(name, { shape: t.shape, data });
  }
  return out;
}
function toF16(values) {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    f[0] = values[i];
    const x = u[0];
    const sign = x >>> 16 & 32768;
    const exp = x >>> 23 & 255;
    let mant = x & 8388607;
    if (exp === 255) {
      out[i] = sign | 31744 | (mant ? 512 : 0);
      continue;
    }
    const e = exp - 127 + 15;
    if (e >= 31) {
      out[i] = sign | 31744;
      continue;
    }
    if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
        continue;
      }
      mant |= 8388608;
      const shift = 14 - e;
      let half2 = mant >> shift;
      const rest2 = mant & (1 << shift) - 1;
      const mid = 1 << shift - 1;
      if (rest2 > mid || rest2 === mid && half2 & 1) half2++;
      out[i] = sign | half2;
      continue;
    }
    let half = e << 10 | mant >> 13;
    const rest = mant & 8191;
    if (rest > 4096 || rest === 4096 && half & 1) half++;
    out[i] = sign | half;
  }
  return out;
}
function unpackModel(buffer) {
  const length = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
  let o = Math.ceil((4 + length) / 4) * 4;
  const pace = {};
  for (const [k, n] of header.pace) {
    pace[k] = new Float32Array(buffer, o, n);
    o += n * 4;
  }
  return { tensors: header.tensors, pace, blob: new Uint8Array(buffer, o) };
}
function readVoice(buffer) {
  const t = readSafetensors(buffer).get("speaker_wavs");
  if (!t) throw new Error("The voice file has no speaker_wavs");
  const [, channels, frames] = t.shape;
  const out = new Float32Array(frames * channels);
  for (let c = 0; c < channels; c++)
    for (let f = 0; f < frames; f++) out[f * channels + c] = t.data[c * frames + f];
  return out;
}

// src/files.ts
var hubFile = (repo, revision, path) => `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
var MODEL_CACHE = "phonon-webgpu";
var DEFAULT_VOICE = { en: "Delia", fr: "Augustin", de: "Mats", es: "Marcos", pt: "Davi" };
var FORTY_REPO = "blixt/phonon-40m-webgpu";
var FORTY_REVISION = "dde41aa27ed6bed7c874621d8b17e77dfece21bf";
var fortyM = (name, revision, languages, align) => ({
  source: { repo: `gradium/${name}`, revision },
  weights: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.int8.bin`),
  tokenizer: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.tokenizer.5.0.0.json`),
  languages,
  license: "CC-BY-NC-4.0",
  align
});
var MODELS = {
  "phonon-31m-en": {
    source: { repo: "gradium/phonon-31m-en", revision: "92741c5b07d085ff840e26fea8ce43110b87852f" },
    weights: hubFile("blixt/phonon-31m-en-webgpu", "f4635a6dddaa6072451e0155c03c97fbe889b6ec", "phonon-31m-en.int8.bin"),
    tokenizer: "tokenizer.5.0.0.json",
    languages: ["en"],
    license: "Apache-2.0",
    align: { layer: 3, head: 0 }
  },
  "phonon-40m-en": fortyM("phonon-40m-en", "bc7e8ae7e543a7a3e4604753e42fbda76ddf0781", ["en"], { layer: 6, head: 0 }),
  "phonon-40m-fr": fortyM("phonon-40m-fr", "d91086970206b88efd2c88ca971d0628b5968f0f", ["fr"], { layer: 4, head: 1 }),
  "phonon-40m-de": fortyM("phonon-40m-de", "79cfc7b47fcfc0b666657b981eb366415023a862", ["de"], { layer: 3, head: 5 }),
  "phonon-40m-es": fortyM("phonon-40m-es", "e5e052b26e31abd5918ade71d3135e69d51a66f8", ["es"], { layer: 4, head: 6 }),
  "phonon-40m-pt": fortyM("phonon-40m-pt", "4cbfd742a1555d60eb5399c6d296e9f2e07389e9", ["pt"], { layer: 4, head: 2 }),
  "phonon-40m-multilingual": fortyM("phonon-40m-multilingual", "1537ce46c64f42903e0d608b132ccb073dac46d2", ["en", "fr", "de", "es", "pt"], { layer: 4, head: 1 })
};
var tokenizerUrl = (m) => m.tokenizer.startsWith("https:") ? m.tokenizer : hubFile(m.source.repo, m.source.revision, m.tokenizer);
var VOICES_REPO = "gradium/phonon-voices";
var VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
var CATALOG_URL = hubFile(VOICES_REPO, VOICES_REVISION, "voices.jsonl");
var voiceUrl = (id) => hubFile(VOICES_REPO, VOICES_REVISION, `voices/${id}.safetensors`);

// src/graph.ts
var STORAGE = 128;
var UNIFORM = 64;
var COPY_DST = 8;
var COPY_SRC = 4;
var COMPUTE = 4;
var SCRATCH = 7.5 * 2 ** 20;
function encode(enc, ops) {
  const pass = enc.beginComputePass();
  for (const o of ops) {
    pass.setPipeline(o.pipe.pipeline);
    pass.setBindGroup(0, o.bindGroup);
    pass.dispatchWorkgroups(...o.groups);
  }
  pass.end();
}
function layoutOf(device, code) {
  const entries = [
    ...code.matchAll(/@binding\((\d+)\) var<(storage|uniform)(?:, (read_write|read))?>/g)
  ].map(([, b, kind, access]) => ({
    binding: Number(b),
    visibility: COMPUTE,
    buffer: {
      type: kind === "uniform" ? "uniform" : access === "read_write" ? "storage" : "read-only-storage"
    }
  }));
  return device.createBindGroupLayout({ entries });
}
function residual2(res, rbuf, ybuf) {
  return rbuf === ybuf ? { res: `${res}_y`, rbuf: void 0 } : { res, rbuf };
}
var Graph = class {
  device;
  weights;
  /** The LM's and codec's state, slots as kernels.ts STATE names them. */
  state;
  /** Bound where a kernel's binding goes unused. */
  zero;
  /** 32-lane subgroups for the GEMV sums. */
  sg;
  tensors;
  pipes = /* @__PURE__ */ new Map();
  compiling = [];
  scratchBuf = null;
  constructor(device, model, sg) {
    this.device = device;
    this.tensors = model.tensors;
    this.sg = sg;
    this.weights = this.buffer(model.blob.byteLength, model.blob);
    this.state = this.buffer(32);
    this.zero = this.buffer(256);
  }
  buffer(size, data) {
    const b = this.device.createBuffer({
      size: Math.max(256, Math.ceil(size / 256) * 256),
      usage: STORAGE | COPY_DST | COPY_SRC
    });
    if (data) this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }
  /** A uniform of a kernel's `fields` (kernels.ts UNIFORMS), 0 where `values` has none. */
  u(fields, values) {
    const given = values;
    const words = new Uint32Array(Math.max(4, Math.ceil(fields.length / 4) * 4));
    fields.forEach((f, i) => words[i] = given[f] ?? 0);
    const b = this.device.createBuffer({ size: words.byteLength, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(b, 0, words);
    return b;
  }
  tensor(name) {
    const t = this.tensors[name];
    if (!t) throw new Error(`No tensor ${name}`);
    return t;
  }
  /** An f16 tensor's offset in halves. */
  half(name) {
    const t = this.tensor(name);
    if (t.scales !== void 0) throw new Error(`${name} is int8`);
    return t.offset / 2;
  }
  /** A tensor's offset in u32 words, and its scales' offset in halves (int8 only). */
  words(name) {
    const t = this.tensor(name);
    return {
      wo: t.offset / 4,
      wso: t.scales === void 0 ? 0 : t.scales / 2,
      q8: t.scales !== void 0
    };
  }
  /** An int8 tensor's offsets, for the kernels that only read int8. */
  q8(name) {
    const w = this.words(name);
    if (!w.q8) throw new Error(`${name} is not int8`);
    return w;
  }
  /** Settles when every pipeline built so far has compiled. */
  async ready() {
    while (this.compiling.length) await Promise.all(this.compiling.splice(0));
  }
  op(code, bindings, groups) {
    let pipe = this.pipes.get(code);
    if (!pipe) {
      const layout = layoutOf(this.device, code);
      const p = { layout, pipeline: null };
      this.compiling.push(
        this.device.createComputePipelineAsync({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" }
        }).then((pipeline) => {
          p.pipeline = pipeline;
        })
      );
      this.pipes.set(code, p);
      pipe = p;
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipe.layout,
      entries: bindings.map((b, i) => ({
        binding: i,
        resource: b instanceof GPUBuffer ? { buffer: b } : b
      }))
    });
    return { pipe, bindGroup, groups };
  }
  /** Split-K partials. Dispatches run in order, so every GEMM's partials share one buffer. */
  scratch(bytes) {
    if (bytes > SCRATCH) throw new Error(`Split-K partials of ${bytes} bytes`);
    this.scratchBuf ??= this.buffer(SCRATCH);
    return { buffer: this.scratchBuf, offset: 0, size: Math.ceil(bytes / 256) * 256 };
  }
  /** y = epilogue(W x) for one row (x at element `xo`), optionally normalizing x first.
   *  A step's work, so it returns at once past the end. */
  gemv(x, wname, ybuf, o) {
    const { ln, combine } = o;
    const { res, rbuf } = residual2(o.res ?? "none", o.rbuf, ybuf);
    const w = this.q8(wname);
    const code = gemv({
      K: o.K,
      N: o.N,
      act: o.act,
      bias: !!o.bias,
      res,
      combine: combine?.splits,
      ln: ln && { affine: !!ln.w, modulate: !!ln.mod, eps: ln.eps },
      sg: this.sg
    });
    const u = this.u(UNIFORMS.gemv, {
      xo: o.xo,
      go: o.go,
      so: ln?.so,
      ho: ln?.ho,
      wo: w.wo,
      bo: o.bias ? this.half(o.bias) : 0,
      vo: o.vname ? this.half(o.vname) : 0,
      lgo: ln?.w ? this.half(ln.w) : 0,
      lbo: ln?.b ? this.half(ln.b) : 0,
      wso: w.wso
    });
    return this.op(
      code,
      [
        x,
        this.weights,
        ybuf,
        combine?.parts ?? rbuf ?? this.zero,
        u,
        o.gbuf ?? this.zero,
        ln?.mod ?? this.zero,
        this.state
      ],
      [Math.ceil(o.N / gemvShape(o.K).rowsPerGroup), 1, 1]
    );
  }
  /** Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. */
  gemm(xbuf, wname, ybuf, o) {
    const { M, K: k, N: n, load = "linear", conv, yo = 0, ro = 0, gate } = o;
    const { res, rbuf } = residual2(o.res ?? "none", o.rbuf, ybuf);
    const w = this.words(wname);
    const phases = load === "convT" ? conv.stride : 1;
    const tiles = Math.ceil(n / 64) * Math.min(4, Math.ceil(M / 64)) * phases;
    const splits = Math.max(1, Math.min(Math.floor(160 / tiles), Math.floor(k / 128)));
    const groups = [
      Math.ceil(n / 64),
      Math.ceil(M / 64),
      phases * splits
    ];
    const bo = o.bias ? this.half(o.bias) : 0;
    const shape = { K: k, N: n, load, conv, q8: w.q8, gate };
    const P = UNIFORMS.gemm;
    if (splits === 1) {
      const code = gemm({ ...shape, act: o.act, bias: !!o.bias, res });
      return [
        this.op(
          code,
          [
            xbuf,
            this.weights,
            ybuf,
            rbuf ?? this.zero,
            this.u(P, { M, yo, ro, wo: w.wo, bo, wso: w.wso }),
            this.state
          ],
          groups
        )
      ];
    }
    const rowsOut = M * phases;
    const part = this.scratch(splits * rowsOut * n * 4);
    return [
      this.op(
        gemm({ ...shape, splits }),
        [
          xbuf,
          this.weights,
          part,
          this.zero,
          this.u(P, { M, rows: rowsOut, wo: w.wo, wso: w.wso }),
          this.state
        ],
        groups
      ),
      this.op(
        gemmReduce({ N: n, splits, act: o.act, bias: !!o.bias, res, gate }),
        [
          part,
          ybuf,
          this.weights,
          rbuf ?? this.zero,
          this.u(UNIFORMS.gemmReduce, { rows: rowsOut, yo, ro, stride: rowsOut, bo }),
          this.state
        ],
        [Math.ceil(rowsOut * n / 256), 1, 1]
      )
    ];
  }
  layernorm(x, y, rows, w, b, eps, gate) {
    return this.op(
      layernorm({ eps, gate }),
      [
        x,
        y,
        this.weights,
        this.u(UNIFORMS.layernorm, { go: this.half(w), bo: this.half(b) }),
        this.state
      ],
      [rows, 1, 1]
    );
  }
};

// src/lm.ts
var H2 = 512;
var FF = 2048;
var PROMPT = 125;
var S_MAX = 1024;
var F_MAX = 768;
var TEXT_MAX = 240;
var PREFILL = [64, 128, 192, 256];
var BOS = 4294967294;
var PAD = 4294967295;
var LN_EPS = "1e-5";
var FLOW_LN_EPS = "1e-6";
var ROPE_THETA = "10000.0";
var EOS_THRESHOLD = "4.0";
var SPLITS = 8;
var KV_BYTES = 2;
function rowBuffers(g, rows) {
  const r = (n) => g.buffer(rows * n * 4);
  return { x: r(H2), h: r(H2), qkv: r(3 * H2), q: r(H2), att: r(H2), ff: r(FF) };
}
function layerRows(g, pre, o) {
  const { B, rows, cache } = o;
  const gate = cache.kind === "codec" ? "chunk" : "none";
  const at = g.u(UNIFORMS.rows, { rows, pos0: o.pos0 });
  return [
    g.layernorm(B.x, B.h, rows, `${pre}.ln1.w`, `${pre}.ln1.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.qkv`, B.qkv, { M: rows, K: H2, N: 3 * H2, gate }),
    g.op(
      ropeKV({ theta: ROPE_THETA, cache }),
      [B.qkv, B.q, o.kc, o.vc, g.state, at],
      [rows, 1, 1]
    ),
    g.op(
      attention({ cache }),
      [B.q, o.kc, o.vc, B.att, g.state, at],
      [Math.ceil(rows / 16), 8, 1]
    ),
    ...g.gemm(B.att, `${pre}.o`, B.x, { M: rows, K: H2, N: H2, res: "add", rbuf: B.x, gate }),
    g.layernorm(B.x, B.h, rows, `${pre}.ln2.w`, `${pre}.ln2.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.fc1`, B.ff, { M: rows, K: H2, N: FF, act: "gelu", gate }),
    ...g.gemm(B.ff, `${pre}.fc2`, o.out?.buf ?? B.x, {
      M: rows,
      K: FF,
      N: H2,
      res: "add",
      rbuf: B.x,
      yo: o.out?.yo ?? 0,
      gate
    })
  ];
}
var LanguageModel = class {
  /** The latents made so far, one row of 32 per frame. */
  lats;
  /** One step: a latent, then the next position. */
  step;
  /** The alignment head's scores over the text, one row of TEXT_MAX per frame. */
  align;
  g;
  rows;
  kc;
  vc;
  ids;
  /** The BOS row's output on its way to row 0, where the step's tail reads. */
  bos;
  voiceIn;
  voiceOps;
  prefills;
  tail;
  cond;
  ada;
  fh;
  ft;
  z;
  lat;
  parts;
  depth;
  /** The head that follows the text as it speaks (files.ts), its scores kept for word times. */
  alignLayer;
  alignAt;
  constructor(g, depth, align) {
    this.g = g;
    this.depth = depth;
    this.alignLayer = align.layer;
    this.alignAt = { lo: PROMPT, width: TEXT_MAX, head: align.head };
    const rows = PREFILL.at(-1);
    this.rows = rowBuffers(g, rows);
    this.kc = Array.from({ length: depth }, () => g.buffer(S_MAX * H2 * KV_BYTES));
    this.vc = Array.from({ length: depth }, () => g.buffer(S_MAX * H2 * KV_BYTES));
    this.cond = g.buffer(H2 * 4);
    this.ada = g.buffer(10240 * 4);
    this.fh = g.buffer(H2 * 4);
    this.ft = g.buffer(H2 * 4);
    this.z = g.buffer(32 * 4);
    this.lat = g.buffer(32 * 4);
    this.lats = g.buffer(F_MAX * 32 * 4);
    this.ids = g.buffer(rows * 4);
    this.bos = g.buffer(H2 * 4);
    this.parts = g.buffer(SPLITS * 8 * 66 * 4);
    this.voiceIn = g.buffer(PROMPT * H2 * 4);
    this.align = g.buffer(F_MAX * TEXT_MAX * 4);
    this.tail = this.tailOps();
    this.step = [...Array.from({ length: depth }, (_, i) => this.stepLayer(i)).flat(), ...this.tail];
    this.voiceOps = [
      ...g.gemm(this.voiceIn, "prompt_proj", this.rows.x, { M: PROMPT, K: H2, N: H2, gate: "none" }),
      ...this.layers(PROMPT, 0)
    ];
    const em = g.q8("embed");
    this.prefills = new Map(
      PREFILL.map((n) => [
        n,
        [
          g.op(
            embedRows({ bos: BOS, pad: PAD }),
            [
              this.ids,
              g.weights,
              this.rows.x,
              g.u(UNIFORMS.embedRows, { eo: em.wo, eso: em.wso, bo: g.half("bos_embed") })
            ],
            [n * H2 / 256, 1, 1]
          ),
          ...this.layers(n, PROMPT, true)
        ]
      ])
    );
  }
  /** The layers over `rows` rows from position `pos0`, in the prefill's way, and for a text's
   *  prefill frame 0's alignment row. */
  layers(rows, pos0, text = false) {
    const { g } = this;
    return Array.from({ length: this.depth }, (_, i) => [
      ...layerRows(g, `l${i}`, {
        B: this.rows,
        rows,
        kc: this.kc[i],
        vc: this.vc[i],
        cache: { kind: "lm" },
        pos0
      }),
      ...text && i === this.alignLayer ? [g.op(alignRow(this.alignAt), [this.rows.q, this.kc[i], this.align, g.state], [1, 1, 1])] : []
    ]).flat();
  }
  /** A voice's prompt rows through every layer, its keys and values kept in a new buffer. */
  voice(latents) {
    const { device } = this.g;
    const bytes = PROMPT * H2 * KV_BYTES;
    const kv = this.g.buffer(2 * this.depth * bytes);
    device.queue.writeBuffer(
      this.voiceIn,
      0,
      latents.buffer,
      latents.byteOffset,
      latents.byteLength
    );
    const enc = device.createCommandEncoder();
    encode(enc, this.voiceOps);
    for (let i = 0; i < this.depth; i++) {
      enc.copyBufferToBuffer(this.kc[i], 0, kv, 2 * i * bytes, bytes);
      enc.copyBufferToBuffer(this.vc[i], 0, kv, (2 * i + 1) * bytes, bytes);
    }
    device.queue.submit([enc.finish()]);
    return kv;
  }
  /** Sets up a text in a voice (its keys and values from `voice`) and encodes its prefill, which
   *  makes frame 0. The state's position starts on BOS, and frame 0's step moves it past. */
  start(enc, ids, voice, seed, temperature, framesAfterEos) {
    const { device } = this.g;
    const n = PREFILL.find((p) => p > ids.length);
    const padded = new Uint32Array(n).fill(PAD);
    padded.set(ids);
    padded[ids.length] = BOS;
    device.queue.writeBuffer(this.ids, 0, padded);
    const state = new Uint32Array(8);
    state[STATE.pos] = PROMPT + ids.length;
    state[STATE.after] = framesAfterEos;
    state[STATE.text] = ids.length;
    state[STATE.seed] = seed >>> 0;
    new Float32Array(state.buffer)[STATE.scale] = Math.sqrt(temperature);
    device.queue.writeBuffer(this.g.state, 0, state);
    const bytes = PROMPT * H2 * KV_BYTES;
    for (let i = 0; i < this.depth; i++) {
      enc.copyBufferToBuffer(voice, 2 * i * bytes, this.kc[i], 0, bytes);
      enc.copyBufferToBuffer(voice, (2 * i + 1) * bytes, this.vc[i], 0, bytes);
    }
    encode(enc, this.prefills.get(n));
    enc.copyBufferToBuffer(this.rows.x, ids.length * H2 * 4, this.bos, 0, H2 * 4);
    enc.copyBufferToBuffer(this.bos, 0, this.rows.x, 0, H2 * 4);
    encode(enc, this.tail);
  }
  /** A step's layer: one row at the state's position, attention split over the keys. */
  stepLayer(i) {
    const { g } = this;
    const B = this.rows;
    const pre = `l${i}`;
    return [
      g.gemv(B.x, `${pre}.qkv`, B.qkv, {
        K: H2,
        N: 3 * H2,
        ln: { w: `${pre}.ln1.w`, b: `${pre}.ln1.b`, eps: LN_EPS }
      }),
      i === this.alignLayer ? g.op(
        attnSplit({ splits: SPLITS, theta: ROPE_THETA, sg: g.sg, align: this.alignAt }),
        [B.qkv, this.kc[i], this.vc[i], this.parts, g.state, this.align],
        [SPLITS, 8, 1]
      ) : g.op(
        attnSplit({ splits: SPLITS, theta: ROPE_THETA, sg: g.sg }),
        [B.qkv, this.kc[i], this.vc[i], this.parts, g.state],
        [SPLITS, 8, 1]
      ),
      g.gemv(g.zero, `${pre}.o`, B.x, {
        K: H2,
        N: H2,
        res: "add",
        rbuf: B.x,
        combine: { splits: SPLITS, parts: this.parts }
      }),
      g.gemv(B.x, `${pre}.fc1`, B.ff, {
        K: H2,
        N: FF,
        act: "gelu",
        ln: { w: `${pre}.ln2.w`, b: `${pre}.ln2.b`, eps: LN_EPS }
      }),
      g.gemv(B.ff, `${pre}.fc2`, B.x, { K: FF, N: H2, res: "add", rbuf: B.x })
    ];
  }
  /** From the final hidden state (row 0 of x): the flow head samples the next latent, then its
   *  EOS logit, the next input row and the next position. */
  tailOps() {
    const { g } = this;
    const x = this.rows.x;
    const blocks = Array.from({ length: 6 }, (_, i) => {
      const a = i * 1536;
      return [
        g.gemv(this.fh, `fh${i}.fc1.w`, this.ft, {
          K: H2,
          N: H2,
          bias: `fh${i}.fc1.b`,
          act: "silu",
          ln: {
            w: `fh${i}.ln.w`,
            b: `fh${i}.ln.b`,
            eps: FLOW_LN_EPS,
            mod: this.ada,
            ho: a,
            so: a + 512
          }
        }),
        g.gemv(this.ft, `fh${i}.fc2.w`, this.fh, {
          K: H2,
          N: H2,
          bias: `fh${i}.fc2.b`,
          res: "gate",
          rbuf: this.fh,
          gbuf: this.ada,
          go: a + 1024
        })
      ];
    }).flat();
    const fa = 6 * 1536;
    return [
      g.gemv(x, "fh.cond.w", this.cond, {
        K: H2,
        N: H2,
        bias: "fh.cond.b",
        act: "silu",
        ln: { w: "norm.w", b: "norm.b", eps: LN_EPS }
      }),
      g.gemv(this.cond, "fh.ada.w", this.ada, { K: H2, N: 10240, bias: "fh.ada.b" }),
      g.op(noise(), [this.z, g.state], [1, 1, 1]),
      g.gemv(this.z, "fh.in.w", this.fh, { K: 32, N: H2, bias: "fh.in.b" }),
      ...blocks,
      g.gemv(this.fh, "fh.out.w", this.lat, {
        K: H2,
        N: 32,
        bias: "fh.out.b",
        res: "add",
        rbuf: this.z,
        ln: { eps: FLOW_LN_EPS, mod: this.ada, ho: fa, so: fa + 512 }
      }),
      g.op(
        finishStep({ eps: LN_EPS, threshold: EOS_THRESHOLD }),
        [
          x,
          g.weights,
          this.lat,
          this.lats,
          g.state,
          g.u(UNIFORMS.finishStep, {
            ewo: g.half("eos.w"),
            ebo: g.half("eos.b"),
            ngo: g.half("norm.w"),
            nbo: g.half("norm.b")
          })
        ],
        [1, 1, 1]
      ),
      g.gemv(this.lat, "latent_proj", x, { K: 32, N: H2, res: "vec", vname: "cond" })
    ];
  }
};

// src/codec.ts
var CHUNKS = [1, 2, 4, 8];
var C_MAX = 8;
var FRAME2 = 1920;
var WINDOW = 251;
var RING = 512;
var Codec = class {
  /** The last chunk's samples, up to C_MAX * FRAME floats. */
  audio;
  g;
  ops;
  stages;
  constructor(g, lats) {
    this.g = g;
    const T = 16 * C_MAX;
    const stage = (ctx, rate, ch) => ({
      ctx,
      rate,
      ch,
      buf: g.buffer((ctx + rate * T) * ch * 4)
    });
    const tr = stage(6, 1, 512);
    const s0 = stage(1, 1, 512);
    const s1 = stage(2, 6, 256);
    const s1h = stage(0, 6, 128);
    const s1r = stage(1, 6, 256);
    const s2 = stage(2, 30, 128);
    const s2h = stage(0, 30, 64);
    const s2r = stage(1, 30, 128);
    const s3 = stage(2, 120, 64);
    const s3h = stage(0, 120, 32);
    const s3r = stage(2, 120, 64);
    this.stages = [tr, s0, s1, s1r, s2, s2r, s3, s3r];
    this.audio = g.buffer(C_MAX * FRAME2 * 4);
    const B = rowBuffers(g, T);
    const rings = [0, 1].map(() => ({ kc: g.buffer(RING * H2 * 4), vc: g.buffer(RING * H2 * 4) }));
    const gate = "chunk";
    const opsFor = (C) => {
      const T2 = 16 * C;
      const conv = (name, src, dst, o) => {
        if (src.ctx !== o.k - 1) throw new Error(`${name} reads ${src.ctx} rows of context`);
        const out = dst instanceof GPUBuffer ? { buf: dst, ctx: 0 } : dst;
        return g.gemm(src.buf, `${name}.w`, out.buf, {
          M: src.rate * T2,
          K: o.k * o.cin,
          N: o.cout,
          bias: `${name}.b`,
          load: "conv",
          conv: { cin: o.cin, k: o.k, dil: 1, elu: o.elu },
          res: o.res ? "add" : "none",
          rbuf: o.res?.buf,
          ro: o.res ? o.res.ctx * o.cout : 0,
          yo: out.ctx * o.cout,
          gate
        });
      };
      const convT = (name, src, dst, o) => g.gemm(src.buf, `${name}.w`, dst.buf, {
        M: src.rate * T2,
        K: 2 * o.cin,
        N: o.cout,
        bias: `${name}.b`,
        load: "convT",
        conv: { cin: o.cin, elu: true, stride: o.s },
        yo: dst.ctx * o.cout,
        gate
      });
      const up = g.q8("up");
      return [
        g.op(
          upsample(),
          [
            lats,
            g.weights,
            B.x,
            g.u(UNIFORMS.upsample, {
              T: T2,
              wo: up.wo,
              wso: up.wso,
              mo: g.half("lat_mean"),
              so: g.half("lat_std")
            }),
            g.state
          ],
          [T2 * 512 / 256, 1, 1]
        ),
        ...[0, 1].flatMap(
          (i) => layerRows(g, `d${i}`, {
            B,
            rows: T2,
            ...rings[i],
            cache: { kind: "codec", window: WINDOW, ring: RING },
            pos0: 0,
            out: i === 1 ? { buf: tr.buf, yo: tr.ctx * 512 } : void 0
          })
        ),
        ...conv("s0", tr, s0, { cin: 512, cout: 512, k: 7, elu: false }),
        ...convT("s1", s0, s1, { cin: 512, cout: 256, s: 6 }),
        ...conv("s1r1", s1, s1h, { cin: 256, cout: 128, k: 3, elu: true }),
        ...conv("s1r2", s1h, s1r, { cin: 128, cout: 256, k: 1, elu: true, res: s1 }),
        ...convT("s2", s1r, s2, { cin: 256, cout: 128, s: 5 }),
        ...conv("s2r1", s2, s2h, { cin: 128, cout: 64, k: 3, elu: true }),
        ...conv("s2r2", s2h, s2r, { cin: 64, cout: 128, k: 1, elu: true, res: s2 }),
        ...convT("s3", s2r, s3, { cin: 128, cout: 64, s: 4 }),
        ...conv("s3r1", s3, s3h, { cin: 64, cout: 32, k: 3, elu: true }),
        ...conv("s3r2", s3h, s3r, { cin: 32, cout: 64, k: 1, elu: true, res: s3 }),
        ...conv("s4", s3r, this.audio, { cin: 64, cout: 1, k: 3, elu: true }),
        // Each stage keeps its last rows as the next chunk's left context.
        ...this.stages.map(
          (st) => g.op(
            shiftRows(),
            [
              st.buf,
              g.u(UNIFORMS.shiftRows, { src: st.rate * T2 * st.ch, count: st.ctx * st.ch })
            ],
            [Math.ceil(st.ctx * st.ch / 256), 1, 1]
          )
        )
      ];
    };
    this.ops = new Map(CHUNKS.map((c) => [c, opsFor(c)]));
  }
  /** Before a text's first chunk: no context yet. */
  reset(enc) {
    for (const st of this.stages) enc.clearBuffer(st.buf, 0, st.ctx * st.ch * 4);
  }
  /** Decodes frames [f0, f0 + size) into `audio`. The position goes to the queue ahead of the
   *  submit that holds this encoder, so a submit holds one chunk at most. */
  encodeChunk(enc, f0, size) {
    this.g.device.queue.writeBuffer(this.g.state, STATE.chunk * 4, new Uint32Array([16 * f0]));
    encode(enc, this.ops.get(size));
  }
};

// src/engine.ts
var DEPTH = 3;
var TEMPERATURE = 0.3;
var FRAMES_AFTER_EOS = 1;
var FRAMES_AFTER_EOS_SHORT = 3;
var SHORT_TEXT_WORDS = 4;
var FRAMES_PER_SECOND = 12.5;
var COPY_DST2 = 8;
var MAP_READ = 1;
var AUDIO_AT = 16;
var ROWS = Math.max(...CHUNKS);
var PhononEngine = class {
  g;
  model;
  lm;
  codec;
  /** Each voice's prompt keys and values (lm.ts). */
  voices = /* @__PURE__ */ new Map();
  stages;
  rowsAt;
  pace = Number.NaN;
  /** Settles when the text in progress has drained. */
  turn = Promise.resolve();
  constructor(device, model, { subgroups, align }) {
    this.model = model;
    this.g = new Graph(device, model, subgroups);
    this.lm = new LanguageModel(this.g, lmLayers(model), align);
    this.codec = new Codec(this.g, this.lm.lats);
    this.rowsAt = AUDIO_AT + ROWS * FRAME2 * 4;
    this.stages = Array.from(
      { length: DEPTH },
      () => device.createBuffer({ size: this.rowsAt + ROWS * TEXT_MAX * 4, usage: COPY_DST2 | MAP_READ })
    );
    this.setPace(0);
  }
  /** Settles when every shader has compiled. */
  ready() {
    return this.g.ready();
  }
  /** Registers a voice by its speaker latents, [125][512]. Its prefill uses the LM's caches, so
   *  it runs in the queue, before a text (`synthesize`'s `latents`). */
  addVoice(name, latents) {
    if (latents.length !== PROMPT * H2)
      throw new Error(`Voice ${name} has ${latents.length} values`);
    this.voices.set(name, this.lm.voice(latents));
  }
  hasVoice(name) {
    return this.voices.has(name);
  }
  /** The LM's pace (pace.ts): its condition and BOS row are written into the blob, so it is set
   *  in the queue, before a text (`synthesize`'s `pace`). */
  setPace(delta) {
    if (delta === this.pace) return;
    this.pace = delta;
    const { cond, bos } = paceVectors(this.model.pace, delta);
    for (const [name, values] of [
      ["cond", cond],
      ["bos_embed", bos]
    ])
      this.g.device.queue.writeBuffer(
        this.g.weights,
        this.model.tensors[name].offset,
        toF16(values)
      );
  }
  /** Text ids to 24 kHz audio in one voice, in chunks as they are made, with the alignment
   *  head's scores of each kept frame over the text as they are made. Null when cancelled. One
   *  text at a time: a call waits until the one before has drained, cancelled or not. */
  synthesize(...args) {
    const next = this.turn.then(() => this.run(...args));
    this.turn = next.then(() => {
    }, () => {
    });
    return next;
  }
  async run(ids, voice, {
    seed = 1,
    words = Infinity,
    pace = 0,
    latents,
    cancelled = () => false,
    onChunk,
    onRows
  } = {}) {
    if (!this.voices.has(voice)) {
      if (!latents) throw new Error(`Unknown voice ${voice}`);
      this.addVoice(voice, latents);
    }
    const kv = this.voices.get(voice);
    this.setPace(pace);
    if (!ids.length) return { chunks: [] };
    if (ids.length > TEXT_MAX) throw new Error(`Text too long: ${ids.length} tokens`);
    const { device } = this.g;
    const { lm, codec } = this;
    const maxFrames = Math.min(
      F_MAX,
      S_MAX - PROMPT - ids.length - 1,
      Math.ceil((ids.length / 3 + 2) * FRAMES_PER_SECOND)
    );
    const after = words <= SHORT_TEXT_WORDS ? FRAMES_AFTER_EOS_SHORT : FRAMES_AFTER_EOS;
    let enc = device.createCommandEncoder();
    codec.reset(enc);
    lm.start(enc, ids, kv, seed, TEMPERATURE, after);
    device.queue.submit([enc.finish()]);
    let issued = 1;
    let decoded = 0;
    let made = 0;
    let rowsCopied = 0;
    let end = null;
    const chunks = [];
    const free = [...this.stages];
    const inflight = [];
    const issue = () => {
      const stage = free.pop();
      enc = device.createCommandEncoder();
      const size = CHUNKS[Math.min(made, CHUNKS.length - 1)];
      const target = Math.min(maxFrames, end ?? Math.max(issued, decoded + size));
      const steps = [];
      for (; issued < target; issued++) steps.push(...lm.step);
      if (steps.length) encode(enc, steps);
      const avail = end ?? issued;
      let f0 = -1;
      if (decoded < avail && (decoded + size <= avail || end !== null)) {
        f0 = decoded;
        codec.encodeChunk(enc, f0, size);
        decoded += size;
        made++;
      }
      enc.copyBufferToBuffer(this.g.state, 0, stage, 0, 16);
      if (f0 >= 0) enc.copyBufferToBuffer(codec.audio, 0, stage, AUDIO_AT, size * FRAME2 * 4);
      const r0 = rowsCopied;
      if (issued > r0) {
        if (issued - r0 > ROWS) throw new Error(`${issued - r0} alignment rows in one submit`);
        const bytes = TEXT_MAX * 4;
        enc.copyBufferToBuffer(lm.align, r0 * bytes, stage, this.rowsAt, (issued - r0) * bytes);
        rowsCopied = issued;
      }
      device.queue.submit([enc.finish()]);
      inflight.push({ stage, f0, size, n: issued, r0, ready: stage.mapAsync(MAP_READ) });
    };
    const drain = async () => {
      const job = inflight.shift();
      await job.ready;
      const m = job.stage.getMappedRange();
      const keep2 = new Uint32Array(m, 0, 4)[STATE.end];
      if (end === null && keep2) end = Math.min(keep2, maxFrames);
      if (end === null && job.n >= maxFrames) end = maxFrames;
      const r1 = Math.min(job.n, end ?? Infinity);
      if (onRows && r1 > job.r0) {
        const n = ids.length;
        const rows = new Float32Array(m, this.rowsAt, (job.n - job.r0) * TEXT_MAX);
        const out = new Float32Array((r1 - job.r0) * n);
        for (let r = 0; r < r1 - job.r0; r++) out.set(rows.subarray(r * TEXT_MAX, r * TEXT_MAX + n), r * n);
        onRows(out);
      }
      if (job.f0 >= 0 && (end === null || job.f0 < end)) {
        const frames = Math.min(job.size, (end ?? Infinity) - job.f0);
        const chunk = new Float32Array(m, AUDIO_AT, frames * FRAME2).slice();
        chunks.push(chunk);
        onChunk?.(chunk);
      }
      job.stage.unmap();
      free.push(job.stage);
    };
    const finished = () => end !== null && decoded >= end && issued >= end;
    for (; ; ) {
      while (!cancelled() && inflight.length < DEPTH && !finished()) issue();
      if (!inflight.length) break;
      await drain();
    }
    if (cancelled()) return null;
    return { chunks };
  }
};

// src/timing.ts
var SAMPLE_RATE = 24e3;
var FRAME_SECS = 0.08;
var LAG = 3;
var SILENT_DB = -45;
var HOP = SAMPLE_RATE / 100;
var STEP_SECS = 5e-3;
var PUNCTUATION = /* @__PURE__ */ new Set([".", ",", "!", "?", ";", ":", ")", '"', "'"]);
function wordRanges(pieces) {
  const words = [];
  pieces.forEach((p, j) => {
    const text = p.replaceAll("\u2581", "");
    const joins = !p.startsWith("\u2581") || !!text && [...text].every((c) => PUNCTUATION.has(c));
    if (words.length && joins) {
      words.at(-1).text += text;
      words.at(-1).to = j + 1;
    } else words.push({ text, from: j, to: j + 1 });
  });
  return words.filter((w) => w.text);
}
var WordTimes = class {
  pieces;
  ranges;
  onWord;
  /** Each frame's position before the never-back rule, and after it (pos[0] waits for frame 1). */
  raw = [];
  pos = [];
  /** Loudness per complete 10 ms window, and the samples not yet in a window. */
  loud = [];
  samples = 0;
  partial = new Float32Array(HOP);
  partialAt = 0;
  /** The next time a token can start at, and each token's start before and after the silence step. */
  t = 0;
  starts;
  at = [];
  emitted = 0;
  ended = false;
  constructor(pieces, onWord) {
    this.pieces = pieces;
    this.ranges = wordRanges(pieces);
    this.onWord = onWord;
    this.starts = Array.from({ length: pieces.length }, () => null);
  }
  /** Alignment rows of the next frames, `pieces.length` scores each. */
  addRows(rows) {
    const n = this.pieces.length;
    for (let r = 0; r + n <= rows.length; r += n) {
      const row = rows.subarray(r, r + n);
      let k = 0;
      for (let i = 1; i < n; i++) if (row[i] > row[k]) k = i;
      let num = 0;
      let den = 0;
      for (let i = Math.max(0, k - 2); i <= Math.min(n - 1, k + 2); i++) {
        const p = Math.exp(row[i] - row[k]);
        num += p * i;
        den += p;
      }
      this.raw.push(num / den);
      this.settle();
    }
    this.advance();
  }
  /** The next samples of the text's audio. */
  addAudio(pcm) {
    for (let i = 0; i < pcm.length; i++) {
      this.partial[this.partialAt++] = pcm[i];
      if (this.partialAt === HOP) {
        let s = 0;
        for (let j = 0; j < HOP; j++) s += this.partial[j] * this.partial[j];
        this.loud.push(20 * Math.log10(Math.sqrt(s / HOP) + 1e-9));
        this.partialAt = 0;
      }
    }
    this.samples += pcm.length;
    this.advance();
  }
  /** The text has ended: every remaining word, and all the times. */
  finish() {
    this.ended = true;
    if (!this.raw.length) return { tokens: [], words: [] };
    this.settle();
    this.advance();
    const tokens = this.pieces.map((_, j) => this.token(j));
    return { tokens, words: this.ranges.map((w) => this.word(w)) };
  }
  get end() {
    return this.samples / SAMPLE_RATE;
  }
  /** Frame 0's row comes from the end-of-text query, which in some models reads the last token:
   *  it counts no further than frame 1's. Positions never move back. */
  settle() {
    const { raw, pos } = this;
    if (!pos.length && (raw.length >= 2 || this.ended && raw.length))
      pos.push(raw.length > 1 ? Math.min(raw[0], raw[1]) : raw[0]);
    while (pos.length && pos.length < raw.length) pos.push(Math.max(raw[pos.length], pos[pos.length - 1]));
  }
  /** The head's position heard at time t, or null while frames still to come could change it. */
  posAt(t) {
    const { pos } = this;
    const g = (t - FRAME_SECS / 2) / FRAME_SECS - LAG;
    if (!pos.length) return null;
    if (g <= 0) return pos[0];
    if (this.ended && g >= pos.length - 1) return pos[pos.length - 1];
    const i = Math.floor(g);
    if (i + 1 >= pos.length) return null;
    return pos[i] * (1 - (g - i)) + pos[i + 1] * (g - i);
  }
  /** Takes every time step and token start that can no longer change, then reports words. */
  advance() {
    const n = this.pieces.length;
    if (!n || !this.raw.length) return;
    for (; this.t <= this.end; this.t += STEP_SECS) {
      const p = this.posAt(this.t);
      if (p === null) break;
      const k = Math.round(p);
      for (let j = 0; j <= k && j < n; j++) this.starts[j] ??= this.t;
    }
    while (this.at.length < n) {
      const j = this.at.length;
      const start = this.starts[j];
      if (start === null && !this.ended) break;
      const from = start ?? this.end;
      let i = Math.floor(from * 100);
      while (i < this.loud.length && this.loud[i] < SILENT_DB) i++;
      if (i >= this.loud.length && !this.ended) break;
      this.at.push(Math.max(from, i / 100, j ? this.at[j - 1] : 0));
    }
    while (this.emitted < this.ranges.length) {
      const w = this.ranges[this.emitted];
      if (!(this.ended || w.to < this.at.length)) break;
      this.onWord?.(this.word(w));
      this.emitted++;
    }
  }
  token(j) {
    return {
      text: this.pieces[j].replaceAll("\u2581", ""),
      start: this.at[j],
      end: j + 1 < this.pieces.length ? this.at[j + 1] : this.end
    };
  }
  word(w) {
    const tokens = Array.from({ length: w.to - w.from }, (_, k) => this.token(w.from + k));
    return { text: w.text, start: tokens[0].start, end: tokens.at(-1).end, tokens };
  }
};

// src/tokenizer.ts
var BYTE = /^<0x([0-9A-F]{2})>$/;
var Tokenizer = class {
  pieces;
  score = /* @__PURE__ */ new Map();
  byte = /* @__PURE__ */ new Map();
  maxLen;
  unk;
  norm;
  constructor(json) {
    const vocab = json.model.vocab;
    this.pieces = vocab.map((v) => v[0]);
    vocab.forEach(([p, s], i) => {
      const m = BYTE.exec(p);
      if (m) this.byte.set(Number.parseInt(m[1], 16), i);
      else this.score.set(p, [i, s]);
    });
    this.maxLen = Math.max(...this.pieces.map((p) => p.length));
    this.unk = json.model.unk_id ?? 0;
    const n = json.normalizer;
    const rules = !n ? [] : n.type === "Sequence" ? n.normalizers : [n];
    this.norm = rules.map((rule) => {
      switch (rule.type) {
        case "Replace": {
          const re = rule.pattern.Regex !== void 0 ? new RegExp(rule.pattern.Regex.replaceAll(/\\x\{([0-9A-Fa-f]+)\}/g, "\\u{$1}"), "gu") : new RegExp(
            (rule.pattern.String ?? "").replaceAll(/[.*+?^${}()|[\]\\]/g, "\\$&"),
            "gu"
          );
          return (s) => s.replaceAll(re, rule.content);
        }
        case "Strip":
          return (s) => rule.strip_left && rule.strip_right ? s.trim() : rule.strip_left ? s.trimStart() : s.trimEnd();
        case "Prepend":
          return (s) => s ? rule.prepend + s : s;
      }
    });
  }
  normalize(text) {
    return this.norm.reduce((s, f) => f(s), text);
  }
  /** Ids for `text`, as the reference tokenizer gives them (no special tokens). */
  encode(text) {
    let s = this.normalize(text).replaceAll(" ", "\u2581");
    if (!s.startsWith("\u2581")) s = "\u2581" + s;
    const ids = [];
    for (const word of s.split(/(?=▁)/u)) if (word) ids.push(...this.word(word));
    return ids;
  }
  /** Viterbi over one word: its best-scoring pieces, bytes for what no piece covers. */
  word(w) {
    const chars = Array.from(w);
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1);
    const id = new Int32Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let s = "";
      for (let j = i; j < Math.min(n, i + this.maxLen); j++) {
        s += chars[j];
        const hit = this.score.get(s);
        if (hit && best[i] + hit[1] > best[j + 1]) {
          best[j + 1] = best[i] + hit[1];
          from[j + 1] = i;
          id[j + 1] = hit[0];
        }
      }
      if (best[i] - 100 > best[i + 1]) {
        best[i + 1] = best[i] - 100;
        from[i + 1] = i;
        id[i + 1] = -1;
      }
    }
    const out = [];
    for (let k = n; k > 0; k = from[k]) {
      if (id[k] >= 0) out.push([id[k]]);
      else
        out.push(
          [...new TextEncoder().encode(chars[k - 1])].map((b) => this.byte.get(b) ?? this.unk)
        );
    }
    return out.reverse().flat();
  }
};

// src/index.ts
async function openCache() {
  try {
    return await caches.open(MODEL_CACHE);
  } catch {
    return null;
  }
}
async function keep(cache, url, bytes) {
  try {
    await cache?.put(url, new Response(bytes));
  } catch {
  }
}
async function download(url, progress) {
  const response = await fetch(url);
  if (!response.ok || !response.body)
    throw new Error(`The voice model could not download (${response.status}).`);
  const total = Number(response.headers.get("content-length")) || 0;
  const parts2 = [];
  let loaded = 0;
  const reader = response.body.getReader();
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    parts2.push(value);
    loaded += value.byteLength;
    if (total && progress) progress(Math.min(100, Math.round(loaded / total * 100)));
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts2) {
    bytes.set(p, o);
    o += p.byteLength;
  }
  return bytes;
}
async function cached(url) {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) return hit.arrayBuffer();
  const bytes = await download(url);
  await keep(cache, url, bytes);
  return bytes.buffer;
}
async function weights(m, progress) {
  const cache = await openCache();
  const hit = await cache?.match(m.weights);
  if (hit) return unpackModel(await hit.arrayBuffer());
  const bytes = await download(m.weights, progress);
  await keep(cache, m.weights, bytes);
  return unpackModel(bytes.buffer);
}
function parts(text, count) {
  if (count(text) <= TEXT_MAX) return [text];
  const out = [];
  let part = "";
  const add = (piece) => {
    if (part && count(part + piece) > TEXT_MAX) {
      out.push(part);
      part = "";
    }
    if (count(piece) <= TEXT_MAX) part += piece;
    else if (/\s/.test(piece.trim())) for (const word of piece.split(/(?<=\s)/)) add(word);
    else for (let i = 0; i < piece.length; i += 32) add(piece.slice(i, i + 32));
  };
  for (const sentence of text.match(/[^.!?]+(?:[.!?]+|$)\s*/g) ?? [text]) add(sentence);
  if (part) out.push(part);
  return out.map((p) => p.trim()).filter(Boolean);
}
var Phonon = class _Phonon {
  engine;
  tokenizer;
  catalog;
  /** Settles when the GPU device is lost (a reset): load again to go on. */
  lost;
  /** The voice `speak` uses when none is given: the default of the model's first language. */
  defaultVoice;
  constructor(engine, tokenizer, catalog, languages, lost) {
    this.engine = engine;
    this.tokenizer = tokenizer;
    this.catalog = catalog;
    this.defaultVoice = DEFAULT_VOICE[languages[0]];
    this.lost = lost;
  }
  /** Loads a model (files.ts MODELS, Phonon 31M by default): a download the first time, then
   *  the browser's cache. `progress` hears the download in percent. */
  static async load({
    model = "phonon-31m-en",
    progress
  } = {}) {
    const files = MODELS[model];
    const gpu = navigator.gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
    if (!adapter) throw new Error("This browser has no WebGPU.");
    const subgroups = subgroups32(adapter);
    const device = await adapter.requestDevice({ requiredFeatures: subgroups ? ["subgroups"] : [] });
    const [blob, tokenizer, catalog] = await Promise.all([
      weights(files, progress ?? (() => {
      })),
      cached(tokenizerUrl(files)).then((b) => new Tokenizer(JSON.parse(new TextDecoder().decode(b)))),
      cached(CATALOG_URL).then(
        (b) => new TextDecoder().decode(b).split("\n").filter(Boolean).map((line) => JSON.parse(line)).filter((v) => files.languages.includes(v.language)).map((v) => ({
          name: v.name,
          id: v.voice_id,
          language: v.language,
          locale: v.locale,
          gender: v.gender,
          description: v.description
        }))
      )
    ]);
    const engine = new PhononEngine(device, blob, { subgroups, align: files.align });
    await engine.ready();
    return new _Phonon(engine, tokenizer, catalog, files.languages, device.lost);
  }
  /** The catalog's voices in the model's languages. */
  get voices() {
    return this.catalog;
  }
  /** The words `speak` gives for a text, in order, each with its tokens' text, as the tokenizer
   *  writes them: a page can lay them out before the times arrive. */
  words(text) {
    return this.parts(text).flatMap((part) => {
      const pieces = this.tokenizer.encode(part).map((id) => this.tokenizer.pieces[id].replaceAll("\u2581", ""));
      return wordRanges(this.tokenizer.encode(part).map((id) => this.tokenizer.pieces[id])).map((w) => ({
        text: w.text,
        tokens: pieces.slice(w.from, w.to)
      }));
    });
  }
  parts(text) {
    return parts(text, (s) => this.tokenizer.encode(s).length);
  }
  /** Text to speech, in parts that fit the model, joined. */
  async speak(text, options = {}) {
    const { voice = this.defaultVoice, speed = 1, seed = 1, onAudio, onWord, signal } = options;
    const v = this.catalog.find((c) => c.name === voice || c.id === voice);
    if (!v) throw new Error(`No voice ${voice}`);
    const latents = this.engine.hasVoice(v.id) ? void 0 : readVoice(await cached(voiceUrl(v.id)));
    const audio = [];
    const words = [];
    const tokens = [];
    let at = 0;
    for (const [i, part] of this.parts(text).entries()) {
      signal?.throwIfAborted();
      const ids = this.tokenizer.encode(part);
      const shift = (s) => ({ ...s, start: s.start + at, end: s.end + at });
      const shiftWord = (w) => ({ ...shift(w), tokens: w.tokens.map(shift) });
      const times = new WordTimes(
        ids.map((id) => this.tokenizer.pieces[id]),
        onWord && ((w) => onWord(shiftWord(w)))
      );
      const out = await this.engine.synthesize(ids, v.id, {
        seed: seed + i,
        words: part.split(/\s+/).filter(Boolean).length,
        pace: paceForSpeed(speed),
        latents,
        onRows: (rows) => times.addRows(rows),
        onChunk: (chunk) => {
          times.addAudio(chunk);
          onAudio?.(chunk);
        },
        cancelled: () => !!signal?.aborted
      });
      signal?.throwIfAborted();
      const pcm = join(out.chunks);
      const timed = times.finish();
      words.push(...timed.words.map(shiftWord));
      tokens.push(...timed.tokens.map(shift));
      audio.push(pcm);
      at += pcm.length / SAMPLE_RATE;
    }
    return { audio: join(audio), sampleRate: SAMPLE_RATE, words, tokens };
  }
};
function join(parts2) {
  const out = new Float32Array(parts2.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of parts2) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
function subgroups32(adapter) {
  const info = adapter.info;
  return adapter.features.has("subgroups") && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
}
export {
  DEFAULT_VOICE,
  MODELS,
  Phonon,
  TEXT_MAX
};
