function Ue(a,e,t){let r=e.length,n=new Float32Array(t);for(let o=0;o<t;o++){let i=0;for(let s=0;s<r;s++)i+=a[o*r+s]*e[s];n[o]=i}return n}function Ge(a,e){let r=new Float32Array(128);for(let s=0;s<64;s++){let h=1e3*e/1e4**(s/63);r[s]=Math.cos(h),r[64+s]=Math.sin(h)}let n=Ue(a.speakersProj,a.speakers,512),o=Ue(a.durationProj,r,512);for(let s=0;s<512;s++)n[s]+=o[s];let i=Ue(a.latentProj,a.bos,512);for(let s=0;s<512;s++)i[s]+=n[s];return{cond:n,bos:i}}var he=[[.77,.6],[.8,.45],[.84,.3],[.93,.15],[1,0],[1.07,-.15],[1.22,-.3],[1.3,-.45]];function e0(a){let e=he.findIndex(([i])=>a<=i);if(e===0)return he[0][1];if(e<0)return he.at(-1)[1];let[t,r]=he[e-1],[n,o]=he[e];return r+(a-t)/(n-t)*(o-r)}var X={pos:0,frame:1,end:2,chunk:3,text:4,after:5},G={layernorm:["go","bo"],rows:["rows","pos0"],embedRows:["eo","eso","bo"],noise:["seed","scale"],upsample:["T","wo","wso","mo","so"],shiftRows:["src","count"],gemm:["M","yo","ro","rows","wo","bo","wso"],gemmReduce:["rows","yo","ro","stride","bo"],gemv:["xo","yo","ro","go","so","ho","wo","bo","vo","lgo","lbo","wso"],finishStep:["ewo","ebo","ngo","nbo"]};function V(a,e={}){let t=Array.from({length:(4-a.length%4)%4},(r,n)=>`p${n}`);return`struct P { ${[...a,...t].map(r=>`${r}: ${e[r]??"u32"}`).join(", ")} }`}var ne=32,{pos:fe,frame:re,end:Y,chunk:Le,text:t0,after:J0}=X,J={none:"",step:`if (st[${Y}] != 0u && st[${re}] >= st[${Y}]) { return; }`,chunk:`if (st[${Y}] != 0u && st[${Le}] / 16u >= st[${Y}]) { return; }`},te=`
fn hf(i: u32) -> f32 { let f = unpack2x16float(wv[i >> 3u][(i >> 1u) & 3u]); return select(f.x, f.y, (i & 1u) == 1u); }
fn h4(i: u32) -> vec4<f32> { let q = wv[i >> 3u]; return select(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), (i & 4u) != 0u); }`,r0=(a,e,t)=>`(f32(unpack4xI8(wv[(${a}) / 4u + (${t}) / 16u][((${t}) / 4u) % 4u])[(${t}) % 4u]) * hf(${e} + (${t}) / ${ne}u))`,Ce=`
fn gelu(v: f32) -> f32 {
  // erf via Abramowitz-Stegun 7.1.26 is too coarse; use a high-precision rational erf.
  let z = v * 0.70710678118654752;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  let erfz = select(r - 1.0, 1.0 - r, z >= 0.0);
  return 0.5 * v * (1.0 + erfz);
}
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }`;function Oe(a,e){return a==="gelu"?`gelu(${e})`:a==="silu"?`silu(${e})`:e}function Ne(a,e){return{none:"",add:`v = r[${e}] + v;`,add_y:"v = y[o] + v;",gate:`v = r[${e}] + gv[p.go + n] * v;`,gate_y:"v = y[o] + gv[p.go + n] * v;",vec:"v = v + hf(p.vo + n);"}[a]}function Q0(a,e){if(e===1)return"fn sumSeg(v: f32, li: u32) -> f32 { return v; }";if(a){if(e===32)return"fn sumSeg(v: f32, li: u32) -> f32 { return subgroupAdd(v); }";let t="var t = v;";for(let r=e/2;r>=1;r/=2)t+=` t += subgroupShuffleXor(t, ${r}u);`;return`fn sumSeg(v: f32, li: u32) -> f32 { ${t} return t; }`}return`var<workgroup> r32: array<f32, 256>;
fn sumSeg(v: f32, li: u32) -> f32 {
  r32[li] = v; workgroupBarrier();
  for (var s = ${e/2}u; s > 0u; s >>= 1u) { if ((li % ${e}u) < s) { r32[li] += r32[li + s]; } workgroupBarrier(); }
  let o = r32[li - (li % ${e}u)]; workgroupBarrier(); return o;
}`}function n0({eps:a,gate:e}){return`
${V(G.layernorm)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
var<workgroup> red: array<f32, 128>;
${te}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &y; _ = &wv; _ = &st;
  ${J[e]}
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
  let inv = inverseSqrt(red[0] / 512.0 + ${a});
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 128u;
    y[base + c] = (v[i] - mean) * inv * hf(p.go + c) + hf(p.bo + c);
  }
}`}function o0(a){let e=a.kind==="codec";return{kv16:!e,at:e?`st[${Le}]`:"0u",gate:J[e?"chunk":"none"],slot:t=>e?`((${t}) % ${a.ring}u)`:`(${t})`,window:e?a.window:0}}function a0({theta:a,cache:e}){let{kv16:t,at:r,gate:n,slot:o}=o0(e),i=o("pos"),s=t?`let kw = (${i} * 512u + h * 64u) / 2u + t;
  kc[kw] = pack2x16float(vec2<f32>(rot(base + 512u, d0, c0, s0), rot(base + 512u, d1, c1, s1)));
  vc[kw] = pack2x16float(vec2<f32>(qkv[base + 1024u + d0], qkv[base + 1024u + d1]));`:`let kb = ${i} * 512u + h * 64u;
  kc[kb + d0] = rot(base + 512u, d0, c0, s0); kc[kb + d1] = rot(base + 512u, d1, c1, s1);
  vc[kb + d0] = qkv[base + 1024u + d0]; vc[kb + d1] = qkv[base + 1024u + d1];`;return`
${V(G.rows)}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<${t?"u32":"f32"}>;
@group(0) @binding(3) var<storage, read_write> vc: array<${t?"u32":"f32"}>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
fn rot(base: u32, d: u32, c: f32, s: f32) -> f32 {
  if (d < 32u) { return qkv[base + d] * c - qkv[base + d + 32u] * s; }
  return qkv[base + d] * c + qkv[base + d - 32u] * s;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &qkv; _ = &qb; _ = &kc; _ = &vc; _ = &st;
  ${n}
  let t = gid.x % 32u;
  let h = (gid.x / 32u) % 8u;
  let m = gid.x / 256u;
  if (m >= p.rows) { return; }
  let pos = p.pos0 + ${r} + m;
  let base = m * 1536u + h * 64u;
  let d0 = 2u * t; let d1 = d0 + 1u;
  let a0 = f32(pos) * pow(${a}, -f32(d0 % 32u) / 32.0);
  let a1 = f32(pos) * pow(${a}, -f32(d1 % 32u) / 32.0);
  let c0 = cos(a0); let s0 = sin(a0); let c1 = cos(a1); let s1 = sin(a1);
  qb[m * 512u + h * 64u + d0] = rot(base, d0, c0, s0);
  qb[m * 512u + h * 64u + d1] = rot(base, d1, c1, s1);
  ${s}
}`}function i0({cache:a}){let{kv16:e,at:t,gate:r,slot:n,window:o}=o0(a),i=e?"vec2<u32>":"vec4<f32>",s=(h,u)=>e?`kv(${h}[${u}])`:`${h}[${u}]`;return`
${V(G.rows)}
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<${i}>;
@group(0) @binding(2) var<storage, read> vc: array<${i}>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> ks: array<array<vec4<f32>, 16>, 16>;
var<workgroup> vs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> sc: array<array<f32, 17>, 16>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn lo(pos: u32) -> u32 { return ${o?`select(0u, pos + 1u - ${o}u, pos + 1u > ${o}u)`:"0u"}; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &vc; _ = &o; _ = &st;
  ${r}
  let h = wid.y; let m0 = wid.x * 16u;
  let r = li / 16u; let c = li % 16u;
  let m = m0 + r;
  let last = min(m0 + 15u, p.rows - 1u);
  let base = p.pos0 + ${t};
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
      let s = ${n("kp")} * 128u + h * 16u + c;
      ks[r][c] = ${s("kc","s")}; vs[r][c] = ${s("vc","s")};
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
}`}function s0({bos:a,pad:e}){return`
${V(G.embedRows)}
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${te}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &ids; _ = &wv; _ = &x;
  let c = gid.x % 512u; let i = gid.x / 512u;
  let id = ids[i];
  var v = 0.0;
  if (id == ${a}u) { v = hf(p.bo + c); } else if (id != ${e}u) { v = ${r0("p.eo","p.eso","id * 512u + c")}; }
  x[i * 512u + c] = v;
}`}function h0(){return`
${V(G.noise,{scale:"f32"})}
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
fn pcg(v: u32) -> u32 { var s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) i: u32) {
  _ = &z; _ = &st;
  ${J.step}
  let f = st[${re}];
  let a = pcg(p.seed ^ pcg(f * 64u + i * 2u)); let b = pcg(a ^ 0x9e3779b9u);
  let u1 = (f32(a >> 8u) + 1.0) / 16777217.0; let u2 = f32(b >> 8u) / 16777216.0;
  z[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2) * p.scale;
}`}function u0(){return`
${V(G.upsample)}
@group(0) @binding(0) var<storage, read> lats: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${te}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &lats; _ = &wv; _ = &y; _ = &st;
  ${J.chunk}
  let c = gid.x % 512u; let tl = gid.x / 512u;
  if (tl >= p.T) { return; }
  let t = st[${Le}] + tl; let ph = t % 16u; let q = t / 16u;
  var acc = 0.0;
  for (var j = 0u; j < 2u; j++) {
    if (q < j) { continue; }
    let src = q - j;
    for (var i = 0u; i < 32u; i++) {
      let v = lats[src * 32u + i] * hf(p.so + i) + hf(p.mo + i);
      acc += ${r0("p.wo","p.wso","((c * 16u + ph) * 2u + j) * 32u + i")} * v;
    }
  }
  y[tl * 512u + c] = acc;
}`}function l0(){return`
${V(G.shiftRows)}
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &b;
  if (gid.x >= p.count) { return; }
  b[gid.x] = b[p.src + gid.x];
}`}function je({K:a,N:e,act:t="none",bias:r=!1,res:n="none",load:o="linear",conv:i=null,splits:s=1,q8:h,gate:u}){let p=o==="convT",w={cin:0,k:1,dil:1,elu:!1,stride:1,...i},d=p?w.stride:1,l=w.cin,g=B=>`select(exp(${B}) - vec4<f32>(1.0), ${B}, ${B} > vec4<f32>(0.0))`,m=`fn loadA(m: u32, k: u32) -> vec4<f32> {
  if (m >= p.M) { return vec4<f32>(0.0); }
  let v = x[(${o==="linear"?`m * ${a}u + k`:o==="conv"?`(m + (k / ${l}u) * ${w.dil}u) * ${l}u + k % ${l}u`:`(m + 1u - k / ${l}u) * ${l}u + k % ${l}u`}) / 4u];
  return ${w.elu&&o!=="linear"?g("v"):"v"};
}`,y=Math.ceil(a/s/32)*32,M=p?d*a:a,R=h?`
    if (li < 128u) {
      let row = li / 2u; let kq = (li % 2u) * 16u;
      let n = tn + row;
      var q = vec4<u32>(0u); var sc = 0.0;
      if (n < ${e}u) {
        let col = wbase + kt + kq;
        q = wv[p.wo / 4u + (n * ${M}u + col) / 16u];
        sc = hf(p.wso + n * ${M/ne}u + col / ${ne}u);
      }
      let k4 = kq / 4u;
      Bs[row][k4] = vec4<f32>(unpack4xI8(q.x)) * sc; Bs[row][k4 + 1u] = vec4<f32>(unpack4xI8(q.y)) * sc;
      Bs[row][k4 + 2u] = vec4<f32>(unpack4xI8(q.z)) * sc; Bs[row][k4 + 3u] = vec4<f32>(unpack4xI8(q.w)) * sc;
    }`:`
    {
      let row = li / 4u; let kq = (li % 4u) * 8u;
      let n = tn + row;
      var q = vec4<u32>(0u);
      if (n < ${e}u) { q = wv[(p.wo * 2u + n * ${M}u + wbase + kt + kq) / 8u]; }
      Bs[row][kq / 4u] = vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y));
      Bs[row][kq / 4u + 1u] = vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w));
    }`,b=s>1?`y[(sp * p.rows + orow) * ${e}u + n] = v;`:`${r?"v += hf(p.bo + n);":""}
      v = ${Oe(t,"v")};
      let o = p.yo + orow * ${e}u + n;
      ${Ne(n,`p.ro + orow * ${e}u + n`)}
      y[o] = v;`;return`
${V(G.gemm)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
var<workgroup> As: array<array<vec4<f32>, 8>, 64>;
var<workgroup> Bs: array<array<vec4<f32>, 8>, 64>;
${te}
${Ce}
${m}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &st;
  ${J[u]}
  let tm = wid.y * 64u;
  let tn = wid.x * 64u;
  let ph = wid.z / ${s}u;
  let sp = wid.z % ${s}u;
  let wbase = ${p?`ph * ${2*l}u`:"0u"};
  let li = lid.y * 16u + lid.x;
  let k0 = sp * ${y}u;
  let k1 = min(${a}u, k0 + ${y}u);
  var acc: array<vec4<f32>, 4>;
  for (var kt = k0; kt < k1; kt += 32u) {
    for (var e = 0u; e < 2u; e++) {
      let idx = li + e * 256u;        // 512 vec4 = 64 rows x 8 vec4 (32 k)
      let row = idx / 8u; let k4 = idx % 8u;
      As[row][k4] = loadA(tm + row, kt + k4 * 4u);
    }
    ${R}
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
    let orow = ${p?`(m * ${d}u + ph)`:"m"};
    for (var j = 0u; j < 4u; j++) {
      let n = tn + lid.x * 4u + j;
      if (n >= ${e}u) { continue; }
      var v = acc[i][j];
      ${b}
    }
  }
}`}function c0({N:a,splits:e,act:t="none",bias:r=!1,res:n="none",gate:o}){return`
${V(G.gemmReduce)}
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
${te}
${Ce}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &part; _ = &y; _ = &wv; _ = &r; _ = &st;
  ${J[o]}
  let i = gid.x;
  if (i >= p.rows * ${a}u) { return; }
  let n = i % ${a}u; let row = i / ${a}u;
  var v = 0.0;
  for (var s = 0u; s < ${e}u; s++) { v += part[(s * p.stride + row) * ${a}u + n]; }
  ${r?"v += hf(p.bo + n);":""}
  v = ${Oe(t,"v")};
  let o = p.yo + i;
  ${Ne(n,"p.ro + i")}
  y[o] = v;
}`}function Ve(a){let r=a/16,n=Math.min(32,r),o=32/n,i=2*o;return{WG:128,R:2,U:r,LPR:n,RPP:o,RS:i,rowsPerGroup:128/32*i}}function p0({K:a,N:e,act:t="none",bias:r=!1,res:n="none",ln:o=null,combine:i=0,sg:s}){let{WG:h,R:u,U:p,LPR:w,RPP:d,RS:l}=Ve(a),g=a/4,f="p.xo / 4u",m=o?`
  // one pass: sum and sum of squares (f32 is ample for 512 values of this size)
  var s1 = 0.0; var s2 = 0.0;
  for (var i = li; i < 128u; i += ${h}u) { let v = x[${f} + i]; s1 += v.x + v.y + v.z + v.w; s2 += dot(v, v); }
  ${s?`s1 = subgroupAdd(s1); s2 = subgroupAdd(s2);
  if ((li & 31u) == 0u) { lnr[li >> 5u] = vec2<f32>(s1, s2); }`:`lnr[li] = vec2<f32>(s1, s2);
  workgroupBarrier();
  for (var s = ${h/2}u; s > 0u; s >>= 1u) { if (li < s) { lnr[li] += lnr[li + s]; } workgroupBarrier(); }`}
  workgroupBarrier();
  var t = vec2<f32>(0.0);
  ${s?`for (var i = 0u; i < ${h/32}u; i++) { t += lnr[i]; }`:"t = lnr[0];"}
  let mean = t.x / 512.0;
  let inv = inverseSqrt(max(t.y / 512.0 - mean * mean, 0.0) + ${o.eps});
  for (var i = li; i < 128u; i += ${h}u) {
    var o = (x[${f} + i] - vec4<f32>(mean)) * inv;
    ${o.affine?"o = o * h4(p.lgo + i * 4u) + h4(p.lbo + i * 4u);":""}
    ${o.modulate?"o = o * (vec4<f32>(1.0) + md[p.so / 4u + i]) + md[p.ho / 4u + i];":""}
    xs[i] = o;
  }
  workgroupBarrier();`:i?`
  // part: [split][head] blocks of 66 floats: 64 dims, max, sum
  for (var c = li; c < 512u; c += ${h}u) {
    let h = c / 64u; let d = c % 64u;
    var mx = -1e30;
    for (var s = 0u; s < ${i}u; s++) { mx = max(mx, r[(s * 8u + h) * 66u + 64u]); }
    var num = 0.0; var den = 0.0;
    for (var s = 0u; s < ${i}u; s++) {
      let b = (s * 8u + h) * 66u;
      let f = exp(r[b + 64u] - mx);
      num += r[b + d] * f; den += r[b + 65u] * f;
    }
    xsf[c] = num / den;
  }
  workgroupBarrier();
  for (var i = li; i < 128u; i += ${h}u) { xs[i] = vec4<f32>(xsf[i * 4u], xsf[i * 4u + 1u], xsf[i * 4u + 2u], xsf[i * 4u + 3u]); }
  workgroupBarrier();`:`
  for (var i = li; i < ${g}u; i += ${h}u) { xs[i] = x[${f} + i]; }
  workgroupBarrier();`;return`${s?"enable subgroups;":""}
${V(G.gemv)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> gv: array<f32>;
@group(0) @binding(6) var<storage, read> md: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> st: array<u32>;
var<workgroup> xs: array<vec4<f32>, ${Math.max(g,128)}>;
${o?`var<workgroup> lnr: array<vec2<f32>, ${s?h/32:h}>;`:""}
${i?"var<workgroup> xsf: array<f32, 512>;":""}
${te}
${Ce}
${Q0(s,w)}
@compute @workgroup_size(${h})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &gv; _ = &md; _ = &st;
  ${J.step}
  let lane = li & 31u;
  let ul = lane % ${w}u; let sub = lane / ${w}u;
  let n0 = (wid.x * ${h/32}u + (li >> 5u)) * ${l}u + sub;
  ${m}
  var acc: array<f32, ${u}>;
  for (var j = 0u; j < ${u}u; j++) {
    let n = n0 + j * ${d}u;
    if (n < ${e}u) {
      let wb = p.wo / 4u + n * ${p}u;
      for (var i = 0u; i < ${p/w}u; i++) {
        let u = ul + i * ${w}u;
        let q = wv[wb + u];
        let sc = hf(p.wso + n * ${a/ne}u + (u * 16u) / ${ne}u);
        let a = dot(xs[u * 4u], vec4<f32>(unpack4xI8(q.x))) + dot(xs[u * 4u + 1u], vec4<f32>(unpack4xI8(q.y)))
              + dot(xs[u * 4u + 2u], vec4<f32>(unpack4xI8(q.z))) + dot(xs[u * 4u + 3u], vec4<f32>(unpack4xI8(q.w)));
        acc[j] += a * sc;
      }
    }
  }
  for (var j = 0u; j < ${u}u; j++) {
    let tot = sumSeg(acc[j], li);
    let n = n0 + j * ${d}u;
    if (ul == 0u && n < ${e}u) {
      var v = tot${r?" + hf(p.bo + n)":""};
      v = ${Oe(t,"v")};
      let o = p.yo + n;
      ${Ne(n,"p.ro + n")}
      y[o] = v;
    }
  }
}`}function d0({eps:a,threshold:e}){return`
${V(G.finishStep)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> lat: array<f32>;
@group(0) @binding(3) var<storage, read_write> lats: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
var<workgroup> ended: u32;
${te}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &lat; _ = &lats; _ = &st;
  // st is read_write here, so the gate goes through workgroup memory to stay uniform.
  if (li == 0u) { ended = select(0u, 1u, st[${Y}] != 0u && st[${re}] >= st[${Y}]); }
  if (workgroupUniformLoad(&ended) == 1u) { return; }
  let a = x[li]; let b2 = x[li + 256u];
  red[li] = a + b2; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / 512.0; workgroupBarrier();
  red[li] = (a - mean) * (a - mean) + (b2 - mean) * (b2 - mean); workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 512.0 + ${a}); workgroupBarrier();
  let c2 = li + 256u;
  red[li] = ((a - mean) * inv * hf(p.ngo + li) + hf(p.nbo + li)) * hf(p.ewo + li)
          + ((b2 - mean) * inv * hf(p.ngo + c2) + hf(p.nbo + c2)) * hf(p.ewo + c2);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let f = st[${re}];
  if (li < 32u) { lats[f * 32u + li] = lat[li]; }
  let logit = red[0] + hf(p.ebo);
  workgroupBarrier();
  storageBarrier();
  if (li == 0u) {
    st[${fe}] = st[${fe}] + 1u; st[${re}] = f + 1u;
    if (st[${Y}] == 0u && logit > ${e}) { st[${Y}] = f + st[${J0}] + 1u; }
  }
}`}function ze({splits:a,theta:e,sg:t,align:r}){let n=o=>`vec2<u32>(pack2x16float(${o}.xy), pack2x16float(${o}.zw))`;return`${t?"enable subgroups;":""}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${r?"@group(0) @binding(5) var<storage, read_write> al: array<f32>;":""}
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
var<workgroup> qf: array<f32, 64>;
var<workgroup> kf: array<f32, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> pr: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> acc4: array<vec4<f32>, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qkv; _ = &kc; _ = &vc; _ = &part; _ = &st;${r?" _ = &al;":""}
  ${J.step}
  let sp = wid.x; let h = wid.y;
  let pos = st[${fe}];
  let hb = h * 64u;
  let n = pos + 1u;
  let per = (n + ${a-1}u) / ${a}u;
  let j0 = sp * per; let j1 = min(n, j0 + per);
  let own = pos >= j0 && pos < j1;
  if (li < 32u) {
    let ang = f32(pos) * pow(${e}, -f32(li) / 32.0);
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
    vc[(pos * 512u + hb + d) / 4u] = ${n("vv")};
  }
  workgroupBarrier();
  if (own && li < 16u) { let k4 = vec4<f32>(kf[li * 4u], kf[li * 4u + 1u], kf[li * 4u + 2u], kf[li * 4u + 3u]); kc[(pos * 512u + hb) / 4u + li] = ${n("k4")}; }
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
    ${t?`d += subgroupShuffleXor(d, 4u); d += subgroupShuffleXor(d, 2u); d += subgroupShuffleXor(d, 1u);
    if (part8 == 0u && base + kk < cnt) { sc[base + kk] = d; }`:`pr[li] = d; workgroupBarrier();
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
    ${r?`let k = j0 + j - ${r.lo}u;
    if (h == ${r.head}u && j0 + j >= ${r.lo}u && k < st[${t0}]) { al[st[${re}] * ${r.width}u + k] = sc[j]; }`:""}
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
}`}function f0({lo:a,head:e}){return`
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> al: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &al; _ = &st;
  let m = st[${fe}] - ${a}u;
  for (var i = li; i < st[${t0}]; i += 256u) {
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qb[m * 128u + ${e*16}u + e] * 0.125, kv(kc[(${a}u + i) * 128u + ${e*16}u + e])); }
    al[i] = d;
  }
}`}var m0=a=>Object.keys(a.tensors).filter(e=>/^l\d+\.qkv$/.test(e)).length;function Z0(a){let e=new DataView(a),t=Number(e.getBigUint64(0,!0)),r=JSON.parse(new TextDecoder().decode(new Uint8Array(a,8,t))),n=8+t,o=new Map;for(let[i,s]of Object.entries(r)){if(i==="__metadata__")continue;let[h,u]=s.data_offsets,p=a.slice(n+h,n+u),w;if(s.dtype==="F32")w=new Float32Array(p);else if(s.dtype==="BF16"){let d=new Uint16Array(p),l=new Uint32Array(d.length);for(let g=0;g<d.length;g++)l[g]=d[g]<<16;w=new Float32Array(l.buffer)}else throw new Error(`Unexpected tensor type ${s.dtype} for ${i}`);o.set(i,{shape:s.shape,data:w})}return o}function g0(a){let e=new Float32Array(1),t=new Uint32Array(e.buffer),r=new Uint16Array(a.length);for(let n=0;n<a.length;n++){e[0]=a[n];let o=t[0],i=o>>>16&32768,s=o>>>23&255,h=o&8388607;if(s===255){r[n]=i|31744|(h?512:0);continue}let u=s-127+15;if(u>=31){r[n]=i|31744;continue}if(u<=0){if(u<-10){r[n]=i;continue}h|=8388608;let d=14-u,l=h>>d,g=h&(1<<d)-1,f=1<<d-1;(g>f||g===f&&l&1)&&l++,r[n]=i|l;continue}let p=u<<10|h>>13,w=h&8191;(w>4096||w===4096&&p&1)&&p++,r[n]=i|p}return r}function Ie(a){let e=new DataView(a).getUint32(0,!0),t=JSON.parse(new TextDecoder().decode(new Uint8Array(a,4,e))),r=Math.ceil((4+e)/4)*4,n={};for(let[o,i]of t.pace)n[o]=new Float32Array(a,r,i),r+=i*4;return{tensors:t.tensors,pace:n,blob:new Uint8Array(a,r)}}function b0(a){let e=Z0(a).get("speaker_wavs");if(!e)throw new Error("The voice file has no speaker_wavs");let[,t,r]=e.shape,n=new Float32Array(r*t);for(let o=0;o<t;o++)for(let i=0;i<r;i++)n[i*t+o]=e.data[o*r+i];return n}var ae=(a,e,t)=>`https://huggingface.co/${a}/resolve/${e}/${t}`,me="phonon-webgpu",w0="blixt/phonon-40m-webgpu",v0="dde41aa27ed6bed7c874621d8b17e77dfece21bf",oe=(a,e,t,r)=>({source:{repo:`gradium/${a}`,revision:e},weights:ae(w0,v0,`${a}.int8.bin`),tokenizer:ae(w0,v0,`${a}.tokenizer.5.0.0.json`),languages:t,license:"CC-BY-NC-4.0",align:r}),Q={"phonon-31m-en":{source:{repo:"gradium/phonon-31m-en",revision:"92741c5b07d085ff840e26fea8ce43110b87852f"},weights:ae("blixt/phonon-31m-en-webgpu","f4635a6dddaa6072451e0155c03c97fbe889b6ec","phonon-31m-en.int8.bin"),tokenizer:"tokenizer.5.0.0.json",languages:["en"],license:"Apache-2.0",align:{layer:3,head:0}},"phonon-40m-en":oe("phonon-40m-en","bc7e8ae7e543a7a3e4604753e42fbda76ddf0781",["en"],{layer:6,head:0}),"phonon-40m-fr":oe("phonon-40m-fr","d91086970206b88efd2c88ca971d0628b5968f0f",["fr"],{layer:4,head:1}),"phonon-40m-de":oe("phonon-40m-de","79cfc7b47fcfc0b666657b981eb366415023a862",["de"],{layer:3,head:5}),"phonon-40m-es":oe("phonon-40m-es","e5e052b26e31abd5918ade71d3135e69d51a66f8",["es"],{layer:4,head:6}),"phonon-40m-pt":oe("phonon-40m-pt","4cbfd742a1555d60eb5399c6d296e9f2e07389e9",["pt"],{layer:4,head:2}),"phonon-40m-multilingual":oe("phonon-40m-multilingual","1537ce46c64f42903e0d608b132ccb073dac46d2",["en","fr","de","es","pt"],{layer:4,head:1})},y0=a=>a.tokenizer.startsWith("https:")?a.tokenizer:ae(a.source.repo,a.source.revision,a.tokenizer),k0="gradium/phonon-voices",x0="0ee17205f254b5abc43b266c8a0f215565c352f7",M0=ae(k0,x0,"voices.jsonl"),$0=a=>ae(k0,x0,`voices/${a}.safetensors`);var et=128,tt=64,A0=8,rt=4,nt=4,_0=8<<20;function Z(a,e){let t=a.beginComputePass();for(let r of e)t.setPipeline(r.pipe.pipeline),t.setBindGroup(0,r.bindGroup),t.dispatchWorkgroups(...r.groups);t.end()}function ot(a,e){let t=[...e.matchAll(/@binding\((\d+)\) var<(storage|uniform)(?:, (read_write|read))?>/g)].map(([,r,n,o])=>({binding:Number(r),visibility:nt,buffer:{type:n==="uniform"?"uniform":o==="read_write"?"storage":"read-only-storage"}}));return a.createBindGroupLayout({entries:t})}function B0(a,e,t){return e===t?{res:`${a}_y`,rbuf:void 0}:{res:a,rbuf:e}}var ge=class{device;weights;state;zero;sg;tensors;pipes=new Map;compiling=[];scratchBuf=null;constructor(e,t,r){this.device=e,this.tensors=t.tensors,this.sg=r,this.weights=this.buffer(t.blob.byteLength,t.blob),this.state=this.buffer(32),this.zero=this.buffer(256)}buffer(e,t){let r=this.device.createBuffer({size:Math.max(256,Math.ceil(e/256)*256),usage:et|A0|rt});return t&&this.device.queue.writeBuffer(r,0,t.buffer,t.byteOffset,t.byteLength),r}u(e,t){let r=t,n=new Uint32Array(Math.max(4,Math.ceil(e.length/4)*4));e.forEach((i,s)=>n[s]=r[i]??0);let o=this.device.createBuffer({size:n.byteLength,usage:tt|A0});return this.device.queue.writeBuffer(o,0,n),o}tensor(e){let t=this.tensors[e];if(!t)throw new Error(`No tensor ${e}`);return t}half(e){let t=this.tensor(e);if(t.scales!==void 0)throw new Error(`${e} is int8`);return t.offset/2}words(e){let t=this.tensor(e);return{wo:t.offset/4,wso:t.scales===void 0?0:t.scales/2,q8:t.scales!==void 0}}q8(e){let t=this.words(e);if(!t.q8)throw new Error(`${e} is not int8`);return t}async ready(){for(;this.compiling.length;)await Promise.all(this.compiling.splice(0))}op(e,t,r){let n=this.pipes.get(e);if(!n){let i=ot(this.device,e),s={layout:i,pipeline:null};this.compiling.push(this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[i]}),compute:{module:this.device.createShaderModule({code:e}),entryPoint:"main"}}).then(h=>{s.pipeline=h})),this.pipes.set(e,s),n=s}let o=this.device.createBindGroup({layout:n.layout,entries:t.map((i,s)=>({binding:s,resource:i instanceof GPUBuffer?{buffer:i}:i}))});return{pipe:n,bindGroup:o,groups:r}}scratch(e){if(e>_0)throw new Error(`Split-K partials of ${e} bytes`);return this.scratchBuf??=this.buffer(_0),{buffer:this.scratchBuf,offset:0,size:Math.ceil(e/256)*256}}gemv(e,t,r,n){let{ln:o,combine:i}=n,{res:s,rbuf:h}=B0(n.res??"none",n.rbuf,r),u=this.q8(t),p=p0({K:n.K,N:n.N,act:n.act,bias:!!n.bias,res:s,combine:i?.splits,ln:o&&{affine:!!o.w,modulate:!!o.mod,eps:o.eps},sg:this.sg}),w=this.u(G.gemv,{xo:n.xo,go:n.go,so:o?.so,ho:o?.ho,wo:u.wo,bo:n.bias?this.half(n.bias):0,vo:n.vname?this.half(n.vname):0,lgo:o?.w?this.half(o.w):0,lbo:o?.b?this.half(o.b):0,wso:u.wso});return this.op(p,[e,this.weights,r,i?.parts??h??this.zero,w,n.gbuf??this.zero,o?.mod??this.zero,this.state],[Math.ceil(n.N/Ve(n.K).rowsPerGroup),1,1])}gemm(e,t,r,n){let{M:o,K:i,N:s,load:h="linear",conv:u,yo:p=0,ro:w=0,gate:d}=n,{res:l,rbuf:g}=B0(n.res??"none",n.rbuf,r),f=this.words(t),m=h==="convT"?u.stride:1,y=Math.ceil(s/64)*Math.min(4,Math.ceil(o/64))*m,M=Math.max(1,Math.min(Math.floor(160/y),Math.floor(i/128))),R=[Math.ceil(s/64),Math.ceil(o/64),m*M],b=n.bias?this.half(n.bias):0,B={K:i,N:s,load:h,conv:u,q8:f.q8,gate:d},P=G.gemm;if(M===1){let c=je({...B,act:n.act,bias:!!n.bias,res:l});return[this.op(c,[e,this.weights,r,g??this.zero,this.u(P,{M:o,yo:p,ro:w,wo:f.wo,bo:b,wso:f.wso}),this.state],R)]}let T=o*m,v=this.scratch(M*T*s*4);return[this.op(je({...B,splits:M}),[e,this.weights,v,this.zero,this.u(P,{M:o,rows:T,wo:f.wo,wso:f.wso}),this.state],R),this.op(c0({N:s,splits:M,act:n.act,bias:!!n.bias,res:l,gate:d}),[v,r,this.weights,g??this.zero,this.u(G.gemmReduce,{rows:T,yo:p,ro:w,stride:T,bo:b}),this.state],[Math.ceil(T*s/256),1,1])]}layernorm(e,t,r,n,o,i,s){return this.op(n0({eps:i,gate:s}),[e,t,this.weights,this.u(G.layernorm,{go:this.half(n),bo:this.half(o)}),this.state],[r,1,1])}};var x=512,ue=2048,z=125,we=1024,ve=768,j=240,He=[64,128,192,256],S0=4294967294,P0=4294967295,se="1e-5",T0="1e-6",De="10000.0",at="4.0",ie=8,be=2;function We(a,e){let t=r=>a.buffer(e*r*4);return{x:t(x),h:t(x),qkv:t(3*x),q:t(x),att:t(x),ff:t(ue)}}function Xe(a,e,t){let{B:r,rows:n,cache:o}=t,i=o.kind==="codec"?"chunk":"none",s=a.u(G.rows,{rows:n,pos0:t.pos0});return[a.layernorm(r.x,r.h,n,`${e}.ln1.w`,`${e}.ln1.b`,se,i),...a.gemm(r.h,`${e}.qkv`,r.qkv,{M:n,K:x,N:3*x,gate:i}),a.op(a0({theta:De,cache:o}),[r.qkv,r.q,t.kc,t.vc,a.state,s],[n,1,1]),a.op(i0({cache:o}),[r.q,t.kc,t.vc,r.att,a.state,s],[Math.ceil(n/16),8,1]),...a.gemm(r.att,`${e}.o`,r.x,{M:n,K:x,N:x,res:"add",rbuf:r.x,gate:i}),a.layernorm(r.x,r.h,n,`${e}.ln2.w`,`${e}.ln2.b`,se,i),...a.gemm(r.h,`${e}.fc1`,r.ff,{M:n,K:x,N:ue,act:"gelu",gate:i}),...a.gemm(r.ff,`${e}.fc2`,t.out?.buf??r.x,{M:n,K:ue,N:x,res:"add",rbuf:r.x,yo:t.out?.yo??0,gate:i})]}var ye=class{lats;step;align;g;rows;kc;vc;ids;noise;bos;voiceIn;voiceOps;prefills;tail;cond;ada;fh;ft;z;lat;parts;depth;alignLayer;alignAt;constructor(e,t,r){this.g=e,this.depth=t,this.alignLayer=r.layer,this.alignAt={lo:z,width:j,head:r.head};let n=He.at(-1);this.rows=We(e,n),this.kc=Array.from({length:t},()=>e.buffer(we*x*be)),this.vc=Array.from({length:t},()=>e.buffer(we*x*be)),this.cond=e.buffer(x*4),this.ada=e.buffer(10240*4),this.fh=e.buffer(x*4),this.ft=e.buffer(x*4),this.z=e.buffer(128),this.lat=e.buffer(128),this.lats=e.buffer(ve*32*4),this.ids=e.buffer(n*4),this.bos=e.buffer(x*4),this.parts=e.buffer(ie*8*66*4),this.voiceIn=e.buffer(z*x*4),this.align=e.buffer(ve*j*4),this.noise=e.u(G.noise,{}),this.tail=this.tailOps(),this.step=[...Array.from({length:t},(i,s)=>this.stepLayer(s)).flat(),...this.tail],this.voiceOps=[...e.gemm(this.voiceIn,"prompt_proj",this.rows.x,{M:z,K:x,N:x,gate:"none"}),...this.layers(z,0)];let o=e.q8("embed");this.prefills=new Map(He.map(i=>[i,[e.op(s0({bos:S0,pad:P0}),[this.ids,e.weights,this.rows.x,e.u(G.embedRows,{eo:o.wo,eso:o.wso,bo:e.half("bos_embed")})],[i*x/256,1,1]),...this.layers(i,z,!0)]]))}layers(e,t,r=!1){let{g:n}=this;return Array.from({length:this.depth},(o,i)=>[...Xe(n,`l${i}`,{B:this.rows,rows:e,kc:this.kc[i],vc:this.vc[i],cache:{kind:"lm"},pos0:t}),...r&&i===this.alignLayer?[n.op(f0(this.alignAt),[this.rows.q,this.kc[i],this.align,n.state],[1,1,1])]:[]]).flat()}voice(e){let{device:t}=this.g,r=z*x*be,n=this.g.buffer(2*this.depth*r);t.queue.writeBuffer(this.voiceIn,0,e.buffer,e.byteOffset,e.byteLength);let o=t.createCommandEncoder();Z(o,this.voiceOps);for(let i=0;i<this.depth;i++)o.copyBufferToBuffer(this.kc[i],0,n,2*i*r,r),o.copyBufferToBuffer(this.vc[i],0,n,(2*i+1)*r,r);return t.queue.submit([o.finish()]),n}start(e,t,r,n,o,i){let{device:s}=this.g,h=He.find(l=>l>t.length),u=new Uint32Array(h).fill(P0);u.set(t),u[t.length]=S0,s.queue.writeBuffer(this.ids,0,u);let p=new DataView(new ArrayBuffer(16));p.setUint32(0,n>>>0,!0),p.setFloat32(4,Math.sqrt(o),!0),s.queue.writeBuffer(this.noise,0,p.buffer);let w=new Uint32Array(8);w[X.pos]=z+t.length,w[X.after]=i,w[X.text]=t.length,s.queue.writeBuffer(this.g.state,0,w);let d=z*x*be;for(let l=0;l<this.depth;l++)e.copyBufferToBuffer(r,2*l*d,this.kc[l],0,d),e.copyBufferToBuffer(r,(2*l+1)*d,this.vc[l],0,d);Z(e,this.prefills.get(h)),e.copyBufferToBuffer(this.rows.x,t.length*x*4,this.bos,0,x*4),e.copyBufferToBuffer(this.bos,0,this.rows.x,0,x*4),Z(e,this.tail)}stepLayer(e){let{g:t}=this,r=this.rows,n=`l${e}`;return[t.gemv(r.x,`${n}.qkv`,r.qkv,{K:x,N:3*x,ln:{w:`${n}.ln1.w`,b:`${n}.ln1.b`,eps:se}}),e===this.alignLayer?t.op(ze({splits:ie,theta:De,sg:t.sg,align:this.alignAt}),[r.qkv,this.kc[e],this.vc[e],this.parts,t.state,this.align],[ie,8,1]):t.op(ze({splits:ie,theta:De,sg:t.sg}),[r.qkv,this.kc[e],this.vc[e],this.parts,t.state],[ie,8,1]),t.gemv(t.zero,`${n}.o`,r.x,{K:x,N:x,res:"add",rbuf:r.x,combine:{splits:ie,parts:this.parts}}),t.gemv(r.x,`${n}.fc1`,r.ff,{K:x,N:ue,act:"gelu",ln:{w:`${n}.ln2.w`,b:`${n}.ln2.b`,eps:se}}),t.gemv(r.ff,`${n}.fc2`,r.x,{K:ue,N:x,res:"add",rbuf:r.x})]}tailOps(){let{g:e}=this,t=this.rows.x,r=Array.from({length:6},(o,i)=>{let s=i*1536;return[e.gemv(this.fh,`fh${i}.fc1.w`,this.ft,{K:x,N:x,bias:`fh${i}.fc1.b`,act:"silu",ln:{w:`fh${i}.ln.w`,b:`fh${i}.ln.b`,eps:T0,mod:this.ada,ho:s,so:s+512}}),e.gemv(this.ft,`fh${i}.fc2.w`,this.fh,{K:x,N:x,bias:`fh${i}.fc2.b`,res:"gate",rbuf:this.fh,gbuf:this.ada,go:s+1024})]}).flat(),n=6*1536;return[e.gemv(t,"fh.cond.w",this.cond,{K:x,N:x,bias:"fh.cond.b",act:"silu",ln:{w:"norm.w",b:"norm.b",eps:se}}),e.gemv(this.cond,"fh.ada.w",this.ada,{K:x,N:10240,bias:"fh.ada.b"}),e.op(h0(),[this.z,e.state,this.noise],[1,1,1]),e.gemv(this.z,"fh.in.w",this.fh,{K:32,N:x,bias:"fh.in.b"}),...r,e.gemv(this.fh,"fh.out.w",this.lat,{K:x,N:32,bias:"fh.out.b",res:"add",rbuf:this.z,ln:{eps:T0,mod:this.ada,ho:n,so:n+512}}),e.op(d0({eps:se,threshold:at}),[t,e.weights,this.lat,this.lats,e.state,e.u(G.finishStep,{ewo:e.half("eos.w"),ebo:e.half("eos.b"),ngo:e.half("norm.w"),nbo:e.half("norm.b")})],[1,1,1]),e.gemv(this.lat,"latent_proj",t,{K:32,N:x,res:"vec",vname:"cond"})]}};var le=[1,2,4,8],E0=8,ce=1920,it=251,Ye=512,ke=class{audio;g;ops;stages;constructor(e,t){this.g=e;let r=16*E0,n=(b,B,P)=>({ctx:b,rate:B,ch:P,buf:e.buffer((b+B*r)*P*4)}),o=n(6,1,512),i=n(1,1,512),s=n(2,6,256),h=n(0,6,128),u=n(1,6,256),p=n(2,30,128),w=n(0,30,64),d=n(1,30,128),l=n(2,120,64),g=n(0,120,32),f=n(2,120,64);this.stages=[o,i,s,u,p,d,l,f],this.audio=e.buffer(E0*ce*4);let m=We(e,r),y=[0,1].map(()=>({kc:e.buffer(Ye*x*4),vc:e.buffer(Ye*x*4)})),M="chunk",R=b=>{let B=16*b,P=(c,A,L,S)=>{if(A.ctx!==S.k-1)throw new Error(`${c} reads ${A.ctx} rows of context`);let C=L instanceof GPUBuffer?{buf:L,ctx:0}:L;return e.gemm(A.buf,`${c}.w`,C.buf,{M:A.rate*B,K:S.k*S.cin,N:S.cout,bias:`${c}.b`,load:"conv",conv:{cin:S.cin,k:S.k,dil:1,elu:S.elu},res:S.res?"add":"none",rbuf:S.res?.buf,ro:S.res?S.res.ctx*S.cout:0,yo:C.ctx*S.cout,gate:M})},T=(c,A,L,S)=>e.gemm(A.buf,`${c}.w`,L.buf,{M:A.rate*B,K:2*S.cin,N:S.cout,bias:`${c}.b`,load:"convT",conv:{cin:S.cin,elu:!0,stride:S.s},yo:L.ctx*S.cout,gate:M}),v=e.q8("up");return[e.op(u0(),[t,e.weights,m.x,e.u(G.upsample,{T:B,wo:v.wo,wso:v.wso,mo:e.half("lat_mean"),so:e.half("lat_std")}),e.state],[B*512/256,1,1]),...[0,1].flatMap(c=>Xe(e,`d${c}`,{B:m,rows:B,...y[c],cache:{kind:"codec",window:it,ring:Ye},pos0:0,out:c===1?{buf:o.buf,yo:o.ctx*512}:void 0})),...P("s0",o,i,{cin:512,cout:512,k:7,elu:!1}),...T("s1",i,s,{cin:512,cout:256,s:6}),...P("s1r1",s,h,{cin:256,cout:128,k:3,elu:!0}),...P("s1r2",h,u,{cin:128,cout:256,k:1,elu:!0,res:s}),...T("s2",u,p,{cin:256,cout:128,s:5}),...P("s2r1",p,w,{cin:128,cout:64,k:3,elu:!0}),...P("s2r2",w,d,{cin:64,cout:128,k:1,elu:!0,res:p}),...T("s3",d,l,{cin:128,cout:64,s:4}),...P("s3r1",l,g,{cin:64,cout:32,k:3,elu:!0}),...P("s3r2",g,f,{cin:32,cout:64,k:1,elu:!0,res:l}),...P("s4",f,this.audio,{cin:64,cout:1,k:3,elu:!0}),...this.stages.map(c=>e.op(l0(),[c.buf,e.u(G.shiftRows,{src:c.rate*B*c.ch,count:c.ctx*c.ch})],[Math.ceil(c.ctx*c.ch/256),1,1]))]};this.ops=new Map(le.map(b=>[b,R(b)]))}reset(e){for(let t of this.stages)e.clearBuffer(t.buf,0,t.ctx*t.ch*4)}encodeChunk(e,t,r){this.g.device.queue.writeBuffer(this.g.state,X.chunk*4,new Uint32Array([16*t])),Z(e,this.ops.get(r))}};var R0=3,st=.3,ht=1,ut=3,lt=4,ct=12.5,F0=8,xe=1,Je=16,Me=class{g;model;lm;codec;voices=new Map;stages;pace=Number.NaN;constructor(e,t,{subgroups:r,align:n}){this.model=t,this.g=new ge(e,t,r),this.lm=new ye(this.g,m0(t),n),this.codec=new ke(this.g,this.lm.lats);let o=Math.max(...le);this.stages=Array.from({length:R0+1},()=>e.createBuffer({size:Je+o*ce*4,usage:F0|xe})),this.setPace(0)}ready(){return this.g.ready()}addVoice(e,t){if(t.length!==z*x)throw new Error(`Voice ${e} has ${t.length} values`);this.voices.set(e,this.lm.voice(t))}hasVoice(e){return this.voices.has(e)}setPace(e){if(e===this.pace)return;this.pace=e;let{cond:t,bos:r}=Ge(this.model.pace,e);for(let[n,o]of[["cond",t],["bos_embed",r]])this.g.device.queue.writeBuffer(this.g.weights,this.model.tensors[n].offset,g0(o))}async synthesize(e,t,{seed:r=1,words:n=1/0,cancelled:o=()=>!1,onChunk:i}={}){let s=this.voices.get(t);if(!s)throw new Error(`Unknown voice ${t}`);if(!e.length)return{chunks:[],align:new Float32Array(0)};if(e.length>j)throw new Error(`Text too long: ${e.length} tokens`);let{device:h}=this.g,{lm:u,codec:p}=this,w=Math.min(ve,we-z-e.length-1,Math.ceil((e.length/3+2)*ct)),d=n<=lt?ut:ht,l=h.createCommandEncoder();p.reset(l),u.start(l,e,s,r,st,d),h.queue.submit([l.finish()]);let g=1,f=0,m=0,y=null,M=[],R=[...this.stages],b=[],B=()=>{let v=R.pop();l=h.createCommandEncoder();let c=le[Math.min(m,le.length-1)],A=Math.min(w,y??Math.max(g,f+c)),L=[];for(;g<A;g++)L.push(...u.step);L.length&&Z(l,L);let S=y??g,C=-1;f<S&&(f+c<=S||y!==null)&&(C=f,p.encodeChunk(l,C,c),f+=c,m++),l.copyBufferToBuffer(this.g.state,0,v,0,16),C>=0&&l.copyBufferToBuffer(p.audio,0,v,Je,c*ce*4),h.queue.submit([l.finish()]),b.push({stage:v,f0:C,size:c,n:g,ready:v.mapAsync(xe)})},P=async()=>{let v=b.shift();await v.ready;let c=v.stage.getMappedRange(),A=new Uint32Array(c,0,4)[X.end];if(y===null&&A&&(y=Math.min(A,w)),y===null&&v.n>=w&&(y=w),v.f0>=0&&(y===null||v.f0<y)){let L=Math.min(v.size,(y??1/0)-v.f0),S=new Float32Array(c,Je,L*ce).slice();M.push(S),i?.(S)}v.stage.unmap(),R.push(v.stage)},T=()=>y!==null&&f>=y&&g>=y;for(;;){for(;!o()&&b.length<R0&&!T();)B();if(!b.length)break;await P()}return o()?null:{chunks:M,align:await this.alignment(y,e.length)}}async alignment(e,t){let{device:r}=this.g,n=e*j*4,o=r.createBuffer({size:n,usage:F0|xe}),i=r.createCommandEncoder();i.copyBufferToBuffer(this.lm.align,0,o,0,n),r.queue.submit([i.finish()]),await o.mapAsync(xe);let s=new Float32Array(o.getMappedRange()),h=new Float32Array(e*t);for(let u=0;u<e;u++)h.set(s.subarray(u*j,u*j+t),u*t);return o.destroy(),h}};var pt=new Set([".",",","!","?",";",":",")",'"',"'"]);function dt(a,e,t){let r=new Float32Array(e);for(let n=0;n<e;n++){let o=a.subarray(n*t,n*t+t),i=0;for(let u=1;u<t;u++)o[u]>o[i]&&(i=u);let s=0,h=0;for(let u=Math.max(0,i-2);u<=Math.min(t-1,i+2);u++){let p=Math.exp(o[u]-o[i]);s+=p*u,h+=p}r[n]=s/h}e>1&&(r[0]=Math.min(r[0],r[1]));for(let n=1;n<e;n++)r[n]=Math.max(r[n],r[n-1]);return r}function ft(a){let t=new Float32Array(Math.floor(a.length/240));for(let r=0;r<t.length;r++){let n=0;for(let o=r*240;o<(r+1)*240;o++)n+=a[o]*a[o];t[r]=20*Math.log10(Math.sqrt(n/240)+1e-9)}return t}function q0(a,e,t){let r=e.length,n=r?a.length/r:0,o=t.length/24e3;if(!r||!n)return{tokens:[],words:[]};let i=dt(a,n,r),s=l=>{let g=(l-.04)/.08-3;if(g<=0)return i[0];if(g>=n-1)return i[n-1];let f=Math.floor(g);return i[f]*(1-(g-f))+i[f+1]*(g-f)},h=Array.from({length:r},()=>null);for(let l=0;l<=o;l+=.005){let g=Math.round(s(l));for(let f=0;f<=g&&f<r;f++)h[f]??=l}let u=ft(t),p=[];for(let l=0;l<r;l++){let g=Math.floor((h[l]??o)*100);for(;g<u.length&&u[g]<-45;)g++;p.push(Math.max(h[l]??o,g/100,l?p[l-1]:0))}let w=e.map((l,g)=>({text:l.replaceAll("\u2581",""),start:p[g],end:g+1<r?p[g+1]:o})),d=[];return e.forEach((l,g)=>{let f=l.replaceAll("\u2581",""),m=!l.startsWith("\u2581")||!!f&&[...f].every(y=>pt.has(y));if(d.length&&m){let y=d.at(-1);y.text+=f,y.end=w[g].end}else d.push({text:f,start:w[g].start,end:w[g].end})}),{tokens:w,words:d.filter(l=>l.text)}}var mt=/^<0x([0-9A-F]{2})>$/,$e=class{pieces;score=new Map;byte=new Map;maxLen;unk;norm;constructor(e){let t=e.model.vocab;this.pieces=t.map(o=>o[0]),t.forEach(([o,i],s)=>{let h=mt.exec(o);h?this.byte.set(Number.parseInt(h[1],16),s):this.score.set(o,[s,i])}),this.maxLen=Math.max(...this.pieces.map(o=>o.length)),this.unk=e.model.unk_id??0;let r=e.normalizer,n=r?r.type==="Sequence"?r.normalizers:[r]:[];this.norm=n.map(o=>{switch(o.type){case"Replace":{let i=o.pattern.Regex!==void 0?new RegExp(o.pattern.Regex.replaceAll(/\\x\{([0-9A-Fa-f]+)\}/g,"\\u{$1}"),"gu"):new RegExp((o.pattern.String??"").replaceAll(/[.*+?^${}()|[\]\\]/g,"\\$&"),"gu");return s=>s.replaceAll(i,o.content)}case"Strip":return i=>o.strip_left&&o.strip_right?i.trim():o.strip_left?i.trimStart():i.trimEnd();case"Prepend":return i=>i&&o.prepend+i}})}normalize(e){return this.norm.reduce((t,r)=>r(t),e)}encode(e){let t=this.normalize(e).replaceAll(" ","\u2581");t.startsWith("\u2581")||(t="\u2581"+t);let r=[];for(let n of t.split(/(?=▁)/u))n&&r.push(...this.word(n));return r}word(e){let t=Array.from(e),r=t.length,n=new Float64Array(r+1).fill(-1/0),o=new Int32Array(r+1),i=new Int32Array(r+1);n[0]=0;for(let h=0;h<r;h++){if(n[h]===-1/0)continue;let u="";for(let p=h;p<Math.min(r,h+this.maxLen);p++){u+=t[p];let w=this.score.get(u);w&&n[h]+w[1]>n[p+1]&&(n[p+1]=n[h]+w[1],o[p+1]=h,i[p+1]=w[0])}n[h]-100>n[h+1]&&(n[h+1]=n[h]-100,o[h+1]=h,i[h+1]=-1)}let s=[];for(let h=r;h>0;h=o[h])i[h]>=0?s.push([i[h]]):s.push([...new TextEncoder().encode(t[h-1])].map(u=>this.byte.get(u)??this.unk));return s.reverse().flat()}};async function L0(){try{return await caches.open(me)}catch{return null}}async function C0(a,e,t){try{await a?.put(e,new Response(t))}catch{}}async function O0(a,e){let t=await fetch(a);if(!t.ok||!t.body)throw new Error(`The voice model could not download (${t.status}).`);let r=Number(t.headers.get("content-length"))||0,n=[],o=0,i=t.body.getReader();for(;;){let{done:u,value:p}=await i.read();if(u)break;n.push(p),o+=p.byteLength,r&&e&&e(Math.min(100,Math.round(o/r*100)))}let s=new Uint8Array(o),h=0;for(let u of n)s.set(u,h),h+=u.byteLength;return s}async function Qe(a){let e=await L0(),t=await e?.match(a);if(t)return t.arrayBuffer();let r=await O0(a);return await C0(e,a,r),r.buffer}async function gt(a,e){let t=await L0(),r=await t?.match(a.weights);if(r)return Ie(await r.arrayBuffer());let n=await O0(a.weights,e);return await C0(t,a.weights,n),Ie(n.buffer)}function bt(a,e){if(e(a)<=j)return[a];let t=[],r="",n=o=>{if(r&&e(r+o)>j&&(t.push(r),r=""),e(o)<=j)r+=o;else if(/\s/.test(o.trim()))for(let i of o.split(/(?<=\s)/))n(i);else for(let i=0;i<o.length;i+=32)n(o.slice(i,i+32))};for(let o of a.match(/[^.!?]+(?:[.!?]+|$)\s*/g)??[a])n(o);return r&&t.push(r),t.map(o=>o.trim()).filter(Boolean)}var Ae=class a{engine;tokenizer;catalog;lost;constructor(e,t,r,n){this.engine=e,this.tokenizer=t,this.catalog=r,this.lost=n}static async load({model:e="phonon-31m-en",progress:t}={}){let r=Q[e],o=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"}).catch(()=>null);if(!o)throw new Error("This browser has no WebGPU.");let i=wt(o),s=await o.requestDevice({requiredFeatures:i?["subgroups"]:[]}),[h,u,p]=await Promise.all([gt(r,t??(()=>{})),Qe(y0(r)).then(d=>new $e(JSON.parse(new TextDecoder().decode(d)))),Qe(M0).then(d=>new TextDecoder().decode(d).split(`
`).filter(Boolean).map(l=>JSON.parse(l)).filter(l=>r.languages.includes(l.language)).map(l=>({name:l.name,id:l.voice_id,language:l.language,locale:l.locale,gender:l.gender,description:l.description})))]),w=new Me(s,h,{subgroups:i,align:r.align});return await w.ready(),new a(w,u,p,s.lost)}get voices(){return this.catalog}async speak(e,t={}){let{voice:r="Delia",speed:n=1,seed:o=1,onAudio:i,signal:s}=t,h=this.catalog.find(l=>l.name===r||l.id===r);if(!h)throw new Error(`No voice ${r}`);this.engine.hasVoice(h.id)||this.engine.addVoice(h.id,b0(await Qe($0(h.id)))),this.engine.setPace(e0(n));let u=[],p=[],w=[],d=0;for(let[l,g]of bt(e,f=>this.tokenizer.encode(f).length).entries()){s?.throwIfAborted();let f=this.tokenizer.encode(g),m=await this.engine.synthesize(f,h.id,{seed:o+l,words:g.split(/\s+/).filter(Boolean).length,onChunk:i,cancelled:()=>!!s?.aborted});s?.throwIfAborted();let y=G0(m.chunks),M=q0(m.align,f.map(b=>this.tokenizer.pieces[b]),y),R=b=>({...b,start:b.start+d,end:b.end+d});p.push(...M.words.map(R)),w.push(...M.tokens.map(R)),u.push(y),d+=y.length/24e3}return{audio:G0(u),sampleRate:24e3,words:p,tokens:w}}};function G0(a){let e=new Float32Array(a.reduce((r,n)=>r+n.length,0)),t=0;for(let r of a)e.set(r,t),t+=r.length;return e}function wt(a){let e=a.info;return a.features.has("subgroups")&&e.subgroupMinSize===32&&e.subgroupMaxSize===32}var N0={devices:{mac:{label:"MacBook Pro, M3 Max",hello:{device:"mac",ua:"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36",cores:16,gpu:{vendor:"apple",architecture:"metal-3",description:""},isolated:!0}},pc:{label:"PC, RTX 2080 Ti",hello:{device:"pc",ua:"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/156.0.0.0 Safari/537.36",cores:16,gpu:{vendor:"nvidia",architecture:"turing",description:""},isolated:!0}},iphone:{label:"iPhone 17 Pro, Safari 27.2",hello:{device:"iphone",ua:"Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.2 Mobile/15E148 Safari/604.1",cores:4,gpu:{vendor:"apple",architecture:"apple",description:"apple"},isolated:!0}}},results:{mac:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:3.1750000715255737,p25:3.1162500083446503,p75:3.2450000047683716,n:110},warmRealtime:{median:115.1696096769171,p25:113.15902250937599,p75:120.702017022396,n:110},warmTotalSecs:{median:.4299200000166893,p25:.42643874979019164,p75:.43076249995827676,n:10},coldLoadMs:{median:39.38499993085861,p25:38.3299999833107,p75:42.43249997496605,n:10},coldFirstMs:{median:6.567499935626984,p25:6.446249961853027,p75:6.925000011920929,n:10},coldStartMs:{median:45.85249990224838,p25:44.58374997973442,p75:50.09249997138977,n:10},coldPageMs:{median:2.7075000405311584,p25:2.6625000536441803,p75:2.817500054836273,n:10},firstVisitMs:{median:7171.551186092707,p25:7169.03886919799,p75:7175.2938496708375,n:10},firstVisitMeasuredMs:{median:2017.3624999988824,p25:1944.727499996312,p75:2085.556249995716,n:10},firstVisitMB:{median:44.304259,p25:44.304259,p75:44.304259,n:10},firstVisitNetworkMBps:{median:19.74974236426856,p25:18.963348457991003,p75:20.73588592569058,n:10}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:14.902500033378601,p25:14.330000013113022,p75:18.696250051259995,n:110},warmRealtime:{median:30.8000784333127,p25:30.079311820430522,p75:31.757270885693355,n:110},warmTotalSecs:{median:1.8496575000286102,p25:1.8443787499666213,p75:1.8588324999809265,n:10},coldLoadMs:{median:351.1500000357628,p25:346.40750002861023,p75:352.7012500166893,n:10},coldFirstMs:{median:21.97000002861023,p25:21.726250022649765,p75:22.338750034570694,n:10},coldStartMs:{median:373.14250010252,p25:368.22375002503395,p75:374.74499997496605,n:10},coldPageMs:{median:2.67249995470047,p25:2.5399999916553497,p75:2.7350000143051147,n:10},firstVisitMs:{median:10894.980049768903,p25:10891.231324183263,p75:10901.206385215253,n:10},firstVisitMeasuredMs:{median:3459.7150000017136,p25:3406.6987500023097,p75:3565.8112500030547,n:10},firstVisitMB:{median:66.797344,p25:66.797344,p75:66.797344,n:10},firstVisitNetworkMBps:{median:18.45735792155274,p25:17.11081308013756,p75:19.196958530083755,n:10}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:9.807500004768372,p25:9.549999952316284,p75:11.571250021457672,n:110},warmRealtime:{median:14.825656693879449,p25:14.523527519516664,p75:15.03792956698842,n:110},warmTotalSecs:{median:3.939877499938011,p25:3.8754237501621245,p75:3.967627499848604,n:10},coldLoadMs:{median:309.80000001192093,p25:307.8037499785423,p75:314.49499997496605,n:10},coldFirstMs:{median:14.09500002861023,p25:13.831249952316284,p75:14.27499994635582,n:10},coldStartMs:{median:323.83749997615814,p25:321.7262499332428,p75:328.73750004172325,n:10},coldPageMs:{median:2.9325000047683716,p25:2.770000010728836,p75:3.0637499392032623,n:10},firstVisitMs:{median:10861.583309493455,p25:10854.391715255993,p75:10873.091864179316,n:10},firstVisitMeasuredMs:{median:3406.60000000149,p25:3335.3974999999627,p75:3434.2625000001863,n:10},firstVisitMB:{median:66.813988,p25:66.813988,p75:66.813988,n:10},firstVisitNetworkMBps:{median:18.98596720758132,p25:18.114307452751664,p75:19.934717666513095,n:10}},kokoro:{info:"WebGPU, fp32",errors:[],warmFirstMs:{median:207.36500000953674,p25:195.6424999833107,p75:219.3587500154972,n:110},warmRealtime:{median:14.071478433021998,p25:13.318973926002432,p75:14.441482288225238,n:110},warmTotalSecs:{median:4.078314999997616,p25:4.076352499783039,p75:4.092417500168085,n:10},coldLoadMs:{median:597.9274999499321,p25:592.8337499797344,p75:603.1312499940395,n:10},coldFirstMs:{median:248.6350000500679,p25:247.55500003695488,p75:250.78125002980232,n:10},coldStartMs:{median:846.8849999904633,p25:842.2349999845028,p75:854.8075000345707,n:10},coldPageMs:{median:2.76500004529953,p25:2.6287499964237213,p75:3.0587500035762787,n:10},firstVisitMs:{median:54124.318824688846,p25:54068.012708960974,p75:54147.51759422006,n:10},firstVisitMeasuredMs:{median:10902.14499999769,p25:10717.69000000041,p75:11079.444999998435,n:10},firstVisitMB:{median:331.077447,p25:331.077447,p75:331.077447,n:10},firstVisitNetworkMBps:{median:19.18668832311547,p25:18.487976052775018,p75:20.24503788527656,n:10}}},pc:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:6.092500030994415,p25:5.561250001192093,p75:6.851250007748604,n:110},warmRealtime:{median:72.56056121529261,p25:67.49720650684579,p75:78.18004083711425,n:110},warmTotalSecs:{median:.6740724999904633,p25:.66417874994874,p75:.6858324999809264,n:10},coldLoadMs:{median:164.24000000953674,p25:160.62750001251698,p75:171.5037499666214,n:10},coldFirstMs:{median:11.732499986886978,p25:11.577500015497208,p75:13.441249996423721,n:10},coldStartMs:{median:176.80749997496605,p25:173.17125000059605,p75:184.60499997437,n:10},coldPageMs:{median:1420.4925000071526,p25:1418.6875000447035,p75:1426.3150000423193,n:10},firstVisitMs:{median:7346.496513221922,p25:7339.7851338726905,p75:7365.2877608132985,n:10},firstVisitMeasuredMs:{median:2279.797500014305,p25:2174.868750050664,p75:2471.852500036359,n:10},firstVisitMB:{median:44.304259,p25:44.304259,p75:44.304259,n:10},firstVisitNetworkMBps:{median:17.438610158223685,p25:17.078302224871038,p75:18.17311980172149,n:10}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:26.52250000834465,p25:25.697500005364418,p75:28.509999975562096,n:110},warmRealtime:{median:20.721876551442556,p25:19.94918978842568,p75:21.81879188346197,n:110},warmTotalSecs:{median:2.7333600000143052,p25:2.621691249981523,p75:2.741347500011325,n:10},coldLoadMs:{median:671.4875000119209,p25:658.3387500047684,p75:674.3937499672174,n:10},coldFirstMs:{median:30.78999999165535,p25:30.407499983906746,p75:31.52250000834465,n:10},coldStartMs:{median:701.8000000119209,p25:689.8062500059605,p75:705.3724999576807,n:10},coldPageMs:{median:1417.2349999845028,p25:1410.5312500298023,p75:1422.2524999976158,n:10},firstVisitMs:{median:11291.867706018009,p25:11247.378760757365,p75:11330.053724067726,n:10},firstVisitMeasuredMs:{median:4169.990000039339,p25:3906.660000011325,p75:4780.474999979138,n:10},firstVisitMB:{median:66.797344,p25:66.797344,p75:66.797344,n:10},firstVisitNetworkMBps:{median:17.876268161404077,p25:15.744177496348485,p75:18.568815836226882,n:10}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:25.542500019073486,p25:24.32125000655651,p75:31.58625002205372,n:110},warmRealtime:{median:5.796722196959148,p25:5.596017593829441,p75:5.9833478152999735,n:110},warmTotalSecs:{median:10.057502499967814,p25:9.635757500007749,p75:10.215211249917745,n:10},coldLoadMs:{median:534.9350000321865,p25:533.7549999952316,p75:538.9474999904633,n:10},coldFirstMs:{median:34.14749997854233,p25:32.90374997258186,p75:34.56374999880791,n:10},coldStartMs:{median:568.5575000047684,p25:566.5849999636412,p75:574.5374999940395,n:10},coldPageMs:{median:1493.4749999940395,p25:1469.2225000113249,p75:1504.1575000286102,n:10},firstVisitMs:{median:11078.18814838441,p25:11075.05201069029,p75:11092.896737208086,n:10},firstVisitMeasuredMs:{median:3763.1949999928474,p25:3703.587500050664,p75:4063.6325000077486,n:10},firstVisitMB:{median:66.813988,p25:66.813988,p75:66.813988,n:10},firstVisitNetworkMBps:{median:17.881281873165186,p25:16.680633632515544,p75:18.503345710434523,n:10}},kokoro:{info:"WebGPU, fp32",errors:[],warmFirstMs:{median:271.7024999856949,p25:253.52625000476837,p75:289.7374999821186,n:110},warmRealtime:{median:11.021885639622568,p25:9.992253726554184,p75:11.375304172443556,n:110},warmTotalSecs:{median:5.151475000053644,p25:5.094148749932646,p75:5.250540000051259,n:10},coldLoadMs:{median:1540.902499973774,p25:1510.947499960661,p75:1577.4649999886751,n:10},coldFirstMs:{median:478.3599999845028,p25:474.93375001847744,p75:482.71875,n:10},coldStartMs:{median:2021.6774999797344,p25:1987.7074999213219,p75:2054.0862499773502,n:10},coldPageMs:{median:1422.1525000035763,p25:1419.9475000202656,p75:1429.268750011921,n:10},firstVisitMs:{median:55500.7326339963,p25:55456.65040764433,p75:55558.49667716646,n:10},firstVisitMeasuredMs:{median:13709.399999976158,p25:13054.564999997616,p75:13994.09249997139,n:10},firstVisitMB:{median:329.95533,p25:329.953707,p75:329.95533,n:10},firstVisitNetworkMBps:{median:16.383774895257872,p25:14.72115942949406,p75:16.741665022942094,n:10}}},iphone:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:10.720000000000027,p25:10.284999999999997,p75:12.769999999999982,n:110},warmRealtime:{median:26.124510457154834,p25:21.683927516785175,p75:28.45132833446453,n:110},warmTotalSecs:{median:2.09046,p25:1.76812,p75:2.2830500000000002,n:10},coldLoadMs:{median:117.53,p25:98.91999999999999,p75:130.435,n:10},coldFirstMs:{median:28.549999999999997,p25:28.269999999999996,p75:30.400000000000013,n:10},coldStartMs:{median:146.77000000000004,p25:128.31,p75:161.525,n:10},coldPageMs:{median:25.43,p25:18.54499999999999,p75:29.349999999999998,n:10},firstVisitMs:{median:7300.48142046875,p25:7290.636261777345,p75:7310.6212300390625,n:10},firstVisitMeasuredMs:{median:3565.42,p25:3248.825,p75:4462.530000000001,n:10},firstVisitMB:{median:44.304259,p25:44.304259,p75:44.304259,n:10},firstVisitNetworkMBps:{median:10.32219226462792,p25:8.293721163619626,p75:13.304907141352851,n:10}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:43.76999999999998,p25:35.739999999999895,p75:66.44999999999993,n:110},warmRealtime:{median:12.684434130556903,p25:8.407709815930131,p75:15.117454205790903,n:110},warmTotalSecs:{median:5.0526800000000005,p25:4.215245,p75:7.192645000000001,n:10},coldLoadMs:{median:510.02000000000004,p25:357.66499999999996,p75:635.705,n:10},coldFirstMs:{median:38.589999999999975,p25:34.97500000000002,p75:45.52000000000004,n:10},coldStartMs:{median:550.02,p25:393,p75:679.385,n:10},coldPageMs:{median:42.65,p25:24.525,p75:91.91,n:10},firstVisitMs:{median:11313.306032304688,p25:11111.04382798828,p75:11389.751195878907,n:10},firstVisitMeasuredMs:{median:5927.62,p25:5468.015,p75:6158.97,n:10},firstVisitMB:{median:66.775412,p25:66.775412,p75:66.775412,n:10},firstVisitNetworkMBps:{median:9.634019785936122,p25:8.288919647193998,p75:13.198108139960876,n:10}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:31.219999999999914,p25:26.620000000000232,p75:40.15999999999963,n:110},warmRealtime:{median:5.526628444932615,p25:4.563444538163105,p75:7.13722288026017,n:110},warmTotalSecs:{median:10.546879999999998,p25:8.506165000000001,p75:11.888845,n:10},coldLoadMs:{median:524.99,p25:505.37,p75:712.165,n:10},coldFirstMs:{median:42.88000000000011,p25:42.15999999999994,p75:45.03,n:10},coldStartMs:{median:567.7,p25:546.73,p75:755.3800000000001,n:10},coldPageMs:{median:27.74,p25:22.379999999999995,p75:36.075,n:10},firstVisitMs:{median:12003.936086015627,p25:11184.792289003906,p75:12155.791071367188,n:10},firstVisitMeasuredMs:{median:5643.3,p25:5203.065,p75:6441.164999999999,n:10},firstVisitMB:{median:66.775412,p25:66.775412,p75:66.790891,n:10},firstVisitNetworkMBps:{median:10.988238133971585,p25:9.345212544335448,p75:12.989562030831156,n:10}},kokoro:{info:null,errors:["did not finish"],warmFirstMs:null,warmRealtime:null,warmTotalSecs:null,coldLoadMs:null,coldFirstMs:null,coldStartMs:null,coldPageMs:null,firstVisitMs:null,firstVisitMeasuredMs:null,firstVisitMB:null,firstVisitNetworkMBps:null}}},phrases:{mac:{ours:[[2.88,117.5,"h01"],[2.72,114.3,"h02"],[2.56,112.1,"h03"],[2.72,114.1,"h04"],[2.72,112.3,"h05"],[2.64,111.4,"h06"],[2.72,114,"h07"],[3.04,123.7,"h08"],[2.72,112.4,"h09"],[3.2,120.8,"h10"],[27.92,148.8,"rainbow"],[2.8,115.8,"h01"],[2.64,109,"h02"],[2.56,113,"h03"],[2.72,113.7,"h04"],[2.72,115.1,"h05"],[2.64,113.2,"h06"],[2.8,115.7,"h07"],[3.12,126.5,"h08"],[2.64,111.8,"h09"],[3.28,118.8,"h10"],[28.16,149.2,"rainbow"],[2.88,117.1,"h01"],[2.72,115.4,"h02"],[2.56,113.5,"h03"],[2.72,113.1,"h04"],[2.72,116.3,"h05"],[2.64,112,"h06"],[2.8,115.9,"h07"],[3.04,122,"h08"],[2.64,111.9,"h09"],[3.2,120.3,"h10"],[27.44,149.6,"rainbow"],[2.8,119,"h01"],[2.72,111.3,"h02"],[2.56,112.4,"h03"],[2.72,114.6,"h04"],[2.72,114.6,"h05"],[2.64,112.1,"h06"],[2.72,113.8,"h07"],[3.2,121.6,"h08"],[2.64,112.5,"h09"],[3.2,121.7,"h10"],[28.16,148.6,"rainbow"],[2.88,122.4,"h01"],[2.72,114.3,"h02"],[2.64,112.4,"h03"],[2.72,116.5,"h04"],[2.72,112.8,"h05"],[2.64,112.6,"h06"],[2.72,118.5,"h07"],[3.12,127.9,"h08"],[2.64,113.4,"h09"],[3.28,120.5,"h10"],[28.32,148.3,"rainbow"],[2.8,118.1,"h01"],[2.72,113,"h02"],[2.56,114,"h03"],[2.72,115,"h04"],[2.64,112.2,"h05"],[2.64,112.3,"h06"],[2.72,114.5,"h07"],[3.12,126,"h08"],[2.64,113.2,"h09"],[3.2,122.4,"h10"],[27.68,147.8,"rainbow"],[2.88,121.2,"h01"],[2.72,113.3,"h02"],[2.56,114.2,"h03"],[2.72,113.5,"h04"],[2.72,115.3,"h05"],[2.64,112.2,"h06"],[2.8,119.3,"h07"],[2.96,117.9,"h08"],[2.72,112.1,"h09"],[3.2,121.3,"h10"],[27.92,149.4,"rainbow"],[2.88,120.8,"h01"],[2.72,113.3,"h02"],[2.56,114.4,"h03"],[2.72,112.6,"h04"],[2.72,115.4,"h05"],[2.64,113.2,"h06"],[2.8,115.6,"h07"],[3.12,123.7,"h08"],[2.64,110.7,"h09"],[3.2,119.1,"h10"],[27.44,149.2,"rainbow"],[2.8,119.8,"h01"],[2.72,107.9,"h02"],[2.56,115.1,"h03"],[2.72,114,"h04"],[2.72,113.4,"h05"],[2.64,112.4,"h06"],[2.88,119.7,"h07"],[3.04,124.8,"h08"],[2.72,115.5,"h09"],[3.12,127,"h10"],[29.04,149.1,"rainbow"],[2.8,119.9,"h01"],[2.72,115.7,"h02"],[2.56,113.8,"h03"],[2.72,115.2,"h04"],[2.72,113.5,"h05"],[2.56,112.6,"h06"],[2.72,113.8,"h07"],[3.04,122.4,"h08"],[2.72,115.3,"h09"],[3.2,120.9,"h10"],[28.72,149.8,"rainbow"]],gradiumWebgpu:[[2.88,28.9,"h01"],[2.72,30.8,"h02"],[2.56,32.4,"h03"],[2.72,30.4,"h04"],[2.72,30.9,"h05"],[2.64,29.9,"h06"],[2.72,30.9,"h07"],[3.12,30.6,"h08"],[2.8,31.7,"h09"],[3.2,32.8,"h10"],[30.08,32.1,"rainbow"],[2.88,29,"h01"],[2.72,30.8,"h02"],[2.48,31.3,"h03"],[2.72,30.9,"h04"],[2.72,30.6,"h05"],[2.64,29.9,"h06"],[2.72,30.9,"h07"],[3.04,29.6,"h08"],[2.72,31.1,"h09"],[3.28,30.5,"h10"],[29.76,31.8,"rainbow"],[2.8,28.2,"h01"],[2.72,30.8,"h02"],[2.64,30,"h03"],[2.8,31.4,"h04"],[2.72,30.6,"h05"],[2.64,29.6,"h06"],[2.8,31.8,"h07"],[3.12,30.4,"h08"],[2.72,31.1,"h09"],[3.2,32.6,"h10"],[30.48,31.9,"rainbow"],[2.8,28.2,"h01"],[2.72,30.8,"h02"],[2.56,32,"h03"],[2.8,31.4,"h04"],[2.72,30.7,"h05"],[2.64,29.5,"h06"],[2.8,31.4,"h07"],[3.12,30.4,"h08"],[2.64,30.2,"h09"],[3.2,32.4,"h10"],[30.56,31.6,"rainbow"],[2.8,28.3,"h01"],[2.72,30.9,"h02"],[2.56,32.1,"h03"],[2.72,30.7,"h04"],[2.72,30.5,"h05"],[2.56,31.9,"h06"],[2.88,32.8,"h07"],[3.12,30.3,"h08"],[2.72,30.9,"h09"],[3.2,32.7,"h10"],[29.68,32,"rainbow"],[2.8,28.2,"h01"],[2.72,30.9,"h02"],[2.56,32.2,"h03"],[2.8,31.8,"h04"],[2.72,30.7,"h05"],[2.64,29.6,"h06"],[2.72,30.8,"h07"],[3.04,29.5,"h08"],[2.72,30.8,"h09"],[3.12,32,"h10"],[29.36,32,"rainbow"],[2.88,29,"h01"],[2.72,31.1,"h02"],[2.64,29.9,"h03"],[2.72,30.5,"h04"],[2.72,30.7,"h05"],[2.64,29.8,"h06"],[2.72,30.6,"h07"],[3.04,29.4,"h08"],[2.64,29.8,"h09"],[3.2,32.7,"h10"],[30,31.9,"rainbow"],[2.88,28.8,"h01"],[2.72,31,"h02"],[2.56,32.5,"h03"],[2.8,31.4,"h04"],[2.8,31.7,"h05"],[2.64,29.7,"h06"],[2.8,31.5,"h07"],[3.04,29.7,"h08"],[2.64,30.3,"h09"],[3.2,32.8,"h10"],[29.92,32,"rainbow"],[2.88,29.1,"h01"],[2.72,31.1,"h02"],[2.56,32.5,"h03"],[2.72,30.9,"h04"],[2.72,30.5,"h05"],[2.64,29.9,"h06"],[2.88,32.6,"h07"],[3.12,30.5,"h08"],[2.64,29.8,"h09"],[3.28,30.8,"h10"],[30,32.1,"rainbow"],[2.8,28.1,"h01"],[2.72,30.7,"h02"],[2.56,32.4,"h03"],[2.72,30.4,"h04"],[2.72,30.5,"h05"],[2.64,29.8,"h06"],[2.8,31.6,"h07"],[3.04,29.5,"h08"],[2.72,30.8,"h09"],[3.12,31.7,"h10"],[29.68,31.8,"rainbow"]],gradiumCpu:[[2.88,13.9,"h01"],[2.72,14.8,"h02"],[2.56,14.8,"h03"],[2.72,14.6,"h04"],[2.72,14.6,"h05"],[2.64,14.6,"h06"],[2.72,14.8,"h07"],[3.12,14.5,"h08"],[2.72,14.7,"h09"],[3.2,14.7,"h10"],[29.68,14.5,"rainbow"],[2.88,14.7,"h01"],[2.72,15.1,"h02"],[2.48,15,"h03"],[2.72,14.9,"h04"],[2.72,15,"h05"],[2.64,15,"h06"],[2.72,15.1,"h07"],[3.04,14.9,"h08"],[2.72,14.9,"h09"],[3.28,15,"h10"],[29.36,14.7,"rainbow"],[2.8,14.9,"h01"],[2.72,15,"h02"],[2.56,15.1,"h03"],[2.8,15.1,"h04"],[2.72,14.8,"h05"],[2.64,15.1,"h06"],[2.8,15.1,"h07"],[3.12,14.9,"h08"],[2.72,15.2,"h09"],[3.2,15.2,"h10"],[30.4,14.7,"rainbow"],[2.88,14.9,"h01"],[2.72,15,"h02"],[2.48,15,"h03"],[2.8,14.8,"h04"],[2.72,14.8,"h05"],[2.64,15.1,"h06"],[2.8,15.1,"h07"],[3.12,14.9,"h08"],[2.64,15.2,"h09"],[3.2,15.2,"h10"],[29.76,14.7,"rainbow"],[2.8,14.7,"h01"],[2.72,15,"h02"],[2.56,15,"h03"],[2.72,14.9,"h04"],[2.72,15,"h05"],[2.56,14.9,"h06"],[2.88,15.1,"h07"],[3.12,14.8,"h08"],[2.72,15,"h09"],[3.2,15.2,"h10"],[29.44,14.7,"rainbow"],[2.8,14.7,"h01"],[2.72,15.1,"h02"],[2.56,15.1,"h03"],[2.8,15.2,"h04"],[2.64,15,"h05"],[2.64,15,"h06"],[2.72,15.1,"h07"],[3.04,14.9,"h08"],[2.72,15.2,"h09"],[3.12,15.2,"h10"],[29.68,14.7,"rainbow"],[2.8,14.3,"h01"],[2.72,15.2,"h02"],[2.64,15,"h03"],[2.72,15.2,"h04"],[2.72,14.8,"h05"],[2.64,14.9,"h06"],[2.72,14.1,"h07"],[3.04,13.9,"h08"],[2.64,15.2,"h09"],[3.2,14.5,"h10"],[30,14.6,"rainbow"],[2.88,14.8,"h01"],[2.72,14.9,"h02"],[2.56,15,"h03"],[2.72,14.9,"h04"],[2.8,15.1,"h05"],[2.64,14.4,"h06"],[2.8,15,"h07"],[3.04,14.7,"h08"],[2.64,14.7,"h09"],[3.2,15,"h10"],[29.6,14.2,"rainbow"],[2.88,14.4,"h01"],[2.72,14.4,"h02"],[2.56,14.3,"h03"],[2.72,14.8,"h04"],[2.72,14.3,"h05"],[2.64,14.3,"h06"],[2.88,14.3,"h07"],[3.2,14.4,"h08"],[2.64,14.7,"h09"],[3.28,14.5,"h10"],[29.68,14,"rainbow"],[2.8,13.6,"h01"],[2.72,13.7,"h02"],[2.56,13.4,"h03"],[2.72,13.2,"h04"],[2.72,13.1,"h05"],[2.64,14.2,"h06"],[2.72,13.9,"h07"],[3.04,13.3,"h08"],[2.72,13.8,"h09"],[3.04,14.4,"h10"],[29.68,14,"rainbow"]],kokoro:[[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,12.7,"h03"],[3,14.4,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.4,"h07"],[3.15,14.1,"h08"],[2.6,14.4,"h09"],[3.25,14.4,"h10"],[31.8,15.7,"rainbow"],[2.83,13.2,"h01"],[2.88,13.7,"h02"],[2.52,13,"h03"],[3,14.6,"h04"],[2.88,14.4,"h05"],[2.65,13.7,"h06"],[2.83,12.9,"h07"],[3.15,14.5,"h08"],[2.6,14,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,12.9,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.9,"h07"],[3.15,14.5,"h08"],[2.6,14,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,13.2,"h03"],[3,14.5,"h04"],[2.88,14.4,"h05"],[2.65,13.9,"h06"],[2.83,12.9,"h07"],[3.15,14.4,"h08"],[2.6,14.2,"h09"],[3.25,14.5,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,13.7,"h02"],[2.52,12.9,"h03"],[3,14.5,"h04"],[2.88,14.5,"h05"],[2.65,14,"h06"],[2.83,13,"h07"],[3.15,14.6,"h08"],[2.6,13.8,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,12.9,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.9,"h07"],[3.15,14.6,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.7,"rainbow"],[2.83,13.4,"h01"],[2.88,14.3,"h02"],[2.52,13,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,14.1,"h06"],[2.83,12.8,"h07"],[3.15,14.6,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,13,"h03"],[3,14.7,"h04"],[2.88,14.1,"h05"],[2.65,13.9,"h06"],[2.83,12.9,"h07"],[3.15,14.4,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,12.9,"h03"],[3,14.4,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.7,"h07"],[3.15,14.3,"h08"],[2.6,13.8,"h09"],[3.25,14.5,"h10"],[31.8,15.6,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,13.1,"h03"],[3,14.4,"h04"],[2.88,14.3,"h05"],[2.65,13.8,"h06"],[2.83,12.5,"h07"],[3.15,14.4,"h08"],[2.6,13.8,"h09"],[3.25,14.2,"h10"],[31.8,15.5,"rainbow"]]},pc:{ours:[[2.88,80.3,"h01"],[2.72,76.2,"h02"],[2.56,73.4,"h03"],[2.72,78.7,"h04"],[2.72,75.5,"h05"],[2.64,75.6,"h06"],[2.72,75.6,"h07"],[3.04,82.1,"h08"],[2.72,67.1,"h09"],[3.2,80,"h10"],[28.08,101.4,"rainbow"],[2.8,54.4,"h01"],[2.64,50.4,"h02"],[2.56,52.4,"h03"],[2.72,52.8,"h04"],[2.72,72.6,"h05"],[2.64,68.3,"h06"],[2.8,80,"h07"],[3.12,87.2,"h08"],[2.64,69,"h09"],[3.28,78.2,"h10"],[28.08,101.4,"rainbow"],[2.88,71.7,"h01"],[2.72,68.3,"h02"],[2.56,67.5,"h03"],[2.72,66.4,"h04"],[2.72,78.1,"h05"],[2.64,74.6,"h06"],[2.8,79.8,"h07"],[3.04,76.2,"h08"],[2.64,70.1,"h09"],[3.2,80.7,"h10"],[27.92,100.6,"rainbow"],[2.8,58.4,"h01"],[2.72,54,"h02"],[2.56,67.6,"h03"],[2.72,74.5,"h04"],[2.72,71.5,"h05"],[2.64,74.6,"h06"],[2.72,72,"h07"],[3.2,83.6,"h08"],[2.64,76.3,"h09"],[3.2,82,"h10"],[28.16,101.5,"rainbow"],[2.88,72.6,"h01"],[2.72,75.3,"h02"],[2.64,66.1,"h03"],[2.72,76.5,"h04"],[2.72,68.6,"h05"],[2.64,67.8,"h06"],[2.72,64.7,"h07"],[3.12,77,"h08"],[2.64,72.7,"h09"],[3.28,73.6,"h10"],[28.32,100.6,"rainbow"],[2.8,69,"h01"],[2.72,66.5,"h02"],[2.56,74.3,"h03"],[2.72,68.8,"h04"],[2.64,66.2,"h05"],[2.64,74.1,"h06"],[2.72,75.9,"h07"],[3.12,86.2,"h08"],[2.64,67.8,"h09"],[3.2,72.5,"h10"],[27.68,99.6,"rainbow"],[2.88,56.4,"h01"],[2.72,51.5,"h02"],[2.56,54,"h03"],[2.72,53.2,"h04"],[2.72,70.8,"h05"],[2.64,71.4,"h06"],[2.8,77.9,"h07"],[2.96,80.1,"h08"],[2.72,65.4,"h09"],[3.2,79.1,"h10"],[27.92,100.1,"rainbow"],[2.88,58.3,"h01"],[2.72,59.8,"h02"],[2.56,61.2,"h03"],[2.72,65.5,"h04"],[2.72,61.2,"h05"],[2.64,68.7,"h06"],[2.8,69.2,"h07"],[3.12,82.9,"h08"],[2.64,73.4,"h09"],[3.2,80,"h10"],[27.44,101.1,"rainbow"],[2.8,66.6,"h01"],[2.72,65.9,"h02"],[2.56,67.5,"h03"],[2.72,66.5,"h04"],[2.72,67.4,"h05"],[2.64,63.1,"h06"],[2.88,70.8,"h07"],[3.04,75.7,"h08"],[2.72,69.8,"h09"],[3.12,77.6,"h10"],[28,100.3,"rainbow"],[2.8,70.8,"h01"],[2.72,70.4,"h02"],[2.56,68.1,"h03"],[2.72,69.7,"h04"],[2.72,76.7,"h05"],[2.56,73.2,"h06"],[2.72,71.6,"h07"],[3.04,82.1,"h08"],[2.72,68.8,"h09"],[3.2,79.3,"h10"],[28.72,100.9,"rainbow"]],gradiumWebgpu:[[2.88,17.7,"h01"],[2.72,22.7,"h02"],[2.56,23.6,"h03"],[2.72,22.9,"h04"],[2.72,22.9,"h05"],[2.64,22.4,"h06"],[2.72,21.4,"h07"],[3.12,21.8,"h08"],[2.8,23.6,"h09"],[3.2,24.7,"h10"],[30.08,25.1,"rainbow"],[2.88,16.4,"h01"],[2.72,20.1,"h02"],[2.48,20.7,"h03"],[2.72,19.9,"h04"],[2.72,20.2,"h05"],[2.64,18.9,"h06"],[2.72,19.8,"h07"],[3.04,19.1,"h08"],[2.72,17.3,"h09"],[3.28,20.6,"h10"],[29.76,21.3,"rainbow"],[2.8,16.3,"h01"],[2.72,20.1,"h02"],[2.64,19.9,"h03"],[2.8,21,"h04"],[2.72,20.2,"h05"],[2.64,19.1,"h06"],[2.8,19.6,"h07"],[3.12,20.8,"h08"],[2.72,20,"h09"],[3.2,21.7,"h10"],[30.48,22.5,"rainbow"],[2.8,17,"h01"],[2.72,22.7,"h02"],[2.56,23.4,"h03"],[2.8,23.6,"h04"],[2.72,22.7,"h05"],[2.64,22,"h06"],[2.8,22.8,"h07"],[3.12,23.6,"h08"],[2.64,22.2,"h09"],[3.2,24.6,"h10"],[30.56,24.9,"rainbow"],[2.8,14.6,"h01"],[2.72,20.1,"h02"],[2.56,19.9,"h03"],[2.72,19.8,"h04"],[2.72,20.3,"h05"],[2.56,20.9,"h06"],[2.88,21,"h07"],[3.12,21.2,"h08"],[2.72,20.4,"h09"],[3.2,21.7,"h10"],[29.68,22.5,"rainbow"],[2.8,16.1,"h01"],[2.72,19.8,"h02"],[2.56,21.2,"h03"],[2.8,19.9,"h04"],[2.72,21.4,"h05"],[2.64,20.2,"h06"],[2.72,20.3,"h07"],[3.04,21,"h08"],[2.72,20.7,"h09"],[3.12,21.3,"h10"],[29.36,21.8,"rainbow"],[2.88,16.8,"h01"],[2.72,20.4,"h02"],[2.64,18.3,"h03"],[2.72,21.6,"h04"],[2.72,20.4,"h05"],[2.64,20,"h06"],[2.72,20.4,"h07"],[3.04,20.9,"h08"],[2.64,19.1,"h09"],[3.2,22,"h10"],[30,22.5,"rainbow"],[2.88,16.8,"h01"],[2.72,21.4,"h02"],[2.56,21.8,"h03"],[2.8,20.5,"h04"],[2.8,21.3,"h05"],[2.64,19.9,"h06"],[2.8,20.5,"h07"],[3.04,22.4,"h08"],[2.64,20.5,"h09"],[3.2,23.1,"h10"],[29.92,24.3,"rainbow"],[2.88,15.1,"h01"],[2.72,20.8,"h02"],[2.56,21.1,"h03"],[2.72,20.1,"h04"],[2.72,18.5,"h05"],[2.64,20.2,"h06"],[2.88,21.5,"h07"],[3.12,21.4,"h08"],[2.64,20.1,"h09"],[3.28,20.9,"h10"],[30,22.9,"rainbow"],[2.8,15.4,"h01"],[2.72,20.1,"h02"],[2.56,21.3,"h03"],[2.72,20.1,"h04"],[2.72,20.6,"h05"],[2.64,20,"h06"],[2.8,20.7,"h07"],[3.04,19.1,"h08"],[2.72,18.6,"h09"],[3.12,21.8,"h10"],[29.68,22.4,"rainbow"]],gradiumCpu:[[2.8,5.9,"h01"],[2.72,6.1,"h02"],[2.56,5.9,"h03"],[2.72,6,"h04"],[2.72,6.2,"h05"],[2.64,6.1,"h06"],[2.72,6.1,"h07"],[3.12,6.1,"h08"],[2.8,6.2,"h09"],[3.2,6.2,"h10"],[29.6,5.9,"rainbow"],[2.88,5.8,"h01"],[2.72,5.8,"h02"],[2.48,5.9,"h03"],[2.72,5.8,"h04"],[2.72,5.9,"h05"],[2.64,5.9,"h06"],[2.72,6,"h07"],[3.04,5.8,"h08"],[2.72,5.9,"h09"],[3.28,6.2,"h10"],[29.36,6,"rainbow"],[2.88,5.5,"h01"],[2.72,5.9,"h02"],[2.64,5.9,"h03"],[2.8,5.9,"h04"],[2.72,5.9,"h05"],[2.64,5.5,"h06"],[2.8,5.9,"h07"],[3.12,5.4,"h08"],[2.72,5.8,"h09"],[3.2,5.4,"h10"],[30.24,5.9,"rainbow"],[2.88,5.6,"h01"],[2.72,5.3,"h02"],[2.56,5.9,"h03"],[2.8,5.6,"h04"],[2.72,6.2,"h05"],[2.64,6.2,"h06"],[2.8,5.9,"h07"],[3.12,6.1,"h08"],[2.64,6,"h09"],[3.2,6.2,"h10"],[29.76,6,"rainbow"],[2.8,5.6,"h01"],[2.72,5.8,"h02"],[2.56,5.7,"h03"],[2.72,5.6,"h04"],[2.72,5.9,"h05"],[2.56,5.6,"h06"],[2.88,5.8,"h07"],[3.12,5.8,"h08"],[2.72,6,"h09"],[3.2,5.9,"h10"],[29.36,5.7,"rainbow"],[2.8,5,"h01"],[2.72,5.5,"h02"],[2.56,5.6,"h03"],[2.72,5.4,"h04"],[2.64,5.5,"h05"],[2.64,5.7,"h06"],[2.72,5.6,"h07"],[3.04,5.7,"h08"],[2.72,5.8,"h09"],[3.12,5.8,"h10"],[29.68,5.7,"rainbow"],[2.8,5.4,"h01"],[2.72,6,"h02"],[2.64,5.5,"h03"],[2.72,5.6,"h04"],[2.72,5.4,"h05"],[2.64,5.6,"h06"],[2.72,5.6,"h07"],[3.04,5.1,"h08"],[2.64,5.2,"h09"],[3.2,5.6,"h10"],[30,5.6,"rainbow"],[2.88,6,"h01"],[2.72,6.1,"h02"],[2.56,6.1,"h03"],[2.72,6,"h04"],[2.8,6.2,"h05"],[2.64,6.2,"h06"],[2.8,6.2,"h07"],[3.04,6.1,"h08"],[2.64,6.2,"h09"],[3.2,6.1,"h10"],[29.68,6,"rainbow"],[2.88,5.5,"h01"],[2.72,5.7,"h02"],[2.56,5.6,"h03"],[2.72,5.6,"h04"],[2.72,5.7,"h05"],[2.64,5.9,"h06"],[2.88,5.7,"h07"],[3.12,5.7,"h08"],[2.64,5.8,"h09"],[3.28,5.7,"h10"],[29.68,5.8,"rainbow"],[2.8,5.4,"h01"],[2.72,5.4,"h02"],[2.56,5.6,"h03"],[2.72,5.6,"h04"],[2.72,5.7,"h05"],[2.64,5.6,"h06"],[2.8,5.6,"h07"],[3.04,5.5,"h08"],[2.72,6,"h09"],[3.12,5.2,"h10"],[29.68,5.3,"rainbow"]],kokoro:[[2.83,9.2,"h01"],[2.88,8.6,"h02"],[2.52,7.4,"h03"],[3,8.1,"h04"],[2.88,11.9,"h05"],[2.65,8.7,"h06"],[2.83,8.7,"h07"],[3.15,10,"h08"],[2.6,9.3,"h09"],[3.25,10,"h10"],[31.8,11.7,"rainbow"],[2.83,9.6,"h01"],[2.88,10.6,"h02"],[2.52,10.3,"h03"],[3,11.3,"h04"],[2.88,10.4,"h05"],[2.65,11.2,"h06"],[2.83,11.6,"h07"],[3.15,11.6,"h08"],[2.6,11.3,"h09"],[3.25,11.3,"h10"],[31.8,13,"rainbow"],[2.83,9.3,"h01"],[2.88,10.8,"h02"],[2.52,9.4,"h03"],[3,11.6,"h04"],[2.88,11.7,"h05"],[2.65,10.6,"h06"],[2.83,10.2,"h07"],[3.15,11.4,"h08"],[2.6,11.2,"h09"],[3.25,11.7,"h10"],[31.8,12.8,"rainbow"],[2.83,10,"h01"],[2.88,11.1,"h02"],[2.52,9.9,"h03"],[3,11,"h04"],[2.88,10.8,"h05"],[2.65,11.1,"h06"],[2.83,11.1,"h07"],[3.15,11.3,"h08"],[2.6,11.3,"h09"],[3.25,11.8,"h10"],[31.8,13.1,"rainbow"],[2.83,10.2,"h01"],[2.88,11.1,"h02"],[2.52,10,"h03"],[3,11.4,"h04"],[2.88,11.4,"h05"],[2.65,11,"h06"],[2.83,10.1,"h07"],[3.15,8.2,"h08"],[2.6,11,"h09"],[3.25,11.6,"h10"],[31.8,13,"rainbow"],[2.83,8.4,"h01"],[2.88,11,"h02"],[2.52,10.9,"h03"],[3,11.6,"h04"],[2.88,10.8,"h05"],[2.65,11,"h06"],[2.83,11,"h07"],[3.15,10.1,"h08"],[2.6,11.5,"h09"],[3.25,11.8,"h10"],[31.8,13.1,"rainbow"],[2.83,8.1,"h01"],[2.88,9.9,"h02"],[2.52,9.4,"h03"],[3,11.2,"h04"],[2.88,10,"h05"],[2.65,10.5,"h06"],[2.83,11,"h07"],[3.15,11.2,"h08"],[2.6,11.2,"h09"],[3.25,11.2,"h10"],[31.8,12.5,"rainbow"],[2.83,9.8,"h01"],[2.88,10.7,"h02"],[2.52,9.3,"h03"],[3,11.2,"h04"],[2.88,11.2,"h05"],[2.65,11,"h06"],[2.83,9.9,"h07"],[3.15,9.8,"h08"],[2.6,11,"h09"],[3.25,11.5,"h10"],[31.8,13,"rainbow"],[2.83,10.2,"h01"],[2.88,10.7,"h02"],[2.52,9.8,"h03"],[3,11.3,"h04"],[2.88,11.4,"h05"],[2.65,11.2,"h06"],[2.83,9.8,"h07"],[3.15,8.3,"h08"],[2.6,11,"h09"],[3.25,11.5,"h10"],[31.8,12.6,"rainbow"],[2.83,10.3,"h01"],[2.88,11,"h02"],[2.52,10,"h03"],[3,11,"h04"],[2.88,11.5,"h05"],[2.65,11,"h06"],[2.83,9.7,"h07"],[3.15,11.7,"h08"],[2.6,11.3,"h09"],[3.25,11.7,"h10"],[31.8,12.8,"rainbow"]]},iphone:{ours:[[2.88,30.7,"h01"],[2.72,28.7,"h02"],[2.56,28.5,"h03"],[2.72,28.3,"h04"],[2.72,28.3,"h05"],[2.64,27.7,"h06"],[2.72,28.4,"h07"],[3.04,31,"h08"],[2.72,28.3,"h09"],[3.2,30.5,"h10"],[27.92,37.6,"rainbow"],[2.8,29.8,"h01"],[2.64,28.5,"h02"],[2.56,28.5,"h03"],[2.72,28.5,"h04"],[2.72,28.6,"h05"],[2.64,27.7,"h06"],[2.8,29.1,"h07"],[3.12,31.6,"h08"],[2.64,28,"h09"],[3.28,29.6,"h10"],[28.16,37,"rainbow"],[2.88,29.9,"h01"],[2.72,27.5,"h02"],[2.56,26.7,"h03"],[2.72,26.5,"h04"],[2.72,25.3,"h05"],[2.64,26,"h06"],[2.8,25.8,"h07"],[3.04,27.4,"h08"],[2.64,25.2,"h09"],[3.2,28.3,"h10"],[27.68,32,"rainbow"],[2.8,28.7,"h01"],[2.72,25.2,"h02"],[2.56,23.4,"h03"],[2.72,21.7,"h04"],[2.72,21,"h05"],[2.64,20.3,"h06"],[2.72,20.9,"h07"],[3.2,22.5,"h08"],[2.64,19.5,"h09"],[3.2,22.2,"h10"],[28.16,27.3,"rainbow"],[2.88,29.9,"h01"],[2.72,28.2,"h02"],[2.64,27.7,"h03"],[2.72,27.2,"h04"],[2.72,26.1,"h05"],[2.64,25.5,"h06"],[2.72,25.9,"h07"],[3.12,28.7,"h08"],[2.64,25.4,"h09"],[3.28,26.3,"h10"],[28.32,31.7,"rainbow"],[2.8,22.6,"h01"],[2.72,21.5,"h02"],[2.56,21.5,"h03"],[2.72,20.9,"h04"],[2.64,20.9,"h05"],[2.64,20.4,"h06"],[2.72,20.9,"h07"],[3.12,24.4,"h08"],[2.64,21.8,"h09"],[3.2,23.3,"h10"],[27.68,19.9,"rainbow"],[2.88,29.5,"h01"],[2.72,26.7,"h02"],[2.56,25.1,"h03"],[2.72,22.9,"h04"],[2.72,21.5,"h05"],[2.64,21.2,"h06"],[2.8,21.4,"h07"],[2.96,22.3,"h08"],[2.72,20.8,"h09"],[3.2,22.1,"h10"],[27.92,26.1,"rainbow"],[2.88,29.8,"h01"],[2.72,26.6,"h02"],[2.56,25.9,"h03"],[2.72,23.7,"h04"],[2.72,22.5,"h05"],[2.64,20.8,"h06"],[2.8,21.6,"h07"],[3.12,23.3,"h08"],[2.64,20.2,"h09"],[3.2,22.1,"h10"],[27.36,26.1,"rainbow"],[2.8,21.7,"h01"],[2.72,20.7,"h02"],[2.56,20.5,"h03"],[2.72,19.9,"h04"],[2.72,20,"h05"],[2.64,18.8,"h06"],[2.88,20.1,"h07"],[3.04,20.4,"h08"],[2.72,18.5,"h09"],[3.12,20.3,"h10"],[28,23.4,"rainbow"],[2.8,29.1,"h01"],[2.72,28,"h02"],[2.56,28.2,"h03"],[2.72,28.3,"h04"],[2.72,28.4,"h05"],[2.56,28.5,"h06"],[2.72,28.4,"h07"],[3.04,30.9,"h08"],[2.72,28.4,"h09"],[3.2,30.2,"h10"],[27.52,37,"rainbow"]],gradiumWebgpu:[[2.88,19.1,"h01"],[2.72,16.5,"h02"],[2.56,18.3,"h03"],[2.72,17.9,"h04"],[2.72,19.1,"h05"],[2.64,17,"h06"],[2.72,15.7,"h07"],[3.12,17.6,"h08"],[2.8,18.8,"h09"],[3.2,15.8,"h10"],[30.08,17.8,"rainbow"],[2.88,15.9,"h01"],[2.72,14.2,"h02"],[2.48,13.9,"h03"],[2.72,14.7,"h04"],[2.72,14.7,"h05"],[2.64,13.9,"h06"],[2.72,13.4,"h07"],[3.04,13.8,"h08"],[2.72,13.6,"h09"],[3.28,13.9,"h10"],[29.76,13.7,"rainbow"],[2.8,12.1,"h01"],[2.72,16.7,"h02"],[2.64,12.1,"h03"],[2.8,14.1,"h04"],[2.72,12.8,"h05"],[2.64,13.1,"h06"],[2.8,13.3,"h07"],[3.12,12.8,"h08"],[2.72,14.9,"h09"],[3.2,13,"h10"],[30.48,11.8,"rainbow"],[2.8,12.2,"h01"],[2.72,16.1,"h02"],[2.56,15.8,"h03"],[2.8,13.4,"h04"],[2.72,13.4,"h05"],[2.64,12.5,"h06"],[2.8,13.1,"h07"],[3.12,12.9,"h08"],[2.64,11.4,"h09"],[3.2,16.5,"h10"],[30.56,12.8,"rainbow"],[2.8,14.5,"h01"],[2.72,13.9,"h02"],[2.56,14.2,"h03"],[2.72,12,"h04"],[2.72,11.7,"h05"],[2.56,12.5,"h06"],[2.88,11.3,"h07"],[3.12,9.3,"h08"],[2.72,10.3,"h09"],[3.2,9,"h10"],[29.68,10,"rainbow"],[2.8,7.3,"h01"],[2.72,5.7,"h02"],[2.56,7.5,"h03"],[2.8,7.3,"h04"],[2.72,7.5,"h05"],[2.64,6.9,"h06"],[2.72,6.9,"h07"],[3.04,7.2,"h08"],[2.72,7.7,"h09"],[3.12,8.2,"h10"],[29.36,7.5,"rainbow"],[2.88,13.3,"h01"],[2.72,10.3,"h02"],[2.64,11.4,"h03"],[2.72,10.6,"h04"],[2.72,10,"h05"],[2.64,9.4,"h06"],[2.72,8.6,"h07"],[3.04,7.6,"h08"],[2.64,6.7,"h09"],[3.2,7.9,"h10"],[30,8.2,"rainbow"],[2.88,12.4,"h01"],[2.72,14.1,"h02"],[2.56,10,"h03"],[2.8,9.4,"h04"],[2.8,6.4,"h05"],[2.64,5.9,"h06"],[2.8,5.2,"h07"],[3.04,5.9,"h08"],[2.64,5.4,"h09"],[3.2,7.9,"h10"],[29.92,8.4,"rainbow"],[2.88,15.2,"h01"],[2.72,10.5,"h02"],[2.56,10,"h03"],[2.72,8.7,"h04"],[2.72,6.8,"h05"],[2.64,5.4,"h06"],[2.88,5.2,"h07"],[3.12,5.9,"h08"],[2.64,5.2,"h09"],[3.28,7.3,"h10"],[30,8.4,"rainbow"],[2.8,18.2,"h01"],[2.72,18.5,"h02"],[2.56,15.8,"h03"],[2.72,16.5,"h04"],[2.72,16.7,"h05"],[2.64,15.6,"h06"],[2.8,19.4,"h07"],[3.04,16.8,"h08"],[2.72,16,"h09"],[3.12,16.1,"h10"],[29.68,17.2,"rainbow"]],gradiumCpu:[[2.88,7.1,"h01"],[2.72,7.4,"h02"],[2.56,7.3,"h03"],[2.72,7.4,"h04"],[2.72,7.5,"h05"],[2.64,7.5,"h06"],[2.72,7.5,"h07"],[3.12,7.3,"h08"],[2.72,7.4,"h09"],[3.12,7.4,"h10"],[29.44,7.2,"rainbow"],[2.88,7,"h01"],[2.72,7.3,"h02"],[2.48,7.2,"h03"],[2.72,7.2,"h04"],[2.72,7.2,"h05"],[2.64,7.3,"h06"],[2.72,7.2,"h07"],[3.04,7.2,"h08"],[2.72,7.3,"h09"],[3.28,7.2,"h10"],[29.76,7.1,"rainbow"],[2.8,5.5,"h01"],[2.72,5.7,"h02"],[2.64,5.8,"h03"],[2.8,5.8,"h04"],[2.72,5.8,"h05"],[2.64,5.8,"h06"],[2.8,5.9,"h07"],[3.12,5.8,"h08"],[2.72,5.9,"h09"],[3.2,5.9,"h10"],[29.52,5.8,"rainbow"],[2.8,3.7,"h01"],[2.72,3.9,"h02"],[2.56,3.9,"h03"],[2.8,4,"h04"],[2.72,4,"h05"],[2.64,4.1,"h06"],[2.8,4.1,"h07"],[3.12,4,"h08"],[2.64,4.1,"h09"],[3.2,4.1,"h10"],[29.76,4.1,"rainbow"],[2.8,4,"h01"],[2.72,4.2,"h02"],[2.56,4.1,"h03"],[2.72,4.2,"h04"],[2.72,4.2,"h05"],[2.56,4.2,"h06"],[2.88,4.3,"h07"],[3.12,4.2,"h08"],[2.72,4.2,"h09"],[3.2,4.2,"h10"],[29.36,4.2,"rainbow"],[2.8,4.9,"h01"],[2.72,5.2,"h02"],[2.56,5.2,"h03"],[2.8,5.5,"h04"],[2.72,5.7,"h05"],[2.64,5.9,"h06"],[2.72,6.3,"h07"],[3.04,6.1,"h08"],[2.72,6.2,"h09"],[3.12,6.2,"h10"],[29.68,6,"rainbow"],[2.88,4.9,"h01"],[2.72,5,"h02"],[2.64,5.2,"h03"],[2.72,5.5,"h04"],[2.72,5.5,"h05"],[2.64,5.6,"h06"],[2.72,5.6,"h07"],[3.04,4.8,"h08"],[2.64,4.4,"h09"],[3.2,5.1,"h10"],[30,5.1,"rainbow"],[2.88,5.9,"h01"],[2.72,4.8,"h02"],[2.56,4.2,"h03"],[2.8,4.3,"h04"],[2.8,4.7,"h05"],[2.64,4.8,"h06"],[2.72,5.3,"h07"],[3.04,5.2,"h08"],[2.64,5.5,"h09"],[3.2,5.5,"h10"],[29.68,5.4,"rainbow"],[2.88,4,"h01"],[2.72,4.4,"h02"],[2.56,4.5,"h03"],[2.72,4.8,"h04"],[2.72,4.8,"h05"],[2.64,4.9,"h06"],[2.88,4.9,"h07"],[3.12,4.8,"h08"],[2.64,4.9,"h09"],[3.28,4.9,"h10"],[29.44,4.9,"rainbow"],[2.8,7.1,"h01"],[2.72,7.3,"h02"],[2.56,7.3,"h03"],[2.72,7.3,"h04"],[2.72,7.3,"h05"],[2.64,7.3,"h06"],[2.8,7.2,"h07"],[3.04,7.1,"h08"],[2.72,7.4,"h09"],[3.12,7.4,"h10"],[29.84,7.1,"rainbow"]]}},timing:{words:712,meanMs:66.6783707864741,medianMs:64.99999999999928,p90Ms:124.99999999862261,within100ms:.7626404494382022,meanSignedMs:29.55758426963997,medianSignedMs:39.999999999999815,errorsMs:[-50,80,10,70,30,170,80,60,-60,40,105,40,105,85,30,-0,-30,0,-145,120,30,80,80,65,105,-60,20,-25,40,-65,5,85,85,-75,-70,-50,-15,-160,-10,80,-50,-50,80,10,85,70,40,-100,-50,80,-60,70,80,20,90,45,-30,80,80,40,-30,-110,-45,45,50,35,-165,70,105,50,-40,-50,60,-210,-140,75,20,35,-20,75,-20,60,100,-10,30,40,85,5,40,-25,-50,150,105,-165,50,-20,110,20,75,40,80,85,-120,60,-20,-100,-15,-40,80,20,10,55,105,0,65,90,-80,-85,10,65,80,-100,-20,-0,55,80,-60,80,25,-95,105,20,-155,-165,75,45,-40,55,75,-40,65,-60,5,-60,40,-75,20,50,85,80,20,-70,20,90,100,-0,40,115,45,-115,5,20,40,30,55,80,150,120,135,140,-15,100,-90,145,65,35,115,115,10,100,20,25,60,110,55,30,20,75,110,25,120,85,10,-10,-30,95,-15,105,40,105,115,90,110,-40,-10,-105,55,-90,10,105,70,-70,-50,-50,45,-135,-140,65,-75,-10,100,-5,75,75,20,-80,0,100,-15,60,50,-20,120,80,10,120,115,55,-25,-60,35,85,-30,35,-130,70,110,-35,-50,-40,85,-130,-155,30,65,55,-25,115,0,85,120,-20,75,70,140,5,20,110,60,105,110,-25,110,-15,85,30,70,65,105,125,-25,100,-65,-80,45,-55,80,20,20,55,125,-15,100,95,-20,-160,20,150,70,-115,-40,10,60,85,-15,65,95,-75,140,30,-55,-20,95,85,-20,70,70,0,140,-75,45,5,10,-5,30,35,80,135,45,-60,10,95,130,35,80,125,50,-75,65,0,60,75,45,130,100,140,120,105,-5,120,-85,145,110,90,145,125,-30,100,25,30,80,125,60,55,20,90,130,45,120,110,10,0,-30,20,-85,105,40,85,115,95,115,-50,10,-65,80,-100,-5,80,65,-85,-70,-60,45,-120,20,110,-85,-30,100,15,75,95,-35,-70,-30,100,-55,60,75,-10,125,40,-20,100,125,45,-80,-110,-5,45,10,60,-125,50,105,30,-50,-20,110,-160,-155,50,35,80,-20,80,-10,85,100,-15,70,90,125,25,30,20,-20,95,100,-70,90,20,100,20,105,35,140,150,-15,85,-30,-40,35,-35,100,30,-0,35,110,10,85,100,0,-145,30,150,105,-60,-5,0,75,100,-20,85,95,-75,140,20,-240,-30,80,65,-40,95,85,20,75,-60,45,0,10,-25,30,110,60,80,-35,-115,20,80,100,-0,50,140,70,-75,50,20,65,40,15,100,100,85,140,120,5,125,-70,150,125,90,145,120,-10,80,50,55,20,135,100,50,-30,45,115,35,120,85,10,5,-20,10,-115,85,20,70,90,75,140,-50,15,-20,50,-65,25,115,105,-95,-70,-55,10,-155,5,115,-95,-40,80,0,65,55,-35,-120,-10,100,-75,65,60,-0,130,45,-10,95,90,35,-65,-170,10,65,-30,40,-135,60,135,20,-20,-50,30,-235,-150,75,60,65,-5,115,-10,80,100,-30,70,60,140,25,20,-30,-35,140,70,-95,80,-45,100,-0,-90,15,95,140,-75,85,-5,-75,-15,-40,70,10,60,40,85,-20,115,100,-115,-115,20,50,80,-120,0,10,35,80,-40,90,80,-85,90,10,-110,-40,90,60,-30,55,60,-20,80,-125,5,0,30,-20,30,70,80,115,30,-60,20,85,140,5,45,125,55,-110,25,10,-10,40,45,95,150,120,125,120,-15,80,-85,145,105,95,160,120]},quality:{device:"mac",clips:{gradiumCpu:110,gradiumWebgpu:110,kokoro:110,ours:110},wer:{gradiumCpu:[.0016853932584269663,0,.004373566264840013],gradiumWebgpu:[.0022471910112359553,0,.005362002209207331],kokoro:[.011235955056179775,.008196511569374298,.015132599496159143],ours:[0,0,0]},werOursMinusGradiumWebgpu:[-.0022471910112359553,-.00528785865989347,0],speakerSimilarity:{"ours~gradiumWebgpu":[.9883882677916325,.9879526199867027,.9887953618198935],"gradiumWebgpu~gradiumWebgpu":[.9885844175261681,.9881791681758683,.9889836907416882],"ours~ours":[.9882225721773475,.9878083817374826,.9886148152402555],"gradiumCpu~gradiumWebgpu":[.9881263611292598,.9876704000102149,.9885679969658153]},utmos:{gradiumCpu:[4.4043716343966395,4.382951672293923,4.423995101668617],gradiumWebgpu:[4.40301629413258,4.384446497830478,4.419035808714953],kokoro:[4.4293786829168145,4.422673785253005,4.436434451775117],ours:[4.400310355966742,4.382847055413507,4.416018287376923]}}};var j0={source:"Harvard sentences, list 1 (IEEE Recommended Practice for Speech Quality Measurements, 1969), and the first paragraph of the Rainbow Passage (Fairbanks, Voice and Articulation Drillbook, 1960).",cold:"h01",phrases:[{id:"h01",text:"The birch canoe slid on the smooth planks."},{id:"h02",text:"Glue the sheet to the dark blue background."},{id:"h03",text:"It's easy to tell the depth of a well."},{id:"h04",text:"These days a chicken leg is a rare dish."},{id:"h05",text:"Rice is often served in round bowls."},{id:"h06",text:"The juice of lemons makes fine punch."},{id:"h07",text:"The box was thrown beside the parked truck."},{id:"h08",text:"The hogs were fed chopped corn and garbage."},{id:"h09",text:"Four hours of steady work faced us."},{id:"h10",text:"A large size in stockings is hard to sell."},{id:"rainbow",text:"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow."}]};var Be=N0,pe=Be.results,V0=Be.phrases,z0=Be.timing,K0=Be.quality,I0=j0.phrases,_e=a=>{let e=Object.values(pe).find(t=>t[a]?.firstVisitMB)?.[a]?.firstVisitMB;return e?e.median.toFixed(1):"?"};var H0=a=>a==="phonon-31m-en"?_e("ours"):"54.3",H=[{id:"ours",name:"blixt/phonon",short:"blixt/phonon",note:"WebGPU"},{id:"gradiumWebgpu",name:"phonon-tts WebGPU",short:"phonon-tts",note:"WebGPU"},{id:"gradiumCpu",name:"phonon-tts CPU",short:"phonon-tts",note:"CPU"},{id:"kokoro",name:"Kokoro 82M",short:"Kokoro 82M",note:"different model"}],Se=[{id:"mac",name:"M3 Max",short:"M3 Max",full:"MacBook Pro, M3 Max, Chrome"},{id:"pc",name:"RTX 2080 Ti",short:"2080 Ti",full:"PC, RTX 2080 Ti, Chrome"},{id:"iphone",name:"iPhone 17 Pro",short:"iPhone 17",full:"iPhone 17 Pro, Safari"}],Ze=[{id:"warmRealtime",tab:"Speed",caption:"Seconds of speech per second of compute, warm.",unit:"x"},{id:"warmFirstMs",tab:"First audio",caption:"Milliseconds from the call to the first chunk of audio, warm.",unit:"ms"},{id:"coldStartMs",tab:"Cold start",caption:"Milliseconds for a fresh page to load the runtime and speak one sentence, with the files cached.",unit:"ms"},{id:"firstVisitMs",tab:"First visit",caption:`Seconds for a fresh page with nothing stored to speak, with its download at 50 Mbit/s: ${_e("ours")} MB, ${_e("gradiumWebgpu")} MB (phonon-tts) or ${_e("kokoro")} MB (Kokoro).`,unit:"s"}],de=a=>a>=1e3?`${(a/1e3).toFixed(1)} s`:a>=10?`${Math.round(a)} ms`:`${a.toFixed(1)} ms`,Pe=(a,e)=>e==="x"?`${Math.round(a)}\xD7`:de(a),D0=(a,e)=>e==="x"?`${Math.round(a)}\xD7`:e==="s"?(a/1e3).toFixed(1):a<10?a.toFixed(1):String(Math.round(a));var N=a=>document.getElementById(a),kt="http://www.w3.org/2000/svg";function _(a,e={},t){let r=document.createElementNS(kt,a);for(let[n,o]of Object.entries(e))r.setAttribute(n,String(o));return t?.append(r),r}function Re(a){let e=document.createElement("div");return e.className="tip",e.hidden=!0,a.append(e),{show(t,r,n){e.innerHTML=t,e.hidden=!1;let o=e.offsetWidth/2;e.style.left=`${Math.min(Math.max(r,o),a.clientWidth-o)}px`,e.style.top=`${n}px`},hide:()=>e.hidden=!0}}function Fe(a,e){let t="",r=(n=!1)=>{let o=Math.round(a.clientWidth),i=Math.round(a.clientHeight);o&&i&&(n||`${o}x${i}`!==t)&&(t=`${o}x${i}`,a.querySelector("svg")?.remove(),e(o,i))};return new ResizeObserver(()=>r()).observe(a),r(),()=>r(!0)}function Y0(a,e){let t=[...a.querySelectorAll(".tabs button")],r=a.querySelectorAll(".captions .caption");t.forEach((n,o)=>{n.onclick=()=>{t.forEach(i=>i.setAttribute("aria-selected",String(i===n))),r.length>1&&r.forEach((i,s)=>i.classList.toggle("off",s!==o)),e(o)}})}function xt(a,e){let t=(s,h)=>s.x<h.x+h.width+1&&h.x<s.x+s.width+1&&s.y<h.y+h.height&&h.y<s.y+s.height,r=[...a.querySelectorAll("rect.bar, rect.dnf")].map(s=>({g:s.parentElement,b:s.getBBox()})),n=[...a.querySelectorAll("text.val")],o=s=>!!s.parentElement.querySelector(".s-ours");n.sort((s,h)=>Number(o(h))-Number(o(s)));let i=[];for(let s of n){let h=s.getBBox(),u=h.x<0?-h.x:h.x+h.width>e?e-h.x-h.width:0;if(u&&(s.setAttribute("x",String(Number(s.getAttribute("x"))+u)),h=s.getBBox()),o(s))a.append(s);else if(i.some(p=>t(h,p))||r.some(p=>p.g!==s.parentElement&&t(h,p.b))){s.remove();continue}i.push(h)}}var Te={};function Mt(a){let e=a.querySelector("[data-plot]"),t=Re(e),r=Ze[0];Y0(a,i=>{r=Ze[i],o()});let o=Fe(e,(i,s)=>{let h=[...Se.map(c=>({id:c.id,name:c.name,short:c.short,full:c.full,you:!1}))],u=Te[r.id];u&&h.push({id:"you",name:"This device",short:"This device",full:"This device",you:!0});let p=26,w=36,d=(c,A)=>c==="you"?A==="ours"&&u?{median:u,p25:u,p75:u,n:1}:null:pe[c]?.[A]?.[r.id]??null,l=h.flatMap(c=>H.map(A=>d(c.id,A.id))).filter(c=>!!c),g=Math.max(...l.map(c=>c.p75))*1.04,f=s-w,m=c=>Math.min(f-2,p+(s-p-w)*(1-c/g)),y=i<480?14:26,M=u?80:0,R=h.length-(u?1:0),b=Math.min(54,(i-M-y*(h.length-1))/(R*H.length)-4),B=c=>c.you?M:H.length*(b+4)-4,P=h.reduce((c,A)=>c+B(A),0)+y*(h.length-1),T=_("svg",{viewBox:`0 0 ${i} ${s}`,role:"img","aria-label":`${r.tab} by device`}),v=(i-P)/2;for(let c of h){let A=B(c),L=Math.max(...H.map(C=>d(c.id,C.id)?.median??0));(c.you?H.slice(0,1):H).forEach((C,qe)=>{let U=d(c.id,C.id),D=c.you?v+M-b:v+qe*(b+4),W=D+b/2,k=!U&&!!pe[c.id]?.[C.id]?.errors?.length,$=_("g",{},T);if(U){if(_("rect",{class:`bar s-${C.id}`,x:D,y:m(U.median),width:b,height:f-m(U.median),rx:4},$),U.p75>U.p25){_("line",{class:"whisker",x1:W,x2:W,y1:m(U.p25),y2:m(U.p75)},$);for(let F of[U.p25,U.p75])_("line",{class:"whisker",x1:W-4,x2:W+4,y1:m(F),y2:m(F)},$)}_("text",{class:"val",x:W,y:Math.min(m(U.median),m(U.p75))-7,"text-anchor":"middle"},$).textContent=D0(U.median,r.unit)}else k?_("rect",{class:"dnf",x:D+.6,y:m(L),width:b-1.2,height:f-m(L),rx:4},$):_("text",{x:W,y:f-8,"text-anchor":"middle"},$).textContent="\u2013";_("rect",{x:D-2,y:p-20,width:b+4,height:s-p-w+20,fill:"transparent"},$);let q=U?`<b>${Pe(U.median,r.unit)}</b>${U.n>1?` \xB7 ${Pe(U.p25,r.unit)} to ${Pe(U.p75,r.unit)}`:""}`:k?"did not finish":"not measured",E=pe[c.id]?.[C.id],O=c.you?"<br>your last run":r.id==="firstVisitMs"&&E?.firstVisitMeasuredMs&&E.firstVisitNetworkMBps?`<br>measured ${de(E.firstVisitMeasuredMs.median)} at ${E.firstVisitNetworkMBps.median.toFixed(1)} MB/s`:"";$.onpointerenter=$.onpointerdown=()=>t.show(`${C.name} \xB7 ${c.full}<br>${q}${O}`,W,U?Math.min(m(U.median),m(U.p75))-18:f-30),$.onpointerleave=()=>t.hide()}),_("text",{class:c.you?"group you":"group",x:c.you?v+A:v+A/2,y:s-10,"text-anchor":c.you?"end":"middle"},T).textContent=A<92?c.short:c.name,v+=A+y}_("line",{class:"axis",x1:0,x2:i,y1:f,y2:f},T),e.prepend(T),xt(T,i)});return o}function $t(a){let e=a.querySelector("[data-plot]"),t=Re(e),r=new Set,n=Se[0];Y0(a,h=>{n=Se[h],s()});for(let h of a.querySelectorAll(".legend button"))h.onclick=()=>{let u=h.getAttribute("aria-pressed")!=="true";h.setAttribute("aria-pressed",String(u)),u?r.delete(h.dataset.runtime):r.add(h.dataset.runtime),s()};let o=Object.fromEntries(I0.map(h=>[h.id,h.text])),s=Fe(e,(h,u)=>{let g=b=>40+(h-40-10)*Math.log(b/1.5)/Math.log(30),f=b=>8+(u-8-30)*(1-Math.log(b/3)/Math.log(86.66666666666667)),m=_("svg",{viewBox:`0 0 ${h} ${u}`,role:"img","aria-label":`Speed against text length, ${n.full}`}),y=_("g",{class:"grid"},m);for(let b of[2,5,10,20,40])_("line",{x1:g(b),x2:g(b),y1:8,y2:u-30},y),_("text",{x:g(b),y:u-30+18,"text-anchor":"middle"},m).textContent=`${b} s`;for(let b of[5,10,20,50,100,200])_("line",{x1:40,x2:h-10,y1:f(b),y2:f(b)},y),_("text",{x:34,y:f(b)+4,"text-anchor":"end"},m).textContent=`${b}\xD7`;let M=[];for(let b of[...H].reverse()){if(r.has(b.id))continue;let B=_("g",{class:`s-${b.id}`,"fill-opacity":.6},m);for(let[P,T,v]of V0[n.id]?.[b.id]??[]){let c=g(P),A=f(T);_("circle",{cx:c,cy:A,r:3.2},B),M.push({px:c,py:A,rt:b.name,secs:P,speed:T,id:v})}}let R=_("circle",{class:"hot",r:5.5,fill:"none",visibility:"hidden"},m);m.onpointermove=m.onpointerdown=b=>{let B=m.getBoundingClientRect(),P=(b.clientX-B.left)*h/B.width,T=(b.clientY-B.top)*u/B.height,v=M[0],c=1/0;for(let L of M){let S=(L.px-P)**2+(L.py-T)**2;S<c&&(c=S,v=L)}if(!v||c>900)return t.hide(),R.setAttribute("visibility","hidden");R.setAttribute("cx",String(v.px)),R.setAttribute("cy",String(v.py)),R.setAttribute("visibility","visible");let A=o[v.id]??v.id;t.show(`<b>${Math.round(v.speed)}\xD7</b> \xB7 ${v.rt}<br>${v.secs.toFixed(1)} s: \u201C${A.length>44?`${A.slice(0,42)}\u2026`:A}\u201D`,v.px/h*e.clientWidth,v.py/u*e.clientHeight-8)},m.onpointerleave=()=>(t.hide(),R.setAttribute("visibility","hidden")),e.prepend(m)})}function At(a){let e=z0,t=a.querySelector("[data-plot]"),r=Re(t),n=new Map;for(let i of e.errorsMs){let s=Math.max(-300,Math.min(280,Math.floor(i/20)*20));n.set(s,(n.get(s)??0)+1)}let o=Math.max(...n.values());Fe(t,(i,s)=>{let d=m=>8+(i-8-8)*(m+300)/600,l=m=>30+(s-30-30)*(1-m/o),g=_("svg",{viewBox:`0 0 ${i} ${s}`,role:"img","aria-label":"Histogram of word start errors"});for(let m=-300;m<300;m+=20){let y=n.get(m)??0,M=_("g",{},g);_("rect",{class:"bar s-ours",x:d(m)+1,y:l(y),width:d(m+20)-d(m)-2,height:s-30-l(y),rx:2},M),_("rect",{x:d(m),y:30,width:d(m+20)-d(m),height:s-30-30,fill:"transparent"},M),M.onpointerenter=M.onpointerdown=()=>r.show(`<b>${y} words</b><br>${m>0?"+":""}${m} to ${m+20>0?"+":""}${m+20} ms`,d(m+10)/i*t.clientWidth,l(y)/s*t.clientHeight-8),M.onpointerleave=()=>r.hide()}for(let m of i<520?[-200,0,200]:[-300,-200,-100,0,100,200,300])_("text",{x:d(m),y:s-10,"text-anchor":m===-300?"start":m===300?"end":"middle"},g).textContent=`${m>0?"+":""}${m} ms`;let f=e.medianSignedMs;_("line",{class:"median",x1:d(f),x2:d(f),y1:18,y2:s-30},g),_("text",{x:d(f)+6,y:14},g).textContent=`median ${f>0?"+":""}${Math.round(f)} ms (later)`,_("line",{class:"axis",x1:8,x2:i-8,y1:s-30,y2:s-30},g),t.prepend(g)})}function _t(a){let e=K0,t=a.querySelector("[data-plot]"),r=Re(t),n=[{key:"wer",title:"Word errors (Whisper)",max:2,f:o=>`${(o*100).toFixed(2)}%`,scale:100},{key:"utmos",title:"Predicted quality (UTMOS)",max:5,f:o=>o.toFixed(2),scale:1}];Fe(t,(o,i)=>{let s=o>=560,h=s?(o-24)/2:o,u=30,p=Math.min(150,h*.42),w=26+H.length*u,d=_("svg",{viewBox:`0 0 ${o} ${i}`,role:"img","aria-label":"Word error rate and UTMOS per runtime"});n.forEach((l,g)=>{let f=s?g*(h+24):0,m=s?0:g*(w+16);_("text",{class:"group",x:f,y:m+14},d).textContent=l.title;let y=M=>f+p+(h-p-56)*M*l.scale/l.max;H.forEach((M,R)=>{let[b,B,P]=e[l.key][M.id]??[];if(b===void 0)return;let T=m+26+R*u,v=_("g",{},d);_("text",{x:f,y:T+15},v).textContent=M.name,_("rect",{class:`bar s-${M.id}`,x:y(0),y:T+4,width:Math.max(2,y(b)-y(0)),height:u-12,rx:3},v),_("line",{class:"whisker",x1:y(B),x2:y(P),y1:T+u/2-2,y2:T+u/2-2},v),_("text",{class:"val",x:Math.max(y(b),y(P))+6,y:T+15},v).textContent=l.f(b),_("rect",{x:f,y:T,width:h,height:u,fill:"transparent"},v),v.onpointerenter=v.onpointerdown=()=>r.show(`${M.name}<br><b>${l.f(b)}</b> \xB7 95% interval ${l.f(B)} to ${l.f(P)}`,(y(b)+0)/o*t.clientWidth,T/i*t.clientHeight),v.onpointerleave=()=>r.hide()})}),t.prepend(d)})}function Bt(a){let e=a.querySelector("select"),t=new Audio,r=null,n=()=>{t.pause(),r?.setAttribute("aria-pressed","false"),r=null};t.onended=n;for(let o of a.querySelectorAll(".players button"))o.onclick=()=>{let i=r===o;n(),!i&&(t.src=`audio/${o.dataset.runtime}/${e.value}.mp3`,t.play(),r=o,o.setAttribute("aria-pressed","true"))};e.onchange=n}var Ee={en:{name:"English",voice:"Delia",sample:"The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."},fr:{name:"French",voice:"Augustin",sample:"Le petit chat dort au soleil pr\xE8s de la fen\xEAtre. Demain, nous irons au march\xE9 acheter du pain frais."},de:{name:"German",voice:"Mats",sample:"Der kleine Hund spielt im Garten mit einem roten Ball. Morgen fahren wir mit dem Zug nach Berlin."},es:{name:"Spanish",voice:"Marcos",sample:"El ni\xF1o camina por la playa al atardecer. Ma\xF1ana vamos a visitar a mis abuelos en el pueblo."},pt:{name:"Portuguese",voice:"Davi",sample:"O menino corre pelo parque com o seu cachorro. Amanh\xE3 vamos viajar para o Rio de Janeiro."}},W0="phonon-31m-en";function St(a,e){let t=r=>new Option(r.name,r.name);return e.length===1?a.map(t):e.map(r=>{let n=document.createElement("optgroup");return n.label=Ee[r].name,n.append(...a.filter(o=>o.language===r&&o.locale!=="characters").map(t)),n})}async function Pt(a){let e=N("speak"),t=N("words"),r=N("speed");r.oninput=()=>N("speed-out").textContent=`${Number(r.value).toFixed(2).replace(/0$/,"")}\xD7`;let n=N("voice"),o=(k,$,q)=>{let E=N(k);E.querySelector("b").textContent=$,q!==void 0&&(E.querySelector("span").textContent=q)},s=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"}).catch(()=>null);if(!s){e.disabled=!0,o("s-load","\u2013","no WebGPU here");return}let h=s.info,u=`GPU: ${[h.vendor,h.architecture].filter(Boolean).join(" ")||"unknown"}`,p=N("text"),w=N("model"),d=()=>w.value,l=()=>{},g=new Map,f=null,m=()=>f?.voices.find(k=>k.name===n.value)?.language??"en",y=k=>{let $=Q[d()];N("device").textContent=$.license==="Apache-2.0"?u:`${u} \xB7 CC BY-NC 4.0, non-commercial`,k?o("s-load",de(k.ms),k.downloaded?"first visit":"cold start"):o("s-load",`${H0(d())} MB`,"download")};y();let M=Promise.resolve(0),R=k=>M=M.then(async()=>{let $=performance.now(),q=0;return await k.speak("Hello there.",{voice:n.value,onAudio:()=>q||=performance.now()-$}).catch(()=>{}),q}),b=()=>{Object.values(Ee).map($=>$.sample).includes(p.value.trim())&&(p.value=Ee[m()].sample,t.classList.remove("on"))},B=k=>{let $=n.value;n.replaceChildren(...St(k.voices,Q[d()].languages)),n.value=k.voices.some(q=>q.name===$)?$:Ee[Q[d()].languages[0]].voice,b()},P=(k=d())=>{let $=g.get(k);if($)return $;let q=(async()=>{let E=performance.now(),O=!1;k===d()&&o("s-load","\u2026","loading");let F=await Ae.load({model:k,progress:ee=>{O=!0,k===d()&&o("s-load",`${ee}%`,"of the model")}}),K=performance.now()-E;if(k!==d())return F;f=F,B(F);let I=K+await R(F);return!O&&k===W0&&(Te.coldStartMs=I),k===d()&&y({ms:I,downloaded:O}),a(),Object.assign(globalThis,{phonon:F}),F})().catch(E=>{throw g.delete(k),k===d()&&(o("s-load","Failed",String(E?.message??E)),N("s-load").title=String(E?.message??E)),E});return g.set(k,q),q};n.onchange=()=>{b(),f&&R(f)},w.onchange=async()=>{let k=d();l(),f=null,y();let $=await P().catch(()=>null);$&&k===d()&&(f=$,B($),R($))},await(await caches.open(me).catch(()=>null))?.match(Q[d()].weights)&&P().catch(()=>{});let v=null,c=[],A=null,L=null,S=0,C=()=>{for(let k of c)k.stop();c=[],cancelAnimationFrame(S);for(let k of t.children)k.classList.remove("now")},qe=(k,$)=>{let q=v.createBuffer(1,k.length,24e3);q.copyToChannel(k,0);let E=v.createBufferSource();E.buffer=q,E.connect(v.destination),E.start($),c.push(E)},U=(k,$)=>{let q=[...t.children],E=()=>{let O=v.currentTime-$;q.forEach((F,K)=>{let I=O>=k.words[K].start&&O<k.words[K].end;I&&!F.classList.contains("now")&&(F.offsetTop<t.scrollTop||F.offsetTop+F.offsetHeight>t.scrollTop+t.clientHeight)&&(t.scrollTop=F.offsetTop-10),F.classList.toggle("now",I)}),O<k.audio.length/24e3?S=requestAnimationFrame(E):D()};S=requestAnimationFrame(E)},D=()=>{e.textContent="Speak",A=null,c=[]};l=()=>{A?.abort(),C(),D(),t.classList.remove("on")},p.onfocus=()=>t.classList.remove("on");let W=k=>{t.classList.add("on"),t.scrollTop=0,t.replaceChildren(...k.words.flatMap($=>{let q=document.createElement("span");return q.textContent=$.text,q.onclick=()=>{if(!v||L!==k)return;C();let E=$.start,O=v.currentTime+.03,F=v.createBufferSource(),K=v.createBuffer(1,k.audio.length,24e3);K.copyToChannel(k.audio,0),F.buffer=K,F.connect(v.destination),F.start(O,E),c=[F],e.textContent="Stop",U(k,O-E)},[q,document.createTextNode(" ")]}))};e.onclick=async()=>{if(A||c.length){A?.abort(),C(),D();return}v??=new AudioContext({sampleRate:24e3}),v.resume(),A=new AbortController,e.textContent="Stop";let k=A.signal;try{let $=f??await P();if(await M,k.aborted)return;let q=performance.now(),E=0,O=0,F=0;o("s-first","\u2026");let K=()=>{let ee=performance.now()-q;o("s-speed",`${Math.round(O/(ee/1e3))}\xD7`),o("s-audio",`${O.toFixed(1)} s`)};N("s-speed").classList.add("live");let I=await $.speak(p.value,{voice:n.value,speed:Number(r.value),signal:k,onAudio:ee=>{E||(E=performance.now()-q,o("s-first",de(E)),F=v.currentTime+.05),qe(ee,F),F+=ee.length/24e3,O+=ee.length/24e3,K()}});N("s-speed").classList.remove("live"),K(),L=I,d()===W0&&(Te.warmRealtime=O/((performance.now()-q)/1e3),Te.warmFirstMs=E),a(),W(I),U(I,F-I.audio.length/24e3)}catch($){k.aborted||o("s-first","Failed",String($?.message??$)),C(),D()}}}var Tt=Mt(N("bars")),Et={length:$t,timing:At,quality:_t};for(let a of document.querySelectorAll("[data-chart]"))Et[a.dataset.chart]?.(a);var X0=document.getElementById("listen-players");X0&&Bt(X0);Pt(Tt);
