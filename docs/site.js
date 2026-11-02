function xe(a,e,t){let r=e.length,n=new Float32Array(t);for(let o=0;o<t;o++){let s=0;for(let i=0;i<r;i++)s+=a[o*r+i]*e[i];n[o]=s}return n}function $e(a,e){let r=new Float32Array(128);for(let i=0;i<64;i++){let h=1e3*e/1e4**(i/63);r[i]=Math.cos(h),r[64+i]=Math.sin(h)}let n=xe(a.speakersProj,a.speakers,512),o=xe(a.durationProj,r,512);for(let i=0;i<512;i++)n[i]+=o[i];let s=xe(a.latentProj,a.bos,512);for(let i=0;i<512;i++)s[i]+=n[i];return{cond:n,bos:s}}var ee=[[.77,.6],[.8,.45],[.84,.3],[.93,.15],[1,0],[1.07,-.15],[1.22,-.3],[1.3,-.45]];function Ie(a){let e=ee.findIndex(([s])=>a<=s);if(e===0)return ee[0][1];if(e<0)return ee.at(-1)[1];let[t,r]=ee[e-1],[n,o]=ee[e];return r+(a-t)/(n-t)*(o-r)}var K={pos:0,frame:1,end:2,chunk:3,text:4,after:5},q={layernorm:["go","bo"],rows:["rows","pos0"],embedRows:["eo","eso","bo"],noise:["seed","scale"],upsample:["T","wo","wso","mo","so"],shiftRows:["src","count"],gemm:["M","yo","ro","rows","wo","bo","wso"],gemmReduce:["rows","yo","ro","stride","bo"],gemv:["xo","yo","ro","go","so","ho","wo","bo","vo","lgo","lbo","wso"],finishStep:["ewo","ebo","ngo","nbo"]};function j(a,e={}){let t=Array.from({length:(4-a.length%4)%4},(r,n)=>`p${n}`);return`struct P { ${[...a,...t].map(r=>`${r}: ${e[r]??"u32"}`).join(", ")} }`}var Y=32,{pos:ae,frame:X,end:H,chunk:Me,text:He,after:U0}=K,W={none:"",step:`if (st[${H}] != 0u && st[${X}] >= st[${H}]) { return; }`,chunk:`if (st[${H}] != 0u && st[${Me}] / 16u >= st[${H}]) { return; }`},D=`
fn hf(i: u32) -> f32 { let f = unpack2x16float(wv[i >> 3u][(i >> 1u) & 3u]); return select(f.x, f.y, (i & 1u) == 1u); }
fn h4(i: u32) -> vec4<f32> { let q = wv[i >> 3u]; return select(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), (i & 4u) != 0u); }`,We=(a,e,t)=>`(f32(unpack4xI8(wv[(${a}) / 4u + (${t}) / 16u][((${t}) / 4u) % 4u])[(${t}) % 4u]) * hf(${e} + (${t}) / ${Y}u))`,_e=`
fn gelu(v: f32) -> f32 {
  // erf via Abramowitz-Stegun 7.1.26 is too coarse; use a high-precision rational erf.
  let z = v * 0.70710678118654752;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  let erfz = select(r - 1.0, 1.0 - r, z >= 0.0);
  return 0.5 * v * (1.0 + erfz);
}
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }`;function Ae(a,e){return a==="gelu"?`gelu(${e})`:a==="silu"?`silu(${e})`:e}function Pe(a,e){return{none:"",add:`v = r[${e}] + v;`,add_y:"v = y[o] + v;",gate:`v = r[${e}] + gv[p.go + n] * v;`,gate_y:"v = y[o] + gv[p.go + n] * v;",vec:"v = v + hf(p.vo + n);"}[a]}function G0(a,e){if(e===1)return"fn sumSeg(v: f32, li: u32) -> f32 { return v; }";if(a){if(e===32)return"fn sumSeg(v: f32, li: u32) -> f32 { return subgroupAdd(v); }";let t="var t = v;";for(let r=e/2;r>=1;r/=2)t+=` t += subgroupShuffleXor(t, ${r}u);`;return`fn sumSeg(v: f32, li: u32) -> f32 { ${t} return t; }`}return`var<workgroup> r32: array<f32, 256>;
fn sumSeg(v: f32, li: u32) -> f32 {
  r32[li] = v; workgroupBarrier();
  for (var s = ${e/2}u; s > 0u; s >>= 1u) { if ((li % ${e}u) < s) { r32[li] += r32[li + s]; } workgroupBarrier(); }
  let o = r32[li - (li % ${e}u)]; workgroupBarrier(); return o;
}`}function Ve({eps:a,gate:e}){return`
${j(q.layernorm)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
var<workgroup> red: array<f32, 128>;
${D}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &y; _ = &wv; _ = &st;
  ${W[e]}
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
}`}function De(a){let e=a.kind==="codec";return{kv16:!e,at:e?`st[${Me}]`:"0u",gate:W[e?"chunk":"none"],slot:t=>e?`((${t}) % ${a.ring}u)`:`(${t})`,window:e?a.window:0}}function Xe({theta:a,cache:e}){let{kv16:t,at:r,gate:n,slot:o}=De(e),s=o("pos"),i=t?`let kw = (${s} * 512u + h * 64u) / 2u + t;
  kc[kw] = pack2x16float(vec2<f32>(rot(base + 512u, d0, c0, s0), rot(base + 512u, d1, c1, s1)));
  vc[kw] = pack2x16float(vec2<f32>(qkv[base + 1024u + d0], qkv[base + 1024u + d1]));`:`let kb = ${s} * 512u + h * 64u;
  kc[kb + d0] = rot(base + 512u, d0, c0, s0); kc[kb + d1] = rot(base + 512u, d1, c1, s1);
  vc[kb + d0] = qkv[base + 1024u + d0]; vc[kb + d1] = qkv[base + 1024u + d1];`;return`
${j(q.rows)}
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
  ${i}
}`}function Ye({cache:a}){let{kv16:e,at:t,gate:r,slot:n,window:o}=De(a),s=e?"vec2<u32>":"vec4<f32>",i=(h,c)=>e?`kv(${h}[${c}])`:`${h}[${c}]`;return`
${j(q.rows)}
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<${s}>;
@group(0) @binding(2) var<storage, read> vc: array<${s}>;
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
      ks[r][c] = ${i("kc","s")}; vs[r][c] = ${i("vc","s")};
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
}`}function Je({bos:a,pad:e}){return`
${j(q.embedRows)}
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${D}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &ids; _ = &wv; _ = &x;
  let c = gid.x % 512u; let i = gid.x / 512u;
  let id = ids[i];
  var v = 0.0;
  if (id == ${a}u) { v = hf(p.bo + c); } else if (id != ${e}u) { v = ${We("p.eo","p.eso","id * 512u + c")}; }
  x[i * 512u + c] = v;
}`}function Qe(){return`
${j(q.noise,{scale:"f32"})}
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
fn pcg(v: u32) -> u32 { var s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) i: u32) {
  _ = &z; _ = &st;
  ${W.step}
  let f = st[${X}];
  let a = pcg(p.seed ^ pcg(f * 64u + i * 2u)); let b = pcg(a ^ 0x9e3779b9u);
  let u1 = (f32(a >> 8u) + 1.0) / 16777217.0; let u2 = f32(b >> 8u) / 16777216.0;
  z[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2) * p.scale;
}`}function Ze(){return`
${j(q.upsample)}
@group(0) @binding(0) var<storage, read> lats: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${D}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &lats; _ = &wv; _ = &y; _ = &st;
  ${W.chunk}
  let c = gid.x % 512u; let tl = gid.x / 512u;
  if (tl >= p.T) { return; }
  let t = st[${Me}] + tl; let ph = t % 16u; let q = t / 16u;
  var acc = 0.0;
  for (var j = 0u; j < 2u; j++) {
    if (q < j) { continue; }
    let src = q - j;
    for (var i = 0u; i < 32u; i++) {
      let v = lats[src * 32u + i] * hf(p.so + i) + hf(p.mo + i);
      acc += ${We("p.wo","p.wso","((c * 16u + ph) * 2u + j) * 32u + i")} * v;
    }
  }
  y[tl * 512u + c] = acc;
}`}function e0(){return`
${j(q.shiftRows)}
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &b;
  if (gid.x >= p.count) { return; }
  b[gid.x] = b[p.src + gid.x];
}`}function Se({K:a,N:e,act:t="none",bias:r=!1,res:n="none",load:o="linear",conv:s=null,splits:i=1,q8:h,gate:c}){let u=o==="convT",p={cin:0,k:1,dil:1,elu:!1,stride:1,...s},b=u?p.stride:1,l=p.cin,m=k=>`select(exp(${k}) - vec4<f32>(1.0), ${k}, ${k} > vec4<f32>(0.0))`,f=`fn loadA(m: u32, k: u32) -> vec4<f32> {
  if (m >= p.M) { return vec4<f32>(0.0); }
  let v = x[(${o==="linear"?`m * ${a}u + k`:o==="conv"?`(m + (k / ${l}u) * ${p.dil}u) * ${l}u + k % ${l}u`:`(m + 1u - k / ${l}u) * ${l}u + k % ${l}u`}) / 4u];
  return ${p.elu&&o!=="linear"?m("v"):"v"};
}`,g=Math.ceil(a/i/32)*32,M=u?b*a:a,E=h?`
    if (li < 128u) {
      let row = li / 2u; let kq = (li % 2u) * 16u;
      let n = tn + row;
      var q = vec4<u32>(0u); var sc = 0.0;
      if (n < ${e}u) {
        let col = wbase + kt + kq;
        q = wv[p.wo / 4u + (n * ${M}u + col) / 16u];
        sc = hf(p.wso + n * ${M/Y}u + col / ${Y}u);
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
    }`,x=i>1?`y[(sp * p.rows + orow) * ${e}u + n] = v;`:`${r?"v += hf(p.bo + n);":""}
      v = ${Ae(t,"v")};
      let o = p.yo + orow * ${e}u + n;
      ${Pe(n,`p.ro + orow * ${e}u + n`)}
      y[o] = v;`;return`
${j(q.gemm)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
var<workgroup> As: array<array<vec4<f32>, 8>, 64>;
var<workgroup> Bs: array<array<vec4<f32>, 8>, 64>;
${D}
${_e}
${f}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &st;
  ${W[c]}
  let tm = wid.y * 64u;
  let tn = wid.x * 64u;
  let ph = wid.z / ${i}u;
  let sp = wid.z % ${i}u;
  let wbase = ${u?`ph * ${2*l}u`:"0u"};
  let li = lid.y * 16u + lid.x;
  let k0 = sp * ${g}u;
  let k1 = min(${a}u, k0 + ${g}u);
  var acc: array<vec4<f32>, 4>;
  for (var kt = k0; kt < k1; kt += 32u) {
    for (var e = 0u; e < 2u; e++) {
      let idx = li + e * 256u;        // 512 vec4 = 64 rows x 8 vec4 (32 k)
      let row = idx / 8u; let k4 = idx % 8u;
      As[row][k4] = loadA(tm + row, kt + k4 * 4u);
    }
    ${E}
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
    let orow = ${u?`(m * ${b}u + ph)`:"m"};
    for (var j = 0u; j < 4u; j++) {
      let n = tn + lid.x * 4u + j;
      if (n >= ${e}u) { continue; }
      var v = acc[i][j];
      ${x}
    }
  }
}`}function t0({N:a,splits:e,act:t="none",bias:r=!1,res:n="none",gate:o}){return`
${j(q.gemmReduce)}
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
${D}
${_e}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &part; _ = &y; _ = &wv; _ = &r; _ = &st;
  ${W[o]}
  let i = gid.x;
  if (i >= p.rows * ${a}u) { return; }
  let n = i % ${a}u; let row = i / ${a}u;
  var v = 0.0;
  for (var s = 0u; s < ${e}u; s++) { v += part[(s * p.stride + row) * ${a}u + n]; }
  ${r?"v += hf(p.bo + n);":""}
  v = ${Ae(t,"v")};
  let o = p.yo + i;
  ${Pe(n,"p.ro + i")}
  y[o] = v;
}`}function Be(a){let r=a/16,n=Math.min(32,r),o=32/n,s=2*o;return{WG:128,R:2,U:r,LPR:n,RPP:o,RS:s,rowsPerGroup:128/32*s}}function r0({K:a,N:e,act:t="none",bias:r=!1,res:n="none",ln:o=null,combine:s=0,sg:i}){let{WG:h,R:c,U:u,LPR:p,RPP:b,RS:l}=Be(a),m=a/4,w="p.xo / 4u",f=o?`
  // one pass: sum and sum of squares (f32 is ample for 512 values of this size)
  var s1 = 0.0; var s2 = 0.0;
  for (var i = li; i < 128u; i += ${h}u) { let v = x[${w} + i]; s1 += v.x + v.y + v.z + v.w; s2 += dot(v, v); }
  ${i?`s1 = subgroupAdd(s1); s2 = subgroupAdd(s2);
  if ((li & 31u) == 0u) { lnr[li >> 5u] = vec2<f32>(s1, s2); }`:`lnr[li] = vec2<f32>(s1, s2);
  workgroupBarrier();
  for (var s = ${h/2}u; s > 0u; s >>= 1u) { if (li < s) { lnr[li] += lnr[li + s]; } workgroupBarrier(); }`}
  workgroupBarrier();
  var t = vec2<f32>(0.0);
  ${i?`for (var i = 0u; i < ${h/32}u; i++) { t += lnr[i]; }`:"t = lnr[0];"}
  let mean = t.x / 512.0;
  let inv = inverseSqrt(max(t.y / 512.0 - mean * mean, 0.0) + ${o.eps});
  for (var i = li; i < 128u; i += ${h}u) {
    var o = (x[${w} + i] - vec4<f32>(mean)) * inv;
    ${o.affine?"o = o * h4(p.lgo + i * 4u) + h4(p.lbo + i * 4u);":""}
    ${o.modulate?"o = o * (vec4<f32>(1.0) + md[p.so / 4u + i]) + md[p.ho / 4u + i];":""}
    xs[i] = o;
  }
  workgroupBarrier();`:s?`
  // part: [split][head] blocks of 66 floats: 64 dims, max, sum
  for (var c = li; c < 512u; c += ${h}u) {
    let h = c / 64u; let d = c % 64u;
    var mx = -1e30;
    for (var s = 0u; s < ${s}u; s++) { mx = max(mx, r[(s * 8u + h) * 66u + 64u]); }
    var num = 0.0; var den = 0.0;
    for (var s = 0u; s < ${s}u; s++) {
      let b = (s * 8u + h) * 66u;
      let f = exp(r[b + 64u] - mx);
      num += r[b + d] * f; den += r[b + 65u] * f;
    }
    xsf[c] = num / den;
  }
  workgroupBarrier();
  for (var i = li; i < 128u; i += ${h}u) { xs[i] = vec4<f32>(xsf[i * 4u], xsf[i * 4u + 1u], xsf[i * 4u + 2u], xsf[i * 4u + 3u]); }
  workgroupBarrier();`:`
  for (var i = li; i < ${m}u; i += ${h}u) { xs[i] = x[${w} + i]; }
  workgroupBarrier();`;return`${i?"enable subgroups;":""}
${j(q.gemv)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> gv: array<f32>;
@group(0) @binding(6) var<storage, read> md: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> st: array<u32>;
var<workgroup> xs: array<vec4<f32>, ${Math.max(m,128)}>;
${o?`var<workgroup> lnr: array<vec2<f32>, ${i?h/32:h}>;`:""}
${s?"var<workgroup> xsf: array<f32, 512>;":""}
${D}
${_e}
${G0(i,p)}
@compute @workgroup_size(${h})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &gv; _ = &md; _ = &st;
  ${W.step}
  let lane = li & 31u;
  let ul = lane % ${p}u; let sub = lane / ${p}u;
  let n0 = (wid.x * ${h/32}u + (li >> 5u)) * ${l}u + sub;
  ${f}
  var acc: array<f32, ${c}>;
  for (var j = 0u; j < ${c}u; j++) {
    let n = n0 + j * ${b}u;
    if (n < ${e}u) {
      let wb = p.wo / 4u + n * ${u}u;
      for (var i = 0u; i < ${u/p}u; i++) {
        let u = ul + i * ${p}u;
        let q = wv[wb + u];
        let sc = hf(p.wso + n * ${a/Y}u + (u * 16u) / ${Y}u);
        let a = dot(xs[u * 4u], vec4<f32>(unpack4xI8(q.x))) + dot(xs[u * 4u + 1u], vec4<f32>(unpack4xI8(q.y)))
              + dot(xs[u * 4u + 2u], vec4<f32>(unpack4xI8(q.z))) + dot(xs[u * 4u + 3u], vec4<f32>(unpack4xI8(q.w)));
        acc[j] += a * sc;
      }
    }
  }
  for (var j = 0u; j < ${c}u; j++) {
    let tot = sumSeg(acc[j], li);
    let n = n0 + j * ${b}u;
    if (ul == 0u && n < ${e}u) {
      var v = tot${r?" + hf(p.bo + n)":""};
      v = ${Ae(t,"v")};
      let o = p.yo + n;
      ${Pe(n,"p.ro + n")}
      y[o] = v;
    }
  }
}`}function n0({eps:a,threshold:e}){return`
${j(q.finishStep)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> lat: array<f32>;
@group(0) @binding(3) var<storage, read_write> lats: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
var<workgroup> ended: u32;
${D}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &lat; _ = &lats; _ = &st;
  // st is read_write here, so the gate goes through workgroup memory to stay uniform.
  if (li == 0u) { ended = select(0u, 1u, st[${H}] != 0u && st[${X}] >= st[${H}]); }
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
  let f = st[${X}];
  if (li < 32u) { lats[f * 32u + li] = lat[li]; }
  let logit = red[0] + hf(p.ebo);
  workgroupBarrier();
  storageBarrier();
  if (li == 0u) {
    st[${ae}] = st[${ae}] + 1u; st[${X}] = f + 1u;
    if (st[${H}] == 0u && logit > ${e}) { st[${H}] = f + st[${U0}] + 1u; }
  }
}`}function Te({splits:a,theta:e,sg:t,align:r}){let n=o=>`vec2<u32>(pack2x16float(${o}.xy), pack2x16float(${o}.zw))`;return`${t?"enable subgroups;":""}
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
  ${W.step}
  let sp = wid.x; let h = wid.y;
  let pos = st[${ae}];
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
    if (h == 0u && j0 + j >= ${r.lo}u && k < st[${He}]) { al[st[${X}] * ${r.width}u + k] = sc[j]; }`:""}
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
}`}function o0({lo:a}){return`
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> al: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &al; _ = &st;
  let m = st[${ae}] - ${a}u;
  for (var i = li; i < st[${He}]; i += 256u) {
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qb[m * 128u + e] * 0.125, kv(kc[(${a}u + i) * 128u + e])); }
    al[i] = d;
  }
}`}function C0(a){let e=new DataView(a),t=Number(e.getBigUint64(0,!0)),r=JSON.parse(new TextDecoder().decode(new Uint8Array(a,8,t))),n=8+t,o=new Map;for(let[s,i]of Object.entries(r)){if(s==="__metadata__")continue;let[h,c]=i.data_offsets,u=a.slice(n+h,n+c),p;if(i.dtype==="F32")p=new Float32Array(u);else if(i.dtype==="BF16"){let b=new Uint16Array(u),l=new Uint32Array(b.length);for(let m=0;m<b.length;m++)l[m]=b[m]<<16;p=new Float32Array(l.buffer)}else throw new Error(`Unexpected tensor type ${i.dtype} for ${s}`);o.set(s,{shape:i.shape,data:p})}return o}function a0(a){let e=new Float32Array(1),t=new Uint32Array(e.buffer),r=new Uint16Array(a.length);for(let n=0;n<a.length;n++){e[0]=a[n];let o=t[0],s=o>>>16&32768,i=o>>>23&255,h=o&8388607;if(i===255){r[n]=s|31744|(h?512:0);continue}let c=i-127+15;if(c>=31){r[n]=s|31744;continue}if(c<=0){if(c<-10){r[n]=s;continue}h|=8388608;let b=14-c,l=h>>b,m=h&(1<<b)-1,w=1<<b-1;(m>w||m===w&&l&1)&&l++,r[n]=s|l;continue}let u=c<<10|h>>13,p=h&8191;(p>4096||p===4096&&u&1)&&u++,r[n]=s|u}return r}function qe(a){let e=new DataView(a).getUint32(0,!0),t=JSON.parse(new TextDecoder().decode(new Uint8Array(a,4,e))),r=Math.ceil((4+e)/4)*4,n={};for(let[o,s]of t.pace)n[o]=new Float32Array(a,r,s),r+=s*4;return{tensors:t.tensors,pace:n,blob:new Uint8Array(a,r)}}function s0(a){let e=C0(a).get("speaker_wavs");if(!e)throw new Error("The voice file has no speaker_wavs");let[,t,r]=e.shape,n=new Float32Array(r*t);for(let o=0;o<t;o++)for(let s=0;s<r;s++)n[s*t+o]=e.data[o*r+s];return n}var L0="gradium/phonon-31m-en",O0="92741c5b07d085ff840e26fea8ce43110b87852f",i0="gradium/phonon-voices",h0="0ee17205f254b5abc43b266c8a0f215565c352f7",se=(a,e,t)=>`https://huggingface.co/${a}/resolve/${e}/${t}`,j0="blixt/phonon-31m-en-webgpu",N0="f4635a6dddaa6072451e0155c03c97fbe889b6ec",ie="phonon-webgpu",J=se(j0,N0,"phonon-31m-en.int8.bin"),u0=se(L0,O0,"tokenizer.5.0.0.json"),c0=se(i0,h0,"voices.jsonl"),l0=a=>se(i0,h0,`voices/${a}.safetensors`);var z0=128,K0=64,p0=8,I0=4,H0=4,d0=8<<20;function V(a,e){let t=a.beginComputePass();for(let r of e)t.setPipeline(r.pipe.pipeline),t.setBindGroup(0,r.bindGroup),t.dispatchWorkgroups(...r.groups);t.end()}function W0(a,e){let t=[...e.matchAll(/@binding\((\d+)\) var<(storage|uniform)(?:, (read_write|read))?>/g)].map(([,r,n,o])=>({binding:Number(r),visibility:H0,buffer:{type:n==="uniform"?"uniform":o==="read_write"?"storage":"read-only-storage"}}));return a.createBindGroupLayout({entries:t})}function f0(a,e,t){return e===t?{res:`${a}_y`,rbuf:void 0}:{res:a,rbuf:e}}var he=class{device;weights;state;zero;sg;tensors;pipes=new Map;compiling=[];scratchBuf=null;constructor(e,t,r){this.device=e,this.tensors=t.tensors,this.sg=r,this.weights=this.buffer(t.blob.byteLength,t.blob),this.state=this.buffer(32),this.zero=this.buffer(256)}buffer(e,t){let r=this.device.createBuffer({size:Math.max(256,Math.ceil(e/256)*256),usage:z0|p0|I0});return t&&this.device.queue.writeBuffer(r,0,t.buffer,t.byteOffset,t.byteLength),r}u(e,t){let r=t,n=new Uint32Array(Math.max(4,Math.ceil(e.length/4)*4));e.forEach((s,i)=>n[i]=r[s]??0);let o=this.device.createBuffer({size:n.byteLength,usage:K0|p0});return this.device.queue.writeBuffer(o,0,n),o}tensor(e){let t=this.tensors[e];if(!t)throw new Error(`No tensor ${e}`);return t}half(e){let t=this.tensor(e);if(t.scales!==void 0)throw new Error(`${e} is int8`);return t.offset/2}words(e){let t=this.tensor(e);return{wo:t.offset/4,wso:t.scales===void 0?0:t.scales/2,q8:t.scales!==void 0}}q8(e){let t=this.words(e);if(!t.q8)throw new Error(`${e} is not int8`);return t}async ready(){for(;this.compiling.length;)await Promise.all(this.compiling.splice(0))}op(e,t,r){let n=this.pipes.get(e);if(!n){let s=W0(this.device,e),i={layout:s,pipeline:null};this.compiling.push(this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[s]}),compute:{module:this.device.createShaderModule({code:e}),entryPoint:"main"}}).then(h=>{i.pipeline=h})),this.pipes.set(e,i),n=i}let o=this.device.createBindGroup({layout:n.layout,entries:t.map((s,i)=>({binding:i,resource:s instanceof GPUBuffer?{buffer:s}:s}))});return{pipe:n,bindGroup:o,groups:r}}scratch(e){if(e>d0)throw new Error(`Split-K partials of ${e} bytes`);return this.scratchBuf??=this.buffer(d0),{buffer:this.scratchBuf,offset:0,size:Math.ceil(e/256)*256}}gemv(e,t,r,n){let{ln:o,combine:s}=n,{res:i,rbuf:h}=f0(n.res??"none",n.rbuf,r),c=this.q8(t),u=r0({K:n.K,N:n.N,act:n.act,bias:!!n.bias,res:i,combine:s?.splits,ln:o&&{affine:!!o.w,modulate:!!o.mod,eps:o.eps},sg:this.sg}),p=this.u(q.gemv,{xo:n.xo,go:n.go,so:o?.so,ho:o?.ho,wo:c.wo,bo:n.bias?this.half(n.bias):0,vo:n.vname?this.half(n.vname):0,lgo:o?.w?this.half(o.w):0,lbo:o?.b?this.half(o.b):0,wso:c.wso});return this.op(u,[e,this.weights,r,s?.parts??h??this.zero,p,n.gbuf??this.zero,o?.mod??this.zero,this.state],[Math.ceil(n.N/Be(n.K).rowsPerGroup),1,1])}gemm(e,t,r,n){let{M:o,K:s,N:i,load:h="linear",conv:c,yo:u=0,ro:p=0,gate:b}=n,{res:l,rbuf:m}=f0(n.res??"none",n.rbuf,r),w=this.words(t),f=h==="convT"?c.stride:1,g=Math.ceil(i/64)*Math.min(4,Math.ceil(o/64))*f,M=Math.max(1,Math.min(Math.floor(160/g),Math.floor(s/128))),E=[Math.ceil(i/64),Math.ceil(o/64),f*M],x=n.bias?this.half(n.bias):0,k={K:s,N:i,load:h,conv:c,q8:w.q8,gate:b},R=q.gemm;if(M===1){let d=Se({...k,act:n.act,bias:!!n.bias,res:l});return[this.op(d,[e,this.weights,r,m??this.zero,this.u(R,{M:o,yo:u,ro:p,wo:w.wo,bo:x,wso:w.wso}),this.state],E)]}let T=o*f,A=this.scratch(M*T*i*4);return[this.op(Se({...k,splits:M}),[e,this.weights,A,this.zero,this.u(R,{M:o,rows:T,wo:w.wo,wso:w.wso}),this.state],E),this.op(t0({N:i,splits:M,act:n.act,bias:!!n.bias,res:l,gate:b}),[A,r,this.weights,m??this.zero,this.u(q.gemmReduce,{rows:T,yo:u,ro:p,stride:T,bo:x}),this.state],[Math.ceil(T*i/256),1,1])]}layernorm(e,t,r,n,o,s,i){return this.op(Ve({eps:s,gate:i}),[e,t,this.weights,this.u(q.layernorm,{go:this.half(n),bo:this.half(o)}),this.state],[r,1,1])}};var $=512,te=2048,N=125,ce=1024,le=768,O=240,Ee=[64,128,192,256],m0=4294967294,g0=4294967295,Z="1e-5",b0="1e-6",Fe="10000.0",V0="4.0",Q=8,w0=3,v0={lo:N,width:O},ue=2;function Ue(a,e){let t=r=>a.buffer(e*r*4);return{x:t($),h:t($),qkv:t(3*$),q:t($),att:t($),ff:t(te)}}function Ge(a,e,t){let{B:r,rows:n,cache:o}=t,s=o.kind==="codec"?"chunk":"none",i=a.u(q.rows,{rows:n,pos0:t.pos0});return[a.layernorm(r.x,r.h,n,`${e}.ln1.w`,`${e}.ln1.b`,Z,s),...a.gemm(r.h,`${e}.qkv`,r.qkv,{M:n,K:$,N:3*$,gate:s}),a.op(Xe({theta:Fe,cache:o}),[r.qkv,r.q,t.kc,t.vc,a.state,i],[n,1,1]),a.op(Ye({cache:o}),[r.q,t.kc,t.vc,r.att,a.state,i],[Math.ceil(n/16),8,1]),...a.gemm(r.att,`${e}.o`,r.x,{M:n,K:$,N:$,res:"add",rbuf:r.x,gate:s}),a.layernorm(r.x,r.h,n,`${e}.ln2.w`,`${e}.ln2.b`,Z,s),...a.gemm(r.h,`${e}.fc1`,r.ff,{M:n,K:$,N:te,act:"gelu",gate:s}),...a.gemm(r.ff,`${e}.fc2`,t.out?.buf??r.x,{M:n,K:te,N:$,res:"add",rbuf:r.x,yo:t.out?.yo??0,gate:s})]}var pe=class{lats;step;align;g;rows;kc;vc;ids;noise;bos;voiceIn;voiceOps;prefills;tail;cond;ada;fh;ft;z;lat;parts;constructor(e){this.g=e;let t=Ee.at(-1);this.rows=Ue(e,t),this.kc=Array.from({length:6},()=>e.buffer(ce*$*ue)),this.vc=Array.from({length:6},()=>e.buffer(ce*$*ue)),this.cond=e.buffer($*4),this.ada=e.buffer(10240*4),this.fh=e.buffer($*4),this.ft=e.buffer($*4),this.z=e.buffer(128),this.lat=e.buffer(128),this.lats=e.buffer(le*32*4),this.ids=e.buffer(t*4),this.bos=e.buffer($*4),this.parts=e.buffer(Q*8*66*4),this.voiceIn=e.buffer(N*$*4),this.align=e.buffer(le*O*4),this.noise=e.u(q.noise,{}),this.tail=this.tailOps(),this.step=[...Array.from({length:6},(n,o)=>this.stepLayer(o)).flat(),...this.tail],this.voiceOps=[...e.gemm(this.voiceIn,"prompt_proj",this.rows.x,{M:N,K:$,N:$,gate:"none"}),...this.layers(N,0)];let r=e.q8("embed");this.prefills=new Map(Ee.map(n=>[n,[e.op(Je({bos:m0,pad:g0}),[this.ids,e.weights,this.rows.x,e.u(q.embedRows,{eo:r.wo,eso:r.wso,bo:e.half("bos_embed")})],[n*$/256,1,1]),...this.layers(n,N,!0)]]))}layers(e,t,r=!1){let{g:n}=this;return Array.from({length:6},(o,s)=>[...Ge(n,`l${s}`,{B:this.rows,rows:e,kc:this.kc[s],vc:this.vc[s],cache:{kind:"lm"},pos0:t}),...r&&s===w0?[n.op(o0(v0),[this.rows.q,this.kc[s],this.align,n.state],[1,1,1])]:[]]).flat()}voice(e){let{device:t}=this.g,r=N*$*ue,n=this.g.buffer(12*r);t.queue.writeBuffer(this.voiceIn,0,e.buffer,e.byteOffset,e.byteLength);let o=t.createCommandEncoder();V(o,this.voiceOps);for(let s=0;s<6;s++)o.copyBufferToBuffer(this.kc[s],0,n,2*s*r,r),o.copyBufferToBuffer(this.vc[s],0,n,(2*s+1)*r,r);return t.queue.submit([o.finish()]),n}start(e,t,r,n,o,s){let{device:i}=this.g,h=Ee.find(l=>l>t.length),c=new Uint32Array(h).fill(g0);c.set(t),c[t.length]=m0,i.queue.writeBuffer(this.ids,0,c);let u=new DataView(new ArrayBuffer(16));u.setUint32(0,n>>>0,!0),u.setFloat32(4,Math.sqrt(o),!0),i.queue.writeBuffer(this.noise,0,u.buffer);let p=new Uint32Array(8);p[K.pos]=N+t.length,p[K.after]=s,p[K.text]=t.length,i.queue.writeBuffer(this.g.state,0,p);let b=N*$*ue;for(let l=0;l<6;l++)e.copyBufferToBuffer(r,2*l*b,this.kc[l],0,b),e.copyBufferToBuffer(r,(2*l+1)*b,this.vc[l],0,b);V(e,this.prefills.get(h)),e.copyBufferToBuffer(this.rows.x,t.length*$*4,this.bos,0,$*4),e.copyBufferToBuffer(this.bos,0,this.rows.x,0,$*4),V(e,this.tail)}stepLayer(e){let{g:t}=this,r=this.rows,n=`l${e}`;return[t.gemv(r.x,`${n}.qkv`,r.qkv,{K:$,N:3*$,ln:{w:`${n}.ln1.w`,b:`${n}.ln1.b`,eps:Z}}),e===w0?t.op(Te({splits:Q,theta:Fe,sg:t.sg,align:v0}),[r.qkv,this.kc[e],this.vc[e],this.parts,t.state,this.align],[Q,8,1]):t.op(Te({splits:Q,theta:Fe,sg:t.sg}),[r.qkv,this.kc[e],this.vc[e],this.parts,t.state],[Q,8,1]),t.gemv(t.zero,`${n}.o`,r.x,{K:$,N:$,res:"add",rbuf:r.x,combine:{splits:Q,parts:this.parts}}),t.gemv(r.x,`${n}.fc1`,r.ff,{K:$,N:te,act:"gelu",ln:{w:`${n}.ln2.w`,b:`${n}.ln2.b`,eps:Z}}),t.gemv(r.ff,`${n}.fc2`,r.x,{K:te,N:$,res:"add",rbuf:r.x})]}tailOps(){let{g:e}=this,t=this.rows.x,r=Array.from({length:6},(o,s)=>{let i=s*1536;return[e.gemv(this.fh,`fh${s}.fc1.w`,this.ft,{K:$,N:$,bias:`fh${s}.fc1.b`,act:"silu",ln:{w:`fh${s}.ln.w`,b:`fh${s}.ln.b`,eps:b0,mod:this.ada,ho:i,so:i+512}}),e.gemv(this.ft,`fh${s}.fc2.w`,this.fh,{K:$,N:$,bias:`fh${s}.fc2.b`,res:"gate",rbuf:this.fh,gbuf:this.ada,go:i+1024})]}).flat(),n=6*1536;return[e.gemv(t,"fh.cond.w",this.cond,{K:$,N:$,bias:"fh.cond.b",act:"silu",ln:{w:"norm.w",b:"norm.b",eps:Z}}),e.gemv(this.cond,"fh.ada.w",this.ada,{K:$,N:10240,bias:"fh.ada.b"}),e.op(Qe(),[this.z,e.state,this.noise],[1,1,1]),e.gemv(this.z,"fh.in.w",this.fh,{K:32,N:$,bias:"fh.in.b"}),...r,e.gemv(this.fh,"fh.out.w",this.lat,{K:$,N:32,bias:"fh.out.b",res:"add",rbuf:this.z,ln:{eps:b0,mod:this.ada,ho:n,so:n+512}}),e.op(n0({eps:Z,threshold:V0}),[t,e.weights,this.lat,this.lats,e.state,e.u(q.finishStep,{ewo:e.half("eos.w"),ebo:e.half("eos.b"),ngo:e.half("norm.w"),nbo:e.half("norm.b")})],[1,1,1]),e.gemv(this.lat,"latent_proj",t,{K:32,N:$,res:"vec",vname:"cond"})]}};var re=[1,2,4,8],y0=8,ne=1920,D0=251,Ce=512,de=class{audio;g;ops;stages;constructor(e,t){this.g=e;let r=16*y0,n=(x,k,R)=>({ctx:x,rate:k,ch:R,buf:e.buffer((x+k*r)*R*4)}),o=n(6,1,512),s=n(1,1,512),i=n(2,6,256),h=n(0,6,128),c=n(1,6,256),u=n(2,30,128),p=n(0,30,64),b=n(1,30,128),l=n(2,120,64),m=n(0,120,32),w=n(2,120,64);this.stages=[o,s,i,c,u,b,l,w],this.audio=e.buffer(y0*ne*4);let f=Ue(e,r),g=[0,1].map(()=>({kc:e.buffer(Ce*$*4),vc:e.buffer(Ce*$*4)})),M="chunk",E=x=>{let k=16*x,R=(d,v,_,y)=>{if(v.ctx!==y.k-1)throw new Error(`${d} reads ${v.ctx} rows of context`);let P=_ instanceof GPUBuffer?{buf:_,ctx:0}:_;return e.gemm(v.buf,`${d}.w`,P.buf,{M:v.rate*k,K:y.k*y.cin,N:y.cout,bias:`${d}.b`,load:"conv",conv:{cin:y.cin,k:y.k,dil:1,elu:y.elu},res:y.res?"add":"none",rbuf:y.res?.buf,ro:y.res?y.res.ctx*y.cout:0,yo:P.ctx*y.cout,gate:M})},T=(d,v,_,y)=>e.gemm(v.buf,`${d}.w`,_.buf,{M:v.rate*k,K:2*y.cin,N:y.cout,bias:`${d}.b`,load:"convT",conv:{cin:y.cin,elu:!0,stride:y.s},yo:_.ctx*y.cout,gate:M}),A=e.q8("up");return[e.op(Ze(),[t,e.weights,f.x,e.u(q.upsample,{T:k,wo:A.wo,wso:A.wso,mo:e.half("lat_mean"),so:e.half("lat_std")}),e.state],[k*512/256,1,1]),...[0,1].flatMap(d=>Ge(e,`d${d}`,{B:f,rows:k,...g[d],cache:{kind:"codec",window:D0,ring:Ce},pos0:0,out:d===1?{buf:o.buf,yo:o.ctx*512}:void 0})),...R("s0",o,s,{cin:512,cout:512,k:7,elu:!1}),...T("s1",s,i,{cin:512,cout:256,s:6}),...R("s1r1",i,h,{cin:256,cout:128,k:3,elu:!0}),...R("s1r2",h,c,{cin:128,cout:256,k:1,elu:!0,res:i}),...T("s2",c,u,{cin:256,cout:128,s:5}),...R("s2r1",u,p,{cin:128,cout:64,k:3,elu:!0}),...R("s2r2",p,b,{cin:64,cout:128,k:1,elu:!0,res:u}),...T("s3",b,l,{cin:128,cout:64,s:4}),...R("s3r1",l,m,{cin:64,cout:32,k:3,elu:!0}),...R("s3r2",m,w,{cin:32,cout:64,k:1,elu:!0,res:l}),...R("s4",w,this.audio,{cin:64,cout:1,k:3,elu:!0}),...this.stages.map(d=>e.op(e0(),[d.buf,e.u(q.shiftRows,{src:d.rate*k*d.ch,count:d.ctx*d.ch})],[Math.ceil(d.ctx*d.ch/256),1,1]))]};this.ops=new Map(re.map(x=>[x,E(x)]))}reset(e){for(let t of this.stages)e.clearBuffer(t.buf,0,t.ctx*t.ch*4)}encodeChunk(e,t,r){this.g.device.queue.writeBuffer(this.g.state,K.chunk*4,new Uint32Array([16*t])),V(e,this.ops.get(r))}};var k0=3,X0=.3,Y0=1,J0=3,Q0=4,Z0=12.5,x0=8,fe=1,Le=16,me=class{g;model;lm;codec;voices=new Map;stages;pace=Number.NaN;constructor(e,t,{subgroups:r}){this.model=t,this.g=new he(e,t,r),this.lm=new pe(this.g),this.codec=new de(this.g,this.lm.lats);let n=Math.max(...re);this.stages=Array.from({length:k0+1},()=>e.createBuffer({size:Le+n*ne*4,usage:x0|fe})),this.setPace(0)}ready(){return this.g.ready()}addVoice(e,t){if(t.length!==N*$)throw new Error(`Voice ${e} has ${t.length} values`);this.voices.set(e,this.lm.voice(t))}hasVoice(e){return this.voices.has(e)}setPace(e){if(e===this.pace)return;this.pace=e;let{cond:t,bos:r}=$e(this.model.pace,e);for(let[n,o]of[["cond",t],["bos_embed",r]])this.g.device.queue.writeBuffer(this.g.weights,this.model.tensors[n].offset,a0(o))}async synthesize(e,t,{seed:r=1,words:n=1/0,cancelled:o=()=>!1,onChunk:s}={}){let i=this.voices.get(t);if(!i)throw new Error(`Unknown voice ${t}`);if(!e.length)return{chunks:[],align:new Float32Array(0)};if(e.length>O)throw new Error(`Text too long: ${e.length} tokens`);let{device:h}=this.g,{lm:c,codec:u}=this,p=Math.min(le,ce-N-e.length-1,Math.ceil((e.length/3+2)*Z0)),b=n<=Q0?J0:Y0,l=h.createCommandEncoder();u.reset(l),c.start(l,e,i,r,X0,b),h.queue.submit([l.finish()]);let m=1,w=0,f=0,g=null,M=[],E=[...this.stages],x=[],k=()=>{let A=E.pop();l=h.createCommandEncoder();let d=re[Math.min(f,re.length-1)],v=Math.min(p,g??Math.max(m,w+d)),_=[];for(;m<v;m++)_.push(...c.step);_.length&&V(l,_);let y=g??m,P=-1;w<y&&(w+d<=y||g!==null)&&(P=w,u.encodeChunk(l,P,d),w+=d,f++),l.copyBufferToBuffer(this.g.state,0,A,0,16),P>=0&&l.copyBufferToBuffer(u.audio,0,A,Le,d*ne*4),h.queue.submit([l.finish()]),x.push({stage:A,f0:P,size:d,n:m,ready:A.mapAsync(fe)})},R=async()=>{let A=x.shift();await A.ready;let d=A.stage.getMappedRange(),v=new Uint32Array(d,0,4)[K.end];if(g===null&&v&&(g=Math.min(v,p)),g===null&&A.n>=p&&(g=p),A.f0>=0&&(g===null||A.f0<g)){let _=Math.min(A.size,(g??1/0)-A.f0),y=new Float32Array(d,Le,_*ne).slice();M.push(y),s?.(y)}A.stage.unmap(),E.push(A.stage)},T=()=>g!==null&&w>=g&&m>=g;for(;;){for(;!o()&&x.length<k0&&!T();)k();if(!x.length)break;await R()}return o()?null:{chunks:M,align:await this.alignment(g,e.length)}}async alignment(e,t){let{device:r}=this.g,n=e*O*4,o=r.createBuffer({size:n,usage:x0|fe}),s=r.createCommandEncoder();s.copyBufferToBuffer(this.lm.align,0,o,0,n),r.queue.submit([s.finish()]),await o.mapAsync(fe);let i=new Float32Array(o.getMappedRange()),h=new Float32Array(e*t);for(let c=0;c<e;c++)h.set(i.subarray(c*O,c*O+t),c*t);return o.destroy(),h}};var et=new Set([".",",","!","?",";",":",")",'"',"'"]);function tt(a,e,t){let r=new Float32Array(e),n=0;for(let o=0;o<e;o++){let s=a.subarray(o*t,o*t+t),i=0;for(let u=1;u<t;u++)s[u]>s[i]&&(i=u);let h=0,c=0;for(let u=Math.max(0,i-2);u<=Math.min(t-1,i+2);u++){let p=Math.exp(s[u]-s[i]);h+=p*u,c+=p}n=Math.max(n,h/c),r[o]=n}return r}function rt(a){let t=new Float32Array(Math.floor(a.length/240));for(let r=0;r<t.length;r++){let n=0;for(let o=r*240;o<(r+1)*240;o++)n+=a[o]*a[o];t[r]=20*Math.log10(Math.sqrt(n/240)+1e-9)}return t}function $0(a,e,t){let r=e.length,n=r?a.length/r:0,o=t.length/24e3;if(!r||!n)return{tokens:[],words:[]};let s=tt(a,n,r),i=l=>{let m=(l-.04)/.08-3;if(m<=0)return s[0];if(m>=n-1)return s[n-1];let w=Math.floor(m);return s[w]*(1-(m-w))+s[w+1]*(m-w)},h=Array.from({length:r},()=>null);for(let l=0;l<=o;l+=.005){let m=Math.round(i(l));for(let w=0;w<=m&&w<r;w++)h[w]??=l}let c=rt(t),u=[];for(let l=0;l<r;l++){let m=Math.floor((h[l]??o)*100);for(;m<c.length&&c[m]<-45;)m++;u.push(Math.max(h[l]??o,m/100,l?u[l-1]:0))}let p=e.map((l,m)=>({text:l.replaceAll("\u2581",""),start:u[m],end:m+1<r?u[m+1]:o})),b=[];return e.forEach((l,m)=>{let w=l.replaceAll("\u2581",""),f=!l.startsWith("\u2581")||!!w&&[...w].every(g=>et.has(g));if(b.length&&f){let g=b.at(-1);g.text+=w,g.end=p[m].end}else b.push({text:w,start:p[m].start,end:p[m].end})}),{tokens:p,words:b.filter(l=>l.text)}}var nt=/^<0x([0-9A-F]{2})>$/,ge=class{pieces;score=new Map;byte=new Map;maxLen;unk;norm;constructor(e){let t=e.model.vocab;this.pieces=t.map(o=>o[0]),t.forEach(([o,s],i)=>{let h=nt.exec(o);h?this.byte.set(Number.parseInt(h[1],16),i):this.score.set(o,[i,s])}),this.maxLen=Math.max(...this.pieces.map(o=>o.length)),this.unk=e.model.unk_id??0;let r=e.normalizer,n=r?r.type==="Sequence"?r.normalizers:[r]:[];this.norm=n.map(o=>{switch(o.type){case"Replace":{let s=o.pattern.Regex!==void 0?new RegExp(o.pattern.Regex.replaceAll(/\\x\{([0-9A-Fa-f]+)\}/g,"\\u{$1}"),"gu"):new RegExp((o.pattern.String??"").replaceAll(/[.*+?^${}()|[\]\\]/g,"\\$&"),"gu");return i=>i.replaceAll(s,o.content)}case"Strip":return s=>o.strip_left&&o.strip_right?s.trim():o.strip_left?s.trimStart():s.trimEnd();case"Prepend":return s=>s&&o.prepend+s}})}normalize(e){return this.norm.reduce((t,r)=>r(t),e)}encode(e){let t=this.normalize(e).replaceAll(" ","\u2581");t.startsWith("\u2581")||(t="\u2581"+t);let r=[];for(let n of t.split(/(?=▁)/u))n&&r.push(...this.word(n));return r}word(e){let t=Array.from(e),r=t.length,n=new Float64Array(r+1).fill(-1/0),o=new Int32Array(r+1),s=new Int32Array(r+1);n[0]=0;for(let h=0;h<r;h++){if(n[h]===-1/0)continue;let c="";for(let u=h;u<Math.min(r,h+this.maxLen);u++){c+=t[u];let p=this.score.get(c);p&&n[h]+p[1]>n[u+1]&&(n[u+1]=n[h]+p[1],o[u+1]=h,s[u+1]=p[0])}n[h]-100>n[h+1]&&(n[h+1]=n[h]-100,o[h+1]=h,s[h+1]=-1)}let i=[];for(let h=r;h>0;h=o[h])s[h]>=0?i.push([s[h]]):i.push([...new TextEncoder().encode(t[h-1])].map(c=>this.byte.get(c)??this.unk));return i.reverse().flat()}};async function A0(){try{return await caches.open(ie)}catch{return null}}async function P0(a,e,t){try{await a?.put(e,new Response(t))}catch{}}async function S0(a,e){let t=await fetch(a);if(!t.ok||!t.body)throw new Error(`The voice model could not download (${t.status}).`);let r=Number(t.headers.get("content-length"))||0,n=[],o=0,s=t.body.getReader();for(;;){let{done:c,value:u}=await s.read();if(c)break;n.push(u),o+=u.byteLength,r&&e&&e(Math.min(100,Math.round(o/r*100)))}let i=new Uint8Array(o),h=0;for(let c of n)i.set(c,h),h+=c.byteLength;return i}async function Oe(a){let e=await A0(),t=await e?.match(a);if(t)return t.arrayBuffer();let r=await S0(a);return await P0(e,a,r),r.buffer}async function ot(a){let e=await A0(),t=await e?.match(J);if(t)return qe(await t.arrayBuffer());let r=await S0(J,a);return await P0(e,J,r),qe(r.buffer)}function at(a,e){if(e(a)<=O)return[a];let t=[],r="",n=o=>{if(r&&e(r+o)>O&&(t.push(r),r=""),e(o)<=O)r+=o;else if(/\s/.test(o.trim()))for(let s of o.split(/(?<=\s)/))n(s);else for(let s=0;s<o.length;s+=32)n(o.slice(s,s+32))};for(let o of a.match(/[^.!?]+(?:[.!?]+|$)\s*/g)??[a])n(o);return r&&t.push(r),t.map(o=>o.trim()).filter(Boolean)}var be=class a{engine;tokenizer;catalog;lost;constructor(e,t,r,n){this.engine=e,this.tokenizer=t,this.catalog=r,this.lost=n}static async load({progress:e}={}){let r=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"}).catch(()=>null);if(!r)throw new Error("This browser has no WebGPU.");let n=st(r),o=await r.requestDevice({requiredFeatures:n?["subgroups"]:[]}),[s,i,h]=await Promise.all([ot(e??(()=>{})),Oe(u0).then(u=>new ge(JSON.parse(new TextDecoder().decode(u)))),Oe(c0).then(u=>new TextDecoder().decode(u).split(`
`).filter(Boolean).map(p=>JSON.parse(p)).filter(p=>p.language==="en").map(p=>({name:p.name,id:p.voice_id,locale:p.locale,gender:p.gender,description:p.description})))]),c=new me(o,s,{subgroups:n});return await c.ready(),new a(c,i,h,o.lost)}get voices(){return this.catalog}async speak(e,t={}){let{voice:r="Delia",speed:n=1,seed:o=1,onAudio:s,signal:i}=t,h=this.catalog.find(l=>l.name===r||l.id===r);if(!h)throw new Error(`No voice ${r}`);this.engine.hasVoice(h.id)||this.engine.addVoice(h.id,s0(await Oe(l0(h.id)))),this.engine.setPace(Ie(n));let c=[],u=[],p=[],b=0;for(let[l,m]of at(e,w=>this.tokenizer.encode(w).length).entries()){i?.throwIfAborted();let w=this.tokenizer.encode(m),f=await this.engine.synthesize(w,h.id,{seed:o+l,words:m.split(/\s+/).filter(Boolean).length,onChunk:s,cancelled:()=>!!i?.aborted});i?.throwIfAborted();let g=_0(f.chunks),M=$0(f.align,w.map(x=>this.tokenizer.pieces[x]),g),E=x=>({...x,start:x.start+b,end:x.end+b});u.push(...M.words.map(E)),p.push(...M.tokens.map(E)),c.push(g),b+=g.length/24e3}return{audio:_0(c),sampleRate:24e3,words:u,tokens:p}}};function _0(a){let e=new Float32Array(a.reduce((r,n)=>r+n.length,0)),t=0;for(let r of a)e.set(r,t),t+=r.length;return e}function st(a){let e=a.info;return a.features.has("subgroups")&&e.subgroupMinSize===32&&e.subgroupMaxSize===32}var oe={devices:{mac:{label:"MacBook Pro, M3 Max",hello:{device:"mac",ua:"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36",cores:16,gpu:{vendor:"apple",architecture:"metal-3",description:""},isolated:!0}},pc:{label:"PC, RTX 2080 Ti",hello:{device:"pc",ua:"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/156.0.0.0 Safari/537.36",cores:16,gpu:{vendor:"nvidia",architecture:"turing",description:""},isolated:!0}},iphone:{label:"iPhone 17 Pro, Safari 27.2",hello:{device:"iphone",ua:"Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.2 Mobile/15E148 Safari/604.1",cores:4,gpu:{vendor:"apple",architecture:"apple",description:"apple"},isolated:!0}}},results:{mac:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:3.1750000715255737,p25:3.1162500083446503,p75:3.2450000047683716,n:110},warmRealtime:{median:115.1696096769171,p25:113.15902250937599,p75:120.702017022396,n:110},warmTotalSecs:{median:.4299200000166893,p25:.42643874979019164,p75:.43076249995827676,n:10},coldLoadMs:{median:39.38499993085861,p25:38.3299999833107,p75:42.43249997496605,n:10},coldFirstMs:{median:6.567499935626984,p25:6.446249961853027,p75:6.925000011920929,n:10},coldStartMs:{median:45.85249990224838,p25:44.58374997973442,p75:50.09249997138977,n:10},coldPageMs:{median:2.7075000405311584,p25:2.6625000536441803,p75:2.817500054836273,n:10},firstVisitMs:{median:2862.6600000858307,p25:2862.6600000858307,p75:2862.6600000858307,n:1}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:14.902500033378601,p25:14.330000013113022,p75:18.696250051259995,n:110},warmRealtime:{median:30.8000784333127,p25:30.079311820430522,p75:31.757270885693355,n:110},warmTotalSecs:{median:1.8496575000286102,p25:1.8443787499666213,p75:1.8588324999809265,n:10},coldLoadMs:{median:351.1500000357628,p25:346.40750002861023,p75:352.7012500166893,n:10},coldFirstMs:{median:21.97000002861023,p25:21.726250022649765,p75:22.338750034570694,n:10},coldStartMs:{median:373.14250010252,p25:368.22375002503395,p75:374.74499997496605,n:10},coldPageMs:{median:2.67249995470047,p25:2.5399999916553497,p75:2.7350000143051147,n:10},firstVisitMs:{median:3526.295000076294,p25:3526.295000076294,p75:3526.295000076294,n:1}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:9.807500004768372,p25:9.549999952316284,p75:11.571250021457672,n:110},warmRealtime:{median:14.825656693879449,p25:14.523527519516664,p75:15.03792956698842,n:110},warmTotalSecs:{median:3.939877499938011,p25:3.8754237501621245,p75:3.967627499848604,n:10},coldLoadMs:{median:309.80000001192093,p25:307.8037499785423,p75:314.49499997496605,n:10},coldFirstMs:{median:14.09500002861023,p25:13.831249952316284,p75:14.27499994635582,n:10},coldStartMs:{median:323.83749997615814,p25:321.7262499332428,p75:328.73750004172325,n:10},coldPageMs:{median:2.9325000047683716,p25:2.770000010728836,p75:3.0637499392032623,n:10},firstVisitMs:{median:389.55999994277954,p25:389.55999994277954,p75:389.55999994277954,n:1}},kokoro:{info:"WebGPU, fp32",errors:[],warmFirstMs:{median:207.36500000953674,p25:195.6424999833107,p75:219.3587500154972,n:110},warmRealtime:{median:14.071478433021998,p25:13.318973926002432,p75:14.441482288225238,n:110},warmTotalSecs:{median:4.078314999997616,p25:4.076352499783039,p75:4.092417500168085,n:10},coldLoadMs:{median:597.9274999499321,p25:592.8337499797344,p75:603.1312499940395,n:10},coldFirstMs:{median:248.6350000500679,p25:247.55500003695488,p75:250.78125002980232,n:10},coldStartMs:{median:846.8849999904633,p25:842.2349999845028,p75:854.8075000345707,n:10},coldPageMs:{median:2.76500004529953,p25:2.6287499964237213,p75:3.0587500035762787,n:10},firstVisitMs:{median:11953.389999985695,p25:11953.389999985695,p75:11953.389999985695,n:1}}},pc:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:6.092500030994415,p25:5.561250001192093,p75:6.851250007748604,n:110},warmRealtime:{median:72.56056121529261,p25:67.49720650684579,p75:78.18004083711425,n:110},warmTotalSecs:{median:.6740724999904633,p25:.66417874994874,p75:.6858324999809264,n:10},coldLoadMs:{median:164.24000000953674,p25:160.62750001251698,p75:171.5037499666214,n:10},coldFirstMs:{median:11.732499986886978,p25:11.577500015497208,p75:13.441249996423721,n:10},coldStartMs:{median:176.80749997496605,p25:173.17125000059605,p75:184.60499997437,n:10},coldPageMs:{median:1420.4925000071526,p25:1418.6875000447035,p75:1426.3150000423193,n:10},firstVisitMs:{median:3265.775000035763,p25:3265.775000035763,p75:3265.775000035763,n:1}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:26.52250000834465,p25:25.697500005364418,p75:28.509999975562096,n:110},warmRealtime:{median:20.721876551442556,p25:19.94918978842568,p75:21.81879188346197,n:110},warmTotalSecs:{median:2.7333600000143052,p25:2.621691249981523,p75:2.741347500011325,n:10},coldLoadMs:{median:671.4875000119209,p25:658.3387500047684,p75:674.3937499672174,n:10},coldFirstMs:{median:30.78999999165535,p25:30.407499983906746,p75:31.52250000834465,n:10},coldStartMs:{median:701.8000000119209,p25:689.8062500059605,p75:705.3724999576807,n:10},coldPageMs:{median:1417.2349999845028,p25:1410.5312500298023,p75:1422.2524999976158,n:10},firstVisitMs:{median:4371.25,p25:4371.25,p75:4371.25,n:1}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:25.542500019073486,p25:24.32125000655651,p75:31.58625002205372,n:110},warmRealtime:{median:5.796722196959148,p25:5.596017593829441,p75:5.9833478152999735,n:110},warmTotalSecs:{median:10.057502499967814,p25:9.635757500007749,p75:10.215211249917745,n:10},coldLoadMs:{median:534.9350000321865,p25:533.7549999952316,p75:538.9474999904633,n:10},coldFirstMs:{median:34.14749997854233,p25:32.90374997258186,p75:34.56374999880791,n:10},coldStartMs:{median:568.5575000047684,p25:566.5849999636412,p75:574.5374999940395,n:10},coldPageMs:{median:1493.4749999940395,p25:1469.2225000113249,p75:1504.1575000286102,n:10},firstVisitMs:{median:830.8499999046326,p25:830.8499999046326,p75:830.8499999046326,n:1}},kokoro:{info:"WebGPU, fp32",errors:[],warmFirstMs:{median:271.7024999856949,p25:253.52625000476837,p75:289.7374999821186,n:110},warmRealtime:{median:11.021885639622568,p25:9.992253726554184,p75:11.375304172443556,n:110},warmTotalSecs:{median:5.151475000053644,p25:5.094148749932646,p75:5.250540000051259,n:10},coldLoadMs:{median:1540.902499973774,p25:1510.947499960661,p75:1577.4649999886751,n:10},coldFirstMs:{median:478.3599999845028,p25:474.93375001847744,p75:482.71875,n:10},coldStartMs:{median:2021.6774999797344,p25:1987.7074999213219,p75:2054.0862499773502,n:10},coldPageMs:{median:1422.1525000035763,p25:1419.9475000202656,p75:1429.268750011921,n:10},firstVisitMs:{median:15463.824999988079,p25:15463.824999988079,p75:15463.824999988079,n:1}}},iphone:{ours:{info:"WebGPU, int8",errors:[],warmFirstMs:{median:10.720000000000027,p25:10.284999999999997,p75:12.769999999999982,n:110},warmRealtime:{median:26.124510457154834,p25:21.683927516785175,p75:28.45132833446453,n:110},warmTotalSecs:{median:2.09046,p25:1.76812,p75:2.2830500000000002,n:10},coldLoadMs:{median:117.53,p25:98.91999999999999,p75:130.435,n:10},coldFirstMs:{median:28.549999999999997,p25:28.269999999999996,p75:30.400000000000013,n:10},coldStartMs:{median:146.77000000000004,p25:128.31,p75:161.525,n:10},coldPageMs:{median:25.43,p25:18.54499999999999,p75:29.349999999999998,n:10},firstVisitMs:{median:9717.98,p25:9717.98,p75:9717.98,n:1}},gradiumWebgpu:{info:"webgpu, 1 threads (generation runs on the GPU)",errors:[],warmFirstMs:{median:43.76999999999998,p25:35.739999999999895,p75:66.44999999999993,n:110},warmRealtime:{median:12.684434130556903,p25:8.407709815930131,p75:15.117454205790903,n:110},warmTotalSecs:{median:5.0526800000000005,p25:4.215245,p75:7.192645000000001,n:10},coldLoadMs:{median:510.02000000000004,p25:357.66499999999996,p75:635.705,n:10},coldFirstMs:{median:38.589999999999975,p25:34.97500000000002,p75:45.52000000000004,n:10},coldStartMs:{median:550.02,p25:393,p75:679.385,n:10},coldPageMs:{median:42.65,p25:24.525,p75:91.91,n:10},firstVisitMs:{median:18151.86,p25:18151.86,p75:18151.86,n:1}},gradiumCpu:{info:"cpu, 3 threads (default)",errors:[],warmFirstMs:{median:31.219999999999914,p25:26.620000000000232,p75:40.15999999999963,n:110},warmRealtime:{median:5.526628444932615,p25:4.563444538163105,p75:7.13722288026017,n:110},warmTotalSecs:{median:10.546879999999998,p25:8.506165000000001,p75:11.888845,n:10},coldLoadMs:{median:524.99,p25:505.37,p75:712.165,n:10},coldFirstMs:{median:42.88000000000011,p25:42.15999999999994,p75:45.03,n:10},coldStartMs:{median:567.7,p25:546.73,p75:755.3800000000001,n:10},coldPageMs:{median:27.74,p25:22.379999999999995,p75:36.075,n:10},firstVisitMs:{median:1481.9,p25:1481.9,p75:1481.9,n:1}},kokoro:{info:null,errors:["did not finish"],warmFirstMs:null,warmRealtime:null,warmTotalSecs:null,coldLoadMs:null,coldFirstMs:null,coldStartMs:null,coldPageMs:null,firstVisitMs:null}}},phrases:{mac:{ours:[[2.88,117.5,"h01"],[2.72,114.3,"h02"],[2.56,112.1,"h03"],[2.72,114.1,"h04"],[2.72,112.3,"h05"],[2.64,111.4,"h06"],[2.72,114,"h07"],[3.04,123.7,"h08"],[2.72,112.4,"h09"],[3.2,120.8,"h10"],[27.92,148.8,"rainbow"],[2.8,115.8,"h01"],[2.64,109,"h02"],[2.56,113,"h03"],[2.72,113.7,"h04"],[2.72,115.1,"h05"],[2.64,113.2,"h06"],[2.8,115.7,"h07"],[3.12,126.5,"h08"],[2.64,111.8,"h09"],[3.28,118.8,"h10"],[28.16,149.2,"rainbow"],[2.88,117.1,"h01"],[2.72,115.4,"h02"],[2.56,113.5,"h03"],[2.72,113.1,"h04"],[2.72,116.3,"h05"],[2.64,112,"h06"],[2.8,115.9,"h07"],[3.04,122,"h08"],[2.64,111.9,"h09"],[3.2,120.3,"h10"],[27.44,149.6,"rainbow"],[2.8,119,"h01"],[2.72,111.3,"h02"],[2.56,112.4,"h03"],[2.72,114.6,"h04"],[2.72,114.6,"h05"],[2.64,112.1,"h06"],[2.72,113.8,"h07"],[3.2,121.6,"h08"],[2.64,112.5,"h09"],[3.2,121.7,"h10"],[28.16,148.6,"rainbow"],[2.88,122.4,"h01"],[2.72,114.3,"h02"],[2.64,112.4,"h03"],[2.72,116.5,"h04"],[2.72,112.8,"h05"],[2.64,112.6,"h06"],[2.72,118.5,"h07"],[3.12,127.9,"h08"],[2.64,113.4,"h09"],[3.28,120.5,"h10"],[28.32,148.3,"rainbow"],[2.8,118.1,"h01"],[2.72,113,"h02"],[2.56,114,"h03"],[2.72,115,"h04"],[2.64,112.2,"h05"],[2.64,112.3,"h06"],[2.72,114.5,"h07"],[3.12,126,"h08"],[2.64,113.2,"h09"],[3.2,122.4,"h10"],[27.68,147.8,"rainbow"],[2.88,121.2,"h01"],[2.72,113.3,"h02"],[2.56,114.2,"h03"],[2.72,113.5,"h04"],[2.72,115.3,"h05"],[2.64,112.2,"h06"],[2.8,119.3,"h07"],[2.96,117.9,"h08"],[2.72,112.1,"h09"],[3.2,121.3,"h10"],[27.92,149.4,"rainbow"],[2.88,120.8,"h01"],[2.72,113.3,"h02"],[2.56,114.4,"h03"],[2.72,112.6,"h04"],[2.72,115.4,"h05"],[2.64,113.2,"h06"],[2.8,115.6,"h07"],[3.12,123.7,"h08"],[2.64,110.7,"h09"],[3.2,119.1,"h10"],[27.44,149.2,"rainbow"],[2.8,119.8,"h01"],[2.72,107.9,"h02"],[2.56,115.1,"h03"],[2.72,114,"h04"],[2.72,113.4,"h05"],[2.64,112.4,"h06"],[2.88,119.7,"h07"],[3.04,124.8,"h08"],[2.72,115.5,"h09"],[3.12,127,"h10"],[29.04,149.1,"rainbow"],[2.8,119.9,"h01"],[2.72,115.7,"h02"],[2.56,113.8,"h03"],[2.72,115.2,"h04"],[2.72,113.5,"h05"],[2.56,112.6,"h06"],[2.72,113.8,"h07"],[3.04,122.4,"h08"],[2.72,115.3,"h09"],[3.2,120.9,"h10"],[28.72,149.8,"rainbow"]],gradiumWebgpu:[[2.88,28.9,"h01"],[2.72,30.8,"h02"],[2.56,32.4,"h03"],[2.72,30.4,"h04"],[2.72,30.9,"h05"],[2.64,29.9,"h06"],[2.72,30.9,"h07"],[3.12,30.6,"h08"],[2.8,31.7,"h09"],[3.2,32.8,"h10"],[30.08,32.1,"rainbow"],[2.88,29,"h01"],[2.72,30.8,"h02"],[2.48,31.3,"h03"],[2.72,30.9,"h04"],[2.72,30.6,"h05"],[2.64,29.9,"h06"],[2.72,30.9,"h07"],[3.04,29.6,"h08"],[2.72,31.1,"h09"],[3.28,30.5,"h10"],[29.76,31.8,"rainbow"],[2.8,28.2,"h01"],[2.72,30.8,"h02"],[2.64,30,"h03"],[2.8,31.4,"h04"],[2.72,30.6,"h05"],[2.64,29.6,"h06"],[2.8,31.8,"h07"],[3.12,30.4,"h08"],[2.72,31.1,"h09"],[3.2,32.6,"h10"],[30.48,31.9,"rainbow"],[2.8,28.2,"h01"],[2.72,30.8,"h02"],[2.56,32,"h03"],[2.8,31.4,"h04"],[2.72,30.7,"h05"],[2.64,29.5,"h06"],[2.8,31.4,"h07"],[3.12,30.4,"h08"],[2.64,30.2,"h09"],[3.2,32.4,"h10"],[30.56,31.6,"rainbow"],[2.8,28.3,"h01"],[2.72,30.9,"h02"],[2.56,32.1,"h03"],[2.72,30.7,"h04"],[2.72,30.5,"h05"],[2.56,31.9,"h06"],[2.88,32.8,"h07"],[3.12,30.3,"h08"],[2.72,30.9,"h09"],[3.2,32.7,"h10"],[29.68,32,"rainbow"],[2.8,28.2,"h01"],[2.72,30.9,"h02"],[2.56,32.2,"h03"],[2.8,31.8,"h04"],[2.72,30.7,"h05"],[2.64,29.6,"h06"],[2.72,30.8,"h07"],[3.04,29.5,"h08"],[2.72,30.8,"h09"],[3.12,32,"h10"],[29.36,32,"rainbow"],[2.88,29,"h01"],[2.72,31.1,"h02"],[2.64,29.9,"h03"],[2.72,30.5,"h04"],[2.72,30.7,"h05"],[2.64,29.8,"h06"],[2.72,30.6,"h07"],[3.04,29.4,"h08"],[2.64,29.8,"h09"],[3.2,32.7,"h10"],[30,31.9,"rainbow"],[2.88,28.8,"h01"],[2.72,31,"h02"],[2.56,32.5,"h03"],[2.8,31.4,"h04"],[2.8,31.7,"h05"],[2.64,29.7,"h06"],[2.8,31.5,"h07"],[3.04,29.7,"h08"],[2.64,30.3,"h09"],[3.2,32.8,"h10"],[29.92,32,"rainbow"],[2.88,29.1,"h01"],[2.72,31.1,"h02"],[2.56,32.5,"h03"],[2.72,30.9,"h04"],[2.72,30.5,"h05"],[2.64,29.9,"h06"],[2.88,32.6,"h07"],[3.12,30.5,"h08"],[2.64,29.8,"h09"],[3.28,30.8,"h10"],[30,32.1,"rainbow"],[2.8,28.1,"h01"],[2.72,30.7,"h02"],[2.56,32.4,"h03"],[2.72,30.4,"h04"],[2.72,30.5,"h05"],[2.64,29.8,"h06"],[2.8,31.6,"h07"],[3.04,29.5,"h08"],[2.72,30.8,"h09"],[3.12,31.7,"h10"],[29.68,31.8,"rainbow"]],gradiumCpu:[[2.88,13.9,"h01"],[2.72,14.8,"h02"],[2.56,14.8,"h03"],[2.72,14.6,"h04"],[2.72,14.6,"h05"],[2.64,14.6,"h06"],[2.72,14.8,"h07"],[3.12,14.5,"h08"],[2.72,14.7,"h09"],[3.2,14.7,"h10"],[29.68,14.5,"rainbow"],[2.88,14.7,"h01"],[2.72,15.1,"h02"],[2.48,15,"h03"],[2.72,14.9,"h04"],[2.72,15,"h05"],[2.64,15,"h06"],[2.72,15.1,"h07"],[3.04,14.9,"h08"],[2.72,14.9,"h09"],[3.28,15,"h10"],[29.36,14.7,"rainbow"],[2.8,14.9,"h01"],[2.72,15,"h02"],[2.56,15.1,"h03"],[2.8,15.1,"h04"],[2.72,14.8,"h05"],[2.64,15.1,"h06"],[2.8,15.1,"h07"],[3.12,14.9,"h08"],[2.72,15.2,"h09"],[3.2,15.2,"h10"],[30.4,14.7,"rainbow"],[2.88,14.9,"h01"],[2.72,15,"h02"],[2.48,15,"h03"],[2.8,14.8,"h04"],[2.72,14.8,"h05"],[2.64,15.1,"h06"],[2.8,15.1,"h07"],[3.12,14.9,"h08"],[2.64,15.2,"h09"],[3.2,15.2,"h10"],[29.76,14.7,"rainbow"],[2.8,14.7,"h01"],[2.72,15,"h02"],[2.56,15,"h03"],[2.72,14.9,"h04"],[2.72,15,"h05"],[2.56,14.9,"h06"],[2.88,15.1,"h07"],[3.12,14.8,"h08"],[2.72,15,"h09"],[3.2,15.2,"h10"],[29.44,14.7,"rainbow"],[2.8,14.7,"h01"],[2.72,15.1,"h02"],[2.56,15.1,"h03"],[2.8,15.2,"h04"],[2.64,15,"h05"],[2.64,15,"h06"],[2.72,15.1,"h07"],[3.04,14.9,"h08"],[2.72,15.2,"h09"],[3.12,15.2,"h10"],[29.68,14.7,"rainbow"],[2.8,14.3,"h01"],[2.72,15.2,"h02"],[2.64,15,"h03"],[2.72,15.2,"h04"],[2.72,14.8,"h05"],[2.64,14.9,"h06"],[2.72,14.1,"h07"],[3.04,13.9,"h08"],[2.64,15.2,"h09"],[3.2,14.5,"h10"],[30,14.6,"rainbow"],[2.88,14.8,"h01"],[2.72,14.9,"h02"],[2.56,15,"h03"],[2.72,14.9,"h04"],[2.8,15.1,"h05"],[2.64,14.4,"h06"],[2.8,15,"h07"],[3.04,14.7,"h08"],[2.64,14.7,"h09"],[3.2,15,"h10"],[29.6,14.2,"rainbow"],[2.88,14.4,"h01"],[2.72,14.4,"h02"],[2.56,14.3,"h03"],[2.72,14.8,"h04"],[2.72,14.3,"h05"],[2.64,14.3,"h06"],[2.88,14.3,"h07"],[3.2,14.4,"h08"],[2.64,14.7,"h09"],[3.28,14.5,"h10"],[29.68,14,"rainbow"],[2.8,13.6,"h01"],[2.72,13.7,"h02"],[2.56,13.4,"h03"],[2.72,13.2,"h04"],[2.72,13.1,"h05"],[2.64,14.2,"h06"],[2.72,13.9,"h07"],[3.04,13.3,"h08"],[2.72,13.8,"h09"],[3.04,14.4,"h10"],[29.68,14,"rainbow"]],kokoro:[[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,12.7,"h03"],[3,14.4,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.4,"h07"],[3.15,14.1,"h08"],[2.6,14.4,"h09"],[3.25,14.4,"h10"],[31.8,15.7,"rainbow"],[2.83,13.2,"h01"],[2.88,13.7,"h02"],[2.52,13,"h03"],[3,14.6,"h04"],[2.88,14.4,"h05"],[2.65,13.7,"h06"],[2.83,12.9,"h07"],[3.15,14.5,"h08"],[2.6,14,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,12.9,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.9,"h07"],[3.15,14.5,"h08"],[2.6,14,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,13.2,"h03"],[3,14.5,"h04"],[2.88,14.4,"h05"],[2.65,13.9,"h06"],[2.83,12.9,"h07"],[3.15,14.4,"h08"],[2.6,14.2,"h09"],[3.25,14.5,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,13.7,"h02"],[2.52,12.9,"h03"],[3,14.5,"h04"],[2.88,14.5,"h05"],[2.65,14,"h06"],[2.83,13,"h07"],[3.15,14.6,"h08"],[2.6,13.8,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,12.9,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.9,"h07"],[3.15,14.6,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.7,"rainbow"],[2.83,13.4,"h01"],[2.88,14.3,"h02"],[2.52,13,"h03"],[3,14.7,"h04"],[2.88,14.2,"h05"],[2.65,14.1,"h06"],[2.83,12.8,"h07"],[3.15,14.6,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14,"h02"],[2.52,13,"h03"],[3,14.7,"h04"],[2.88,14.1,"h05"],[2.65,13.9,"h06"],[2.83,12.9,"h07"],[3.15,14.4,"h08"],[2.6,14.1,"h09"],[3.25,14.4,"h10"],[31.8,15.8,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,12.9,"h03"],[3,14.4,"h04"],[2.88,14.2,"h05"],[2.65,13.8,"h06"],[2.83,12.7,"h07"],[3.15,14.3,"h08"],[2.6,13.8,"h09"],[3.25,14.5,"h10"],[31.8,15.6,"rainbow"],[2.83,13.3,"h01"],[2.88,14.1,"h02"],[2.52,13.1,"h03"],[3,14.4,"h04"],[2.88,14.3,"h05"],[2.65,13.8,"h06"],[2.83,12.5,"h07"],[3.15,14.4,"h08"],[2.6,13.8,"h09"],[3.25,14.2,"h10"],[31.8,15.5,"rainbow"]]},pc:{ours:[[2.88,80.3,"h01"],[2.72,76.2,"h02"],[2.56,73.4,"h03"],[2.72,78.7,"h04"],[2.72,75.5,"h05"],[2.64,75.6,"h06"],[2.72,75.6,"h07"],[3.04,82.1,"h08"],[2.72,67.1,"h09"],[3.2,80,"h10"],[28.08,101.4,"rainbow"],[2.8,54.4,"h01"],[2.64,50.4,"h02"],[2.56,52.4,"h03"],[2.72,52.8,"h04"],[2.72,72.6,"h05"],[2.64,68.3,"h06"],[2.8,80,"h07"],[3.12,87.2,"h08"],[2.64,69,"h09"],[3.28,78.2,"h10"],[28.08,101.4,"rainbow"],[2.88,71.7,"h01"],[2.72,68.3,"h02"],[2.56,67.5,"h03"],[2.72,66.4,"h04"],[2.72,78.1,"h05"],[2.64,74.6,"h06"],[2.8,79.8,"h07"],[3.04,76.2,"h08"],[2.64,70.1,"h09"],[3.2,80.7,"h10"],[27.92,100.6,"rainbow"],[2.8,58.4,"h01"],[2.72,54,"h02"],[2.56,67.6,"h03"],[2.72,74.5,"h04"],[2.72,71.5,"h05"],[2.64,74.6,"h06"],[2.72,72,"h07"],[3.2,83.6,"h08"],[2.64,76.3,"h09"],[3.2,82,"h10"],[28.16,101.5,"rainbow"],[2.88,72.6,"h01"],[2.72,75.3,"h02"],[2.64,66.1,"h03"],[2.72,76.5,"h04"],[2.72,68.6,"h05"],[2.64,67.8,"h06"],[2.72,64.7,"h07"],[3.12,77,"h08"],[2.64,72.7,"h09"],[3.28,73.6,"h10"],[28.32,100.6,"rainbow"],[2.8,69,"h01"],[2.72,66.5,"h02"],[2.56,74.3,"h03"],[2.72,68.8,"h04"],[2.64,66.2,"h05"],[2.64,74.1,"h06"],[2.72,75.9,"h07"],[3.12,86.2,"h08"],[2.64,67.8,"h09"],[3.2,72.5,"h10"],[27.68,99.6,"rainbow"],[2.88,56.4,"h01"],[2.72,51.5,"h02"],[2.56,54,"h03"],[2.72,53.2,"h04"],[2.72,70.8,"h05"],[2.64,71.4,"h06"],[2.8,77.9,"h07"],[2.96,80.1,"h08"],[2.72,65.4,"h09"],[3.2,79.1,"h10"],[27.92,100.1,"rainbow"],[2.88,58.3,"h01"],[2.72,59.8,"h02"],[2.56,61.2,"h03"],[2.72,65.5,"h04"],[2.72,61.2,"h05"],[2.64,68.7,"h06"],[2.8,69.2,"h07"],[3.12,82.9,"h08"],[2.64,73.4,"h09"],[3.2,80,"h10"],[27.44,101.1,"rainbow"],[2.8,66.6,"h01"],[2.72,65.9,"h02"],[2.56,67.5,"h03"],[2.72,66.5,"h04"],[2.72,67.4,"h05"],[2.64,63.1,"h06"],[2.88,70.8,"h07"],[3.04,75.7,"h08"],[2.72,69.8,"h09"],[3.12,77.6,"h10"],[28,100.3,"rainbow"],[2.8,70.8,"h01"],[2.72,70.4,"h02"],[2.56,68.1,"h03"],[2.72,69.7,"h04"],[2.72,76.7,"h05"],[2.56,73.2,"h06"],[2.72,71.6,"h07"],[3.04,82.1,"h08"],[2.72,68.8,"h09"],[3.2,79.3,"h10"],[28.72,100.9,"rainbow"]],gradiumWebgpu:[[2.88,17.7,"h01"],[2.72,22.7,"h02"],[2.56,23.6,"h03"],[2.72,22.9,"h04"],[2.72,22.9,"h05"],[2.64,22.4,"h06"],[2.72,21.4,"h07"],[3.12,21.8,"h08"],[2.8,23.6,"h09"],[3.2,24.7,"h10"],[30.08,25.1,"rainbow"],[2.88,16.4,"h01"],[2.72,20.1,"h02"],[2.48,20.7,"h03"],[2.72,19.9,"h04"],[2.72,20.2,"h05"],[2.64,18.9,"h06"],[2.72,19.8,"h07"],[3.04,19.1,"h08"],[2.72,17.3,"h09"],[3.28,20.6,"h10"],[29.76,21.3,"rainbow"],[2.8,16.3,"h01"],[2.72,20.1,"h02"],[2.64,19.9,"h03"],[2.8,21,"h04"],[2.72,20.2,"h05"],[2.64,19.1,"h06"],[2.8,19.6,"h07"],[3.12,20.8,"h08"],[2.72,20,"h09"],[3.2,21.7,"h10"],[30.48,22.5,"rainbow"],[2.8,17,"h01"],[2.72,22.7,"h02"],[2.56,23.4,"h03"],[2.8,23.6,"h04"],[2.72,22.7,"h05"],[2.64,22,"h06"],[2.8,22.8,"h07"],[3.12,23.6,"h08"],[2.64,22.2,"h09"],[3.2,24.6,"h10"],[30.56,24.9,"rainbow"],[2.8,14.6,"h01"],[2.72,20.1,"h02"],[2.56,19.9,"h03"],[2.72,19.8,"h04"],[2.72,20.3,"h05"],[2.56,20.9,"h06"],[2.88,21,"h07"],[3.12,21.2,"h08"],[2.72,20.4,"h09"],[3.2,21.7,"h10"],[29.68,22.5,"rainbow"],[2.8,16.1,"h01"],[2.72,19.8,"h02"],[2.56,21.2,"h03"],[2.8,19.9,"h04"],[2.72,21.4,"h05"],[2.64,20.2,"h06"],[2.72,20.3,"h07"],[3.04,21,"h08"],[2.72,20.7,"h09"],[3.12,21.3,"h10"],[29.36,21.8,"rainbow"],[2.88,16.8,"h01"],[2.72,20.4,"h02"],[2.64,18.3,"h03"],[2.72,21.6,"h04"],[2.72,20.4,"h05"],[2.64,20,"h06"],[2.72,20.4,"h07"],[3.04,20.9,"h08"],[2.64,19.1,"h09"],[3.2,22,"h10"],[30,22.5,"rainbow"],[2.88,16.8,"h01"],[2.72,21.4,"h02"],[2.56,21.8,"h03"],[2.8,20.5,"h04"],[2.8,21.3,"h05"],[2.64,19.9,"h06"],[2.8,20.5,"h07"],[3.04,22.4,"h08"],[2.64,20.5,"h09"],[3.2,23.1,"h10"],[29.92,24.3,"rainbow"],[2.88,15.1,"h01"],[2.72,20.8,"h02"],[2.56,21.1,"h03"],[2.72,20.1,"h04"],[2.72,18.5,"h05"],[2.64,20.2,"h06"],[2.88,21.5,"h07"],[3.12,21.4,"h08"],[2.64,20.1,"h09"],[3.28,20.9,"h10"],[30,22.9,"rainbow"],[2.8,15.4,"h01"],[2.72,20.1,"h02"],[2.56,21.3,"h03"],[2.72,20.1,"h04"],[2.72,20.6,"h05"],[2.64,20,"h06"],[2.8,20.7,"h07"],[3.04,19.1,"h08"],[2.72,18.6,"h09"],[3.12,21.8,"h10"],[29.68,22.4,"rainbow"]],gradiumCpu:[[2.8,5.9,"h01"],[2.72,6.1,"h02"],[2.56,5.9,"h03"],[2.72,6,"h04"],[2.72,6.2,"h05"],[2.64,6.1,"h06"],[2.72,6.1,"h07"],[3.12,6.1,"h08"],[2.8,6.2,"h09"],[3.2,6.2,"h10"],[29.6,5.9,"rainbow"],[2.88,5.8,"h01"],[2.72,5.8,"h02"],[2.48,5.9,"h03"],[2.72,5.8,"h04"],[2.72,5.9,"h05"],[2.64,5.9,"h06"],[2.72,6,"h07"],[3.04,5.8,"h08"],[2.72,5.9,"h09"],[3.28,6.2,"h10"],[29.36,6,"rainbow"],[2.88,5.5,"h01"],[2.72,5.9,"h02"],[2.64,5.9,"h03"],[2.8,5.9,"h04"],[2.72,5.9,"h05"],[2.64,5.5,"h06"],[2.8,5.9,"h07"],[3.12,5.4,"h08"],[2.72,5.8,"h09"],[3.2,5.4,"h10"],[30.24,5.9,"rainbow"],[2.88,5.6,"h01"],[2.72,5.3,"h02"],[2.56,5.9,"h03"],[2.8,5.6,"h04"],[2.72,6.2,"h05"],[2.64,6.2,"h06"],[2.8,5.9,"h07"],[3.12,6.1,"h08"],[2.64,6,"h09"],[3.2,6.2,"h10"],[29.76,6,"rainbow"],[2.8,5.6,"h01"],[2.72,5.8,"h02"],[2.56,5.7,"h03"],[2.72,5.6,"h04"],[2.72,5.9,"h05"],[2.56,5.6,"h06"],[2.88,5.8,"h07"],[3.12,5.8,"h08"],[2.72,6,"h09"],[3.2,5.9,"h10"],[29.36,5.7,"rainbow"],[2.8,5,"h01"],[2.72,5.5,"h02"],[2.56,5.6,"h03"],[2.72,5.4,"h04"],[2.64,5.5,"h05"],[2.64,5.7,"h06"],[2.72,5.6,"h07"],[3.04,5.7,"h08"],[2.72,5.8,"h09"],[3.12,5.8,"h10"],[29.68,5.7,"rainbow"],[2.8,5.4,"h01"],[2.72,6,"h02"],[2.64,5.5,"h03"],[2.72,5.6,"h04"],[2.72,5.4,"h05"],[2.64,5.6,"h06"],[2.72,5.6,"h07"],[3.04,5.1,"h08"],[2.64,5.2,"h09"],[3.2,5.6,"h10"],[30,5.6,"rainbow"],[2.88,6,"h01"],[2.72,6.1,"h02"],[2.56,6.1,"h03"],[2.72,6,"h04"],[2.8,6.2,"h05"],[2.64,6.2,"h06"],[2.8,6.2,"h07"],[3.04,6.1,"h08"],[2.64,6.2,"h09"],[3.2,6.1,"h10"],[29.68,6,"rainbow"],[2.88,5.5,"h01"],[2.72,5.7,"h02"],[2.56,5.6,"h03"],[2.72,5.6,"h04"],[2.72,5.7,"h05"],[2.64,5.9,"h06"],[2.88,5.7,"h07"],[3.12,5.7,"h08"],[2.64,5.8,"h09"],[3.28,5.7,"h10"],[29.68,5.8,"rainbow"],[2.8,5.4,"h01"],[2.72,5.4,"h02"],[2.56,5.6,"h03"],[2.72,5.6,"h04"],[2.72,5.7,"h05"],[2.64,5.6,"h06"],[2.8,5.6,"h07"],[3.04,5.5,"h08"],[2.72,6,"h09"],[3.12,5.2,"h10"],[29.68,5.3,"rainbow"]],kokoro:[[2.83,9.2,"h01"],[2.88,8.6,"h02"],[2.52,7.4,"h03"],[3,8.1,"h04"],[2.88,11.9,"h05"],[2.65,8.7,"h06"],[2.83,8.7,"h07"],[3.15,10,"h08"],[2.6,9.3,"h09"],[3.25,10,"h10"],[31.8,11.7,"rainbow"],[2.83,9.6,"h01"],[2.88,10.6,"h02"],[2.52,10.3,"h03"],[3,11.3,"h04"],[2.88,10.4,"h05"],[2.65,11.2,"h06"],[2.83,11.6,"h07"],[3.15,11.6,"h08"],[2.6,11.3,"h09"],[3.25,11.3,"h10"],[31.8,13,"rainbow"],[2.83,9.3,"h01"],[2.88,10.8,"h02"],[2.52,9.4,"h03"],[3,11.6,"h04"],[2.88,11.7,"h05"],[2.65,10.6,"h06"],[2.83,10.2,"h07"],[3.15,11.4,"h08"],[2.6,11.2,"h09"],[3.25,11.7,"h10"],[31.8,12.8,"rainbow"],[2.83,10,"h01"],[2.88,11.1,"h02"],[2.52,9.9,"h03"],[3,11,"h04"],[2.88,10.8,"h05"],[2.65,11.1,"h06"],[2.83,11.1,"h07"],[3.15,11.3,"h08"],[2.6,11.3,"h09"],[3.25,11.8,"h10"],[31.8,13.1,"rainbow"],[2.83,10.2,"h01"],[2.88,11.1,"h02"],[2.52,10,"h03"],[3,11.4,"h04"],[2.88,11.4,"h05"],[2.65,11,"h06"],[2.83,10.1,"h07"],[3.15,8.2,"h08"],[2.6,11,"h09"],[3.25,11.6,"h10"],[31.8,13,"rainbow"],[2.83,8.4,"h01"],[2.88,11,"h02"],[2.52,10.9,"h03"],[3,11.6,"h04"],[2.88,10.8,"h05"],[2.65,11,"h06"],[2.83,11,"h07"],[3.15,10.1,"h08"],[2.6,11.5,"h09"],[3.25,11.8,"h10"],[31.8,13.1,"rainbow"],[2.83,8.1,"h01"],[2.88,9.9,"h02"],[2.52,9.4,"h03"],[3,11.2,"h04"],[2.88,10,"h05"],[2.65,10.5,"h06"],[2.83,11,"h07"],[3.15,11.2,"h08"],[2.6,11.2,"h09"],[3.25,11.2,"h10"],[31.8,12.5,"rainbow"],[2.83,9.8,"h01"],[2.88,10.7,"h02"],[2.52,9.3,"h03"],[3,11.2,"h04"],[2.88,11.2,"h05"],[2.65,11,"h06"],[2.83,9.9,"h07"],[3.15,9.8,"h08"],[2.6,11,"h09"],[3.25,11.5,"h10"],[31.8,13,"rainbow"],[2.83,10.2,"h01"],[2.88,10.7,"h02"],[2.52,9.8,"h03"],[3,11.3,"h04"],[2.88,11.4,"h05"],[2.65,11.2,"h06"],[2.83,9.8,"h07"],[3.15,8.3,"h08"],[2.6,11,"h09"],[3.25,11.5,"h10"],[31.8,12.6,"rainbow"],[2.83,10.3,"h01"],[2.88,11,"h02"],[2.52,10,"h03"],[3,11,"h04"],[2.88,11.5,"h05"],[2.65,11,"h06"],[2.83,9.7,"h07"],[3.15,11.7,"h08"],[2.6,11.3,"h09"],[3.25,11.7,"h10"],[31.8,12.8,"rainbow"]]},iphone:{ours:[[2.88,30.7,"h01"],[2.72,28.7,"h02"],[2.56,28.5,"h03"],[2.72,28.3,"h04"],[2.72,28.3,"h05"],[2.64,27.7,"h06"],[2.72,28.4,"h07"],[3.04,31,"h08"],[2.72,28.3,"h09"],[3.2,30.5,"h10"],[27.92,37.6,"rainbow"],[2.8,29.8,"h01"],[2.64,28.5,"h02"],[2.56,28.5,"h03"],[2.72,28.5,"h04"],[2.72,28.6,"h05"],[2.64,27.7,"h06"],[2.8,29.1,"h07"],[3.12,31.6,"h08"],[2.64,28,"h09"],[3.28,29.6,"h10"],[28.16,37,"rainbow"],[2.88,29.9,"h01"],[2.72,27.5,"h02"],[2.56,26.7,"h03"],[2.72,26.5,"h04"],[2.72,25.3,"h05"],[2.64,26,"h06"],[2.8,25.8,"h07"],[3.04,27.4,"h08"],[2.64,25.2,"h09"],[3.2,28.3,"h10"],[27.68,32,"rainbow"],[2.8,28.7,"h01"],[2.72,25.2,"h02"],[2.56,23.4,"h03"],[2.72,21.7,"h04"],[2.72,21,"h05"],[2.64,20.3,"h06"],[2.72,20.9,"h07"],[3.2,22.5,"h08"],[2.64,19.5,"h09"],[3.2,22.2,"h10"],[28.16,27.3,"rainbow"],[2.88,29.9,"h01"],[2.72,28.2,"h02"],[2.64,27.7,"h03"],[2.72,27.2,"h04"],[2.72,26.1,"h05"],[2.64,25.5,"h06"],[2.72,25.9,"h07"],[3.12,28.7,"h08"],[2.64,25.4,"h09"],[3.28,26.3,"h10"],[28.32,31.7,"rainbow"],[2.8,22.6,"h01"],[2.72,21.5,"h02"],[2.56,21.5,"h03"],[2.72,20.9,"h04"],[2.64,20.9,"h05"],[2.64,20.4,"h06"],[2.72,20.9,"h07"],[3.12,24.4,"h08"],[2.64,21.8,"h09"],[3.2,23.3,"h10"],[27.68,19.9,"rainbow"],[2.88,29.5,"h01"],[2.72,26.7,"h02"],[2.56,25.1,"h03"],[2.72,22.9,"h04"],[2.72,21.5,"h05"],[2.64,21.2,"h06"],[2.8,21.4,"h07"],[2.96,22.3,"h08"],[2.72,20.8,"h09"],[3.2,22.1,"h10"],[27.92,26.1,"rainbow"],[2.88,29.8,"h01"],[2.72,26.6,"h02"],[2.56,25.9,"h03"],[2.72,23.7,"h04"],[2.72,22.5,"h05"],[2.64,20.8,"h06"],[2.8,21.6,"h07"],[3.12,23.3,"h08"],[2.64,20.2,"h09"],[3.2,22.1,"h10"],[27.36,26.1,"rainbow"],[2.8,21.7,"h01"],[2.72,20.7,"h02"],[2.56,20.5,"h03"],[2.72,19.9,"h04"],[2.72,20,"h05"],[2.64,18.8,"h06"],[2.88,20.1,"h07"],[3.04,20.4,"h08"],[2.72,18.5,"h09"],[3.12,20.3,"h10"],[28,23.4,"rainbow"],[2.8,29.1,"h01"],[2.72,28,"h02"],[2.56,28.2,"h03"],[2.72,28.3,"h04"],[2.72,28.4,"h05"],[2.56,28.5,"h06"],[2.72,28.4,"h07"],[3.04,30.9,"h08"],[2.72,28.4,"h09"],[3.2,30.2,"h10"],[27.52,37,"rainbow"]],gradiumWebgpu:[[2.88,19.1,"h01"],[2.72,16.5,"h02"],[2.56,18.3,"h03"],[2.72,17.9,"h04"],[2.72,19.1,"h05"],[2.64,17,"h06"],[2.72,15.7,"h07"],[3.12,17.6,"h08"],[2.8,18.8,"h09"],[3.2,15.8,"h10"],[30.08,17.8,"rainbow"],[2.88,15.9,"h01"],[2.72,14.2,"h02"],[2.48,13.9,"h03"],[2.72,14.7,"h04"],[2.72,14.7,"h05"],[2.64,13.9,"h06"],[2.72,13.4,"h07"],[3.04,13.8,"h08"],[2.72,13.6,"h09"],[3.28,13.9,"h10"],[29.76,13.7,"rainbow"],[2.8,12.1,"h01"],[2.72,16.7,"h02"],[2.64,12.1,"h03"],[2.8,14.1,"h04"],[2.72,12.8,"h05"],[2.64,13.1,"h06"],[2.8,13.3,"h07"],[3.12,12.8,"h08"],[2.72,14.9,"h09"],[3.2,13,"h10"],[30.48,11.8,"rainbow"],[2.8,12.2,"h01"],[2.72,16.1,"h02"],[2.56,15.8,"h03"],[2.8,13.4,"h04"],[2.72,13.4,"h05"],[2.64,12.5,"h06"],[2.8,13.1,"h07"],[3.12,12.9,"h08"],[2.64,11.4,"h09"],[3.2,16.5,"h10"],[30.56,12.8,"rainbow"],[2.8,14.5,"h01"],[2.72,13.9,"h02"],[2.56,14.2,"h03"],[2.72,12,"h04"],[2.72,11.7,"h05"],[2.56,12.5,"h06"],[2.88,11.3,"h07"],[3.12,9.3,"h08"],[2.72,10.3,"h09"],[3.2,9,"h10"],[29.68,10,"rainbow"],[2.8,7.3,"h01"],[2.72,5.7,"h02"],[2.56,7.5,"h03"],[2.8,7.3,"h04"],[2.72,7.5,"h05"],[2.64,6.9,"h06"],[2.72,6.9,"h07"],[3.04,7.2,"h08"],[2.72,7.7,"h09"],[3.12,8.2,"h10"],[29.36,7.5,"rainbow"],[2.88,13.3,"h01"],[2.72,10.3,"h02"],[2.64,11.4,"h03"],[2.72,10.6,"h04"],[2.72,10,"h05"],[2.64,9.4,"h06"],[2.72,8.6,"h07"],[3.04,7.6,"h08"],[2.64,6.7,"h09"],[3.2,7.9,"h10"],[30,8.2,"rainbow"],[2.88,12.4,"h01"],[2.72,14.1,"h02"],[2.56,10,"h03"],[2.8,9.4,"h04"],[2.8,6.4,"h05"],[2.64,5.9,"h06"],[2.8,5.2,"h07"],[3.04,5.9,"h08"],[2.64,5.4,"h09"],[3.2,7.9,"h10"],[29.92,8.4,"rainbow"],[2.88,15.2,"h01"],[2.72,10.5,"h02"],[2.56,10,"h03"],[2.72,8.7,"h04"],[2.72,6.8,"h05"],[2.64,5.4,"h06"],[2.88,5.2,"h07"],[3.12,5.9,"h08"],[2.64,5.2,"h09"],[3.28,7.3,"h10"],[30,8.4,"rainbow"],[2.8,18.2,"h01"],[2.72,18.5,"h02"],[2.56,15.8,"h03"],[2.72,16.5,"h04"],[2.72,16.7,"h05"],[2.64,15.6,"h06"],[2.8,19.4,"h07"],[3.04,16.8,"h08"],[2.72,16,"h09"],[3.12,16.1,"h10"],[29.68,17.2,"rainbow"]],gradiumCpu:[[2.88,7.1,"h01"],[2.72,7.4,"h02"],[2.56,7.3,"h03"],[2.72,7.4,"h04"],[2.72,7.5,"h05"],[2.64,7.5,"h06"],[2.72,7.5,"h07"],[3.12,7.3,"h08"],[2.72,7.4,"h09"],[3.12,7.4,"h10"],[29.44,7.2,"rainbow"],[2.88,7,"h01"],[2.72,7.3,"h02"],[2.48,7.2,"h03"],[2.72,7.2,"h04"],[2.72,7.2,"h05"],[2.64,7.3,"h06"],[2.72,7.2,"h07"],[3.04,7.2,"h08"],[2.72,7.3,"h09"],[3.28,7.2,"h10"],[29.76,7.1,"rainbow"],[2.8,5.5,"h01"],[2.72,5.7,"h02"],[2.64,5.8,"h03"],[2.8,5.8,"h04"],[2.72,5.8,"h05"],[2.64,5.8,"h06"],[2.8,5.9,"h07"],[3.12,5.8,"h08"],[2.72,5.9,"h09"],[3.2,5.9,"h10"],[29.52,5.8,"rainbow"],[2.8,3.7,"h01"],[2.72,3.9,"h02"],[2.56,3.9,"h03"],[2.8,4,"h04"],[2.72,4,"h05"],[2.64,4.1,"h06"],[2.8,4.1,"h07"],[3.12,4,"h08"],[2.64,4.1,"h09"],[3.2,4.1,"h10"],[29.76,4.1,"rainbow"],[2.8,4,"h01"],[2.72,4.2,"h02"],[2.56,4.1,"h03"],[2.72,4.2,"h04"],[2.72,4.2,"h05"],[2.56,4.2,"h06"],[2.88,4.3,"h07"],[3.12,4.2,"h08"],[2.72,4.2,"h09"],[3.2,4.2,"h10"],[29.36,4.2,"rainbow"],[2.8,4.9,"h01"],[2.72,5.2,"h02"],[2.56,5.2,"h03"],[2.8,5.5,"h04"],[2.72,5.7,"h05"],[2.64,5.9,"h06"],[2.72,6.3,"h07"],[3.04,6.1,"h08"],[2.72,6.2,"h09"],[3.12,6.2,"h10"],[29.68,6,"rainbow"],[2.88,4.9,"h01"],[2.72,5,"h02"],[2.64,5.2,"h03"],[2.72,5.5,"h04"],[2.72,5.5,"h05"],[2.64,5.6,"h06"],[2.72,5.6,"h07"],[3.04,4.8,"h08"],[2.64,4.4,"h09"],[3.2,5.1,"h10"],[30,5.1,"rainbow"],[2.88,5.9,"h01"],[2.72,4.8,"h02"],[2.56,4.2,"h03"],[2.8,4.3,"h04"],[2.8,4.7,"h05"],[2.64,4.8,"h06"],[2.72,5.3,"h07"],[3.04,5.2,"h08"],[2.64,5.5,"h09"],[3.2,5.5,"h10"],[29.68,5.4,"rainbow"],[2.88,4,"h01"],[2.72,4.4,"h02"],[2.56,4.5,"h03"],[2.72,4.8,"h04"],[2.72,4.8,"h05"],[2.64,4.9,"h06"],[2.88,4.9,"h07"],[3.12,4.8,"h08"],[2.64,4.9,"h09"],[3.28,4.9,"h10"],[29.44,4.9,"rainbow"],[2.8,7.1,"h01"],[2.72,7.3,"h02"],[2.56,7.3,"h03"],[2.72,7.3,"h04"],[2.72,7.3,"h05"],[2.64,7.3,"h06"],[2.8,7.2,"h07"],[3.04,7.1,"h08"],[2.72,7.4,"h09"],[3.12,7.4,"h10"],[29.84,7.1,"rainbow"]]}},timing:{words:712,meanMs:66.6783707864741,medianMs:64.99999999999928,p90Ms:124.99999999862261,within100ms:.7626404494382022,meanSignedMs:29.55758426963997,medianSignedMs:39.999999999999815,errorsMs:[-50,80,10,70,30,170,80,60,-60,40,105,40,105,85,30,-0,-30,0,-145,120,30,80,80,65,105,-60,20,-25,40,-65,5,85,85,-75,-70,-50,-15,-160,-10,80,-50,-50,80,10,85,70,40,-100,-50,80,-60,70,80,20,90,45,-30,80,80,40,-30,-110,-45,45,50,35,-165,70,105,50,-40,-50,60,-210,-140,75,20,35,-20,75,-20,60,100,-10,30,40,85,5,40,-25,-50,150,105,-165,50,-20,110,20,75,40,80,85,-120,60,-20,-100,-15,-40,80,20,10,55,105,0,65,90,-80,-85,10,65,80,-100,-20,-0,55,80,-60,80,25,-95,105,20,-155,-165,75,45,-40,55,75,-40,65,-60,5,-60,40,-75,20,50,85,80,20,-70,20,90,100,-0,40,115,45,-115,5,20,40,30,55,80,150,120,135,140,-15,100,-90,145,65,35,115,115,10,100,20,25,60,110,55,30,20,75,110,25,120,85,10,-10,-30,95,-15,105,40,105,115,90,110,-40,-10,-105,55,-90,10,105,70,-70,-50,-50,45,-135,-140,65,-75,-10,100,-5,75,75,20,-80,0,100,-15,60,50,-20,120,80,10,120,115,55,-25,-60,35,85,-30,35,-130,70,110,-35,-50,-40,85,-130,-155,30,65,55,-25,115,0,85,120,-20,75,70,140,5,20,110,60,105,110,-25,110,-15,85,30,70,65,105,125,-25,100,-65,-80,45,-55,80,20,20,55,125,-15,100,95,-20,-160,20,150,70,-115,-40,10,60,85,-15,65,95,-75,140,30,-55,-20,95,85,-20,70,70,0,140,-75,45,5,10,-5,30,35,80,135,45,-60,10,95,130,35,80,125,50,-75,65,0,60,75,45,130,100,140,120,105,-5,120,-85,145,110,90,145,125,-30,100,25,30,80,125,60,55,20,90,130,45,120,110,10,0,-30,20,-85,105,40,85,115,95,115,-50,10,-65,80,-100,-5,80,65,-85,-70,-60,45,-120,20,110,-85,-30,100,15,75,95,-35,-70,-30,100,-55,60,75,-10,125,40,-20,100,125,45,-80,-110,-5,45,10,60,-125,50,105,30,-50,-20,110,-160,-155,50,35,80,-20,80,-10,85,100,-15,70,90,125,25,30,20,-20,95,100,-70,90,20,100,20,105,35,140,150,-15,85,-30,-40,35,-35,100,30,-0,35,110,10,85,100,0,-145,30,150,105,-60,-5,0,75,100,-20,85,95,-75,140,20,-240,-30,80,65,-40,95,85,20,75,-60,45,0,10,-25,30,110,60,80,-35,-115,20,80,100,-0,50,140,70,-75,50,20,65,40,15,100,100,85,140,120,5,125,-70,150,125,90,145,120,-10,80,50,55,20,135,100,50,-30,45,115,35,120,85,10,5,-20,10,-115,85,20,70,90,75,140,-50,15,-20,50,-65,25,115,105,-95,-70,-55,10,-155,5,115,-95,-40,80,0,65,55,-35,-120,-10,100,-75,65,60,-0,130,45,-10,95,90,35,-65,-170,10,65,-30,40,-135,60,135,20,-20,-50,30,-235,-150,75,60,65,-5,115,-10,80,100,-30,70,60,140,25,20,-30,-35,140,70,-95,80,-45,100,-0,-90,15,95,140,-75,85,-5,-75,-15,-40,70,10,60,40,85,-20,115,100,-115,-115,20,50,80,-120,0,10,35,80,-40,90,80,-85,90,10,-110,-40,90,60,-30,55,60,-20,80,-125,5,0,30,-20,30,70,80,115,30,-60,20,85,140,5,45,125,55,-110,25,10,-10,40,45,95,150,120,125,120,-15,80,-85,145,105,95,160,120]},quality:{device:"mac",clips:{gradiumCpu:110,gradiumWebgpu:110,kokoro:110,ours:110},wer:{gradiumCpu:[.0016853932584269663,0,.004373566264840013],gradiumWebgpu:[.0022471910112359553,0,.005362002209207331],kokoro:[.011235955056179775,.008196511569374298,.015132599496159143],ours:[0,0,0]},werOursMinusGradiumWebgpu:[-.0022471910112359553,-.00528785865989347,0],speakerSimilarity:{"ours~gradiumWebgpu":[.9883882677916325,.9879526199867027,.9887953618198935],"gradiumWebgpu~gradiumWebgpu":[.9885844175261681,.9881791681758683,.9889836907416882],"ours~ours":[.9882225721773475,.9878083817374826,.9886148152402555],"gradiumCpu~gradiumWebgpu":[.9881263611292598,.9876704000102149,.9885679969658153]},utmos:{gradiumCpu:[4.4043716343966395,4.382951672293923,4.423995101668617],gradiumWebgpu:[4.40301629413258,4.384446497830478,4.419035808714953],kokoro:[4.4293786829168145,4.422673785253005,4.436434451775117],ours:[4.400310355966742,4.382847055413507,4.416018287376923]}}};var je={source:"Harvard sentences, list 1 (IEEE Recommended Practice for Speech Quality Measurements, 1969), and the first paragraph of the Rainbow Passage (Fairbanks, Voice and Articulation Drillbook, 1960).",cold:"h01",phrases:[{id:"h01",text:"The birch canoe slid on the smooth planks."},{id:"h02",text:"Glue the sheet to the dark blue background."},{id:"h03",text:"It's easy to tell the depth of a well."},{id:"h04",text:"These days a chicken leg is a rare dish."},{id:"h05",text:"Rice is often served in round bowls."},{id:"h06",text:"The juice of lemons makes fine punch."},{id:"h07",text:"The box was thrown beside the parked truck."},{id:"h08",text:"The hogs were fed chopped corn and garbage."},{id:"h09",text:"Four hours of steady work faced us."},{id:"h10",text:"A large size in stockings is hard to sell."},{id:"rainbow",text:"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow."}]};var B0=oe.results,ut=oe.phrases,ct=oe.timing,lt=oe.quality,C=[{id:"ours",name:"blixt/phonon"},{id:"gradiumWebgpu",name:"phonon-tts WebGPU"},{id:"gradiumCpu",name:"phonon-tts CPU"},{id:"kokoro",name:"Kokoro 82M"}],Ne=[{id:"mac",name:"M3 Max",full:"MacBook Pro, M3 Max, Chrome"},{id:"pc",name:"RTX 2080 Ti",full:"PC, RTX 2080 Ti, Chrome"},{id:"iphone",name:"iPhone 17 Pro",full:"iPhone 17 Pro, Safari"}],T0=[{id:"warmRealtime",tab:"Speed",caption:"Seconds of speech per second of compute, warm.",unit:"x"},{id:"warmFirstMs",tab:"First audio",caption:"From the call to the first chunk of audio, warm.",unit:"ms"},{id:"coldStartMs",tab:"Cold start",caption:"A fresh page loads the runtime and speaks one sentence, with the files cached.",unit:"ms"},{id:"firstVisitMs",tab:"First visit",caption:"A fresh page with nothing cached. Downloads 43.7 MB, 64.5 MB (phonon-tts) or 326 MB (Kokoro).",unit:"ms"}],U=a=>document.getElementById(a),pt="http://www.w3.org/2000/svg";function S(a,e={},t){let r=document.createElementNS(pt,a);for(let[n,o]of Object.entries(e))r.setAttribute(n,String(o));return t?.append(r),r}var ze=a=>a>=1e3?`${(a/1e3).toFixed(1)} s`:a>=10?`${Math.round(a)} ms`:`${a.toFixed(1)} ms`,we=(a,e)=>e==="x"?`${Math.round(a)}\xD7`:ze(a);function ye(a){let e=document.createElement("div");return e.className="tip",e.hidden=!0,a.append(e),{show(t,r,n){e.innerHTML=t,e.hidden=!1;let o=e.offsetWidth/2;e.style.left=`${Math.min(Math.max(r,o),a.clientWidth-o)}px`,e.style.top=`${n}px`},hide:()=>e.hidden=!0}}function ke(a,e){let t=0,r=()=>{let n=Math.round(a.clientWidth);n&&n!==t&&e(t=n)};return new ResizeObserver(r).observe(a),r(),()=>{t=0,r()}}function q0(a,e,t){a.replaceChildren(...C.filter(r=>e.includes(r.id)).map(r=>{let n=document.createElement("li"),o=`<i class="s-${r.id}"></i>${r.name}`;if(!t)n.innerHTML=o;else{let s=document.createElement("button");s.innerHTML=o,s.setAttribute("aria-pressed","true"),s.onclick=()=>{let i=s.getAttribute("aria-pressed")!=="true";s.setAttribute("aria-pressed",String(i)),t(r.id,i)},n.append(s)}return n}))}var ve={};function dt(){let a=U("bars"),e=ye(a),t=T0[0],r=U("metric");for(let s of T0){let i=document.createElement("button");i.role="tab",i.textContent=s.tab,i.onclick=()=>{t=s;for(let h of r.children)h.setAttribute("aria-selected",String(h===i));U("metric-caption").textContent=s.caption,o()},r.append(i)}r.children[0].setAttribute("aria-selected","true"),U("metric-caption").textContent=t.caption,q0(U("bars-legend"),C.map(s=>s.id));let o=ke(a,s=>{let i=[...Ne.map(d=>({id:d.id,name:d.name,full:d.full,you:!1}))],h=ve[t.id];h&&i.push({id:"you",name:"This device",full:"This device",you:!0});let c=Math.round(Math.min(310,Math.max(250,s*.62))),u=26,p=36,b=(d,v)=>d==="you"?v==="ours"&&h?{median:h,p25:h,p75:h,n:1}:null:B0[d]?.[v]?.[t.id]??null,l=i.flatMap(d=>C.map(v=>b(d.id,v.id))).filter(d=>!!d),m=Math.min(...l.map(d=>d.p25))/2.2,w=Math.max(...l.map(d=>d.p75))*1.15,f=d=>u+(c-u-p)*(1-Math.log(d/m)/Math.log(w/m)),g=c-p,M=s<480?14:26,E=i.reduce((d,v)=>d+(v.you?1:C.length),0),x=Math.min(54,(s-M*(i.length-1))/E-4),k=d=>(d.you?1:C.length)*(x+4)-4,R=i.reduce((d,v)=>d+k(v),0)+M*(i.length-1),T=S("svg",{viewBox:`0 0 ${s} ${c}`,role:"img","aria-label":`${t.tab} by device`}),A=(s-R)/2;for(let d of i){let v=k(d),_=Math.max(...C.map(P=>b(d.id,P.id)?.median??0));(d.you?C.slice(0,1):C).forEach((P,F)=>{let B=b(d.id,P.id),G=A+F*(x+4),L=G+x/2,I=!B&&!!B0[d.id]?.[P.id]?.errors?.length,z=S("g",{},T);if(B){if(S("rect",{class:`bar s-${P.id}`,x:G,y:f(B.median),width:x,height:g-f(B.median),rx:4},z),B.p75>B.p25){S("line",{class:"whisker",x1:L,x2:L,y1:f(B.p25),y2:f(B.p75)},z);for(let Ke of[B.p25,B.p75])S("line",{class:"whisker",x1:L-4,x2:L+4,y1:f(Ke),y2:f(Ke)},z)}(x>=34||P.id==="ours")&&(S("text",{class:"val",x:L,y:Math.min(f(B.median),f(B.p75))-7,"text-anchor":"middle"},z).textContent=we(B.median,t.unit))}else I?S("rect",{class:"dnf",x:G+.6,y:f(_),width:x-1.2,height:g-f(_),rx:4},z):S("text",{x:L,y:g-8,"text-anchor":"middle"},z).textContent="\u2013";S("rect",{x:G-2,y:u-20,width:x+4,height:c-u-p+20,fill:"transparent"},z);let E0=B?`<b>${we(B.median,t.unit)}</b>${B.n>1?` \xB7 ${we(B.p25,t.unit)} to ${we(B.p75,t.unit)}`:""}`:I?"did not finish":"not measured",F0=d.you?"<br>your last run":"";z.onpointerenter=z.onpointerdown=()=>e.show(`${P.name} \xB7 ${d.full}<br>${E0}${F0}`,L,B?Math.min(f(B.median),f(B.p75))-18:g-30),z.onpointerleave=()=>e.hide()}),S("text",{class:d.you?"group you":"group",x:d.you?A+v:A+v/2,y:c-10,"text-anchor":d.you?"end":"middle"},T).textContent=d.name,A+=v+M}S("line",{class:"axis",x1:0,x2:s,y1:g,y2:g},T),a.querySelector("svg")?.remove(),a.prepend(T)});return o}function ft(a){a.innerHTML='<p class="caption">Each dot is one warm phrase. Log scales.</p><div class="tabs" role="tablist"></div><div class="chart"></div><ul class="legend"></ul>';let e=a.querySelector(".chart"),t=a.querySelector(".tabs"),r=ye(e),n=new Set,o=Ne[0];for(let c of Ne){let u=document.createElement("button");u.role="tab",u.textContent=c.name,u.setAttribute("aria-selected",String(c===o)),u.onclick=()=>{o=c;for(let p of t.children)p.setAttribute("aria-selected",String(p===u));h()},t.append(u)}let s=Object.fromEntries(je.phrases.map(c=>[c.id,c.text])),h=ke(e,c=>{let u=Math.round(Math.min(360,Math.max(240,c*.5))),p=40,b=10,l=8,m=30,w=k=>p+(c-p-b)*Math.log(k/1.5)/Math.log(45/1.5),f=k=>l+(u-l-m)*(1-Math.log(k/3)/Math.log(260/3)),g=S("svg",{viewBox:`0 0 ${c} ${u}`,role:"img","aria-label":`Speed against text length, ${o.full}`}),M=S("g",{class:"grid"},g);for(let k of[2,5,10,20,40])S("line",{x1:w(k),x2:w(k),y1:l,y2:u-m},M),S("text",{x:w(k),y:u-m+18,"text-anchor":"middle"},g).textContent=`${k} s`;for(let k of[5,10,20,50,100,200])S("line",{x1:p,x2:c-b,y1:f(k),y2:f(k)},M),S("text",{x:p-6,y:f(k)+4,"text-anchor":"end"},g).textContent=`${k}\xD7`;let E=[];for(let k of[...C].reverse()){if(n.has(k.id))continue;let R=S("g",{class:`s-${k.id}`,"fill-opacity":.6},g);for(let[T,A,d]of ut[o.id]?.[k.id]??[]){let v=w(T),_=f(A);S("circle",{cx:v,cy:_,r:3.2},R),E.push({px:v,py:_,rt:k.name,secs:T,speed:A,id:d})}}let x=S("circle",{class:"hot",r:5.5,fill:"none",visibility:"hidden"},g);g.onpointermove=g.onpointerdown=k=>{let R=g.getBoundingClientRect(),T=(k.clientX-R.left)*c/R.width,A=(k.clientY-R.top)*u/R.height,d=E[0],v=1/0;for(let y of E){let P=(y.px-T)**2+(y.py-A)**2;P<v&&(v=P,d=y)}if(!d||v>900)return r.hide(),x.setAttribute("visibility","hidden");x.setAttribute("cx",String(d.px)),x.setAttribute("cy",String(d.py)),x.setAttribute("visibility","visible");let _=s[d.id]??d.id;r.show(`<b>${Math.round(d.speed)}\xD7</b> \xB7 ${d.rt}<br>${d.secs.toFixed(1)} s: \u201C${_.length>44?`${_.slice(0,42)}\u2026`:_}\u201D`,d.px/c*e.clientWidth,d.py/u*e.clientHeight-8)},g.onpointerleave=()=>(r.hide(),x.setAttribute("visibility","hidden")),e.querySelector("svg")?.remove(),e.prepend(g)});q0(a.querySelector(".legend"),C.map(c=>c.id),(c,u)=>{u?n.delete(c):n.add(c),h()})}function mt(a){let e=ct;a.innerHTML=`<p class="caption">${e.words} words, 4 voices. Median absolute error ${Math.round(e.medianMs)} ms, ${Math.round(e.within100ms*100)}% within 100 ms.</p><div class="chart"></div>`;let t=a.querySelector(".chart"),r=ye(t),n=new Map;for(let s of e.errorsMs){let i=Math.max(-300,Math.min(280,Math.floor(s/20)*20));n.set(i,(n.get(i)??0)+1)}let o=Math.max(...n.values());ke(t,s=>{let i=Math.round(Math.min(240,Math.max(180,s*.32))),h=8,c=8,u=30,p=30,b=f=>h+(s-h-c)*(f+300)/600,l=f=>u+(i-u-p)*(1-f/o),m=S("svg",{viewBox:`0 0 ${s} ${i}`,role:"img","aria-label":"Histogram of word start errors"});for(let f=-300;f<300;f+=20){let g=n.get(f)??0,M=S("g",{},m);S("rect",{class:"bar s-ours",x:b(f)+1,y:l(g),width:b(f+20)-b(f)-2,height:i-p-l(g),rx:2},M),S("rect",{x:b(f),y:u,width:b(f+20)-b(f),height:i-u-p,fill:"transparent"},M),M.onpointerenter=M.onpointerdown=()=>r.show(`<b>${g} words</b><br>${f>0?"+":""}${f} to ${f+20>0?"+":""}${f+20} ms`,b(f+10)/s*t.clientWidth,l(g)/i*t.clientHeight-8),M.onpointerleave=()=>r.hide()}for(let f of[-300,-200,-100,0,100,200,300])S("text",{x:b(f),y:i-10,"text-anchor":f===-300?"start":f===300?"end":"middle"},m).textContent=`${f>0?"+":""}${f} ms`;let w=e.medianSignedMs;S("line",{class:"median",x1:b(w),x2:b(w),y1:u-12,y2:i-p},m),S("text",{x:b(w)+6,y:u-16},m).textContent=`median ${w>0?"+":""}${Math.round(w)} ms (later)`,S("line",{class:"axis",x1:h,x2:s-c,y1:i-p,y2:i-p},m),t.querySelector("svg")?.remove(),t.prepend(m)})}function gt(a){let e=lt;a.innerHTML='<p class="caption">Lines show 95% intervals.</p><div class="chart"></div>';let t=a.querySelector(".chart"),r=ye(t),n=[{key:"wer",title:"Word errors (Whisper)",max:2,f:o=>`${(o*100).toFixed(2)}%`,scale:100},{key:"utmos",title:"Predicted quality (UTMOS)",max:5,f:o=>o.toFixed(2),scale:1}];ke(t,o=>{let s=o>=560,i=s?(o-24)/2:o,h=30,c=Math.min(150,i*.42),u=26+C.length*h,p=s?u:u*2+16,b=S("svg",{viewBox:`0 0 ${o} ${p}`,role:"img","aria-label":"Word error rate and UTMOS per runtime"});n.forEach((l,m)=>{let w=s?m*(i+24):0,f=s?0:m*(u+16);S("text",{class:"group",x:w,y:f+14},b).textContent=l.title;let g=M=>w+c+(i-c-56)*M*l.scale/l.max;C.forEach((M,E)=>{let[x,k,R]=e[l.key][M.id]??[];if(x===void 0)return;let T=f+26+E*h,A=S("g",{},b);S("text",{x:w,y:T+15},A).textContent=M.name,S("rect",{class:`bar s-${M.id}`,x:g(0),y:T+4,width:Math.max(2,g(x)-g(0)),height:h-12,rx:3},A),S("line",{class:"whisker",x1:g(k),x2:g(R),y1:T+h/2-2,y2:T+h/2-2},A),S("text",{class:"val",x:Math.max(g(x),g(R))+6,y:T+15},A).textContent=l.f(x),S("rect",{x:w,y:T,width:i,height:h,fill:"transparent"},A),A.onpointerenter=A.onpointerdown=()=>r.show(`${M.name}<br><b>${l.f(x)}</b> \xB7 95% interval ${l.f(k)} to ${l.f(R)}`,(g(x)+0)/o*t.clientWidth,T/p*t.clientHeight),A.onpointerleave=()=>r.hide()})}),t.querySelector("svg")?.remove(),t.prepend(b)})}function bt(a){let e=document.createElement("select");e.setAttribute("aria-label","Phrase");for(let i of je.phrases)e.add(new Option(i.text.length>90?`${i.text.slice(0,88)}\u2026`:i.text,i.id));let t=document.createElement("div");t.className="players";let r=new Audio,n=null,o=()=>{r.pause(),n?.setAttribute("aria-pressed","false"),n=null};r.onended=o;let s={ours:"WebGPU",gradiumWebgpu:"WebGPU",gradiumCpu:"CPU",kokoro:"different model"};for(let i of C){let h=document.createElement("button");h.setAttribute("aria-pressed","false"),h.innerHTML=`<i class="s-${i.id}"></i><span>${i.name.replace(/ (WebGPU|CPU)$/,"")}<small>${s[i.id]}</small></span>`,h.onclick=()=>{let c=n===h;o(),!c&&(r.src=`audio/${i.id}/${e.value}.mp3`,r.play(),n=h,h.setAttribute("aria-pressed","true"))},t.append(h)}e.onchange=o,a.append(e,t)}async function wt(a){let e=U("speak"),t=U("words"),r=U("speed");r.oninput=()=>U("speed-out").textContent=`${Number(r.value).toFixed(2).replace(/0$/,"")}\xD7`;let n=U("voice"),o=(v,_,y)=>{let P=U(v);P.querySelector("b").textContent=_,y!==void 0&&(P.querySelector("span").textContent=y)},i=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"}).catch(()=>null);if(!i){e.disabled=!0,o("s-load","No WebGPU","in this browser");return}let h=i.info;U("device").textContent=`GPU: ${[h.vendor,h.architecture].filter(Boolean).join(" ")||"unknown"}`;let c=null,u=null,p=!1,b=Promise.resolve(0),l=v=>b=b.then(async()=>{let _=performance.now(),y=0;return await v.speak("Hello there.",{voice:n.value,onAudio:()=>y||=performance.now()-_}).catch(()=>{}),y}),m=()=>u??=(async()=>{let v=performance.now();o("s-load","\u2026","loading");let _=await be.load({progress:F=>{p=!0,o("s-load",`${F}%`,"of 43.7 MB")}}),y=performance.now()-v;n.replaceChildren(..._.voices.map(F=>new Option(F.name,F.name))),n.value="Delia";let P=y+await l(_);return ve[p?"firstVisitMs":"coldStartMs"]=P,o("s-load",ze(P),p?"first visit":"cold start"),a(),c=_,Object.assign(globalThis,{phonon:_}),_})().catch(v=>{throw u=null,o("s-load","Failed",String(v?.message??v)),v});n.onchange=()=>c&&void l(c),await(await caches.open(ie).catch(()=>null))?.match(J)&&m().catch(()=>{});let f=null,g=[],M=null,E=null,x=0,k=()=>{for(let v of g)v.stop();g=[],cancelAnimationFrame(x);for(let v of t.children)v.classList.remove("now")},R=(v,_)=>{let y=f.createBuffer(1,v.length,24e3);y.copyToChannel(v,0);let P=f.createBufferSource();P.buffer=y,P.connect(f.destination),P.start(_),g.push(P)},T=(v,_)=>{let y=[...t.children],P=()=>{let F=f.currentTime-_;y.forEach((B,G)=>B.classList.toggle("now",F>=v.words[G].start&&F<v.words[G].end)),F<v.audio.length/24e3?x=requestAnimationFrame(P):A()};x=requestAnimationFrame(P)},A=()=>{e.textContent="Speak",M=null,g=[]},d=v=>{t.classList.remove("empty"),t.replaceChildren(...v.words.flatMap(_=>{let y=document.createElement("span");return y.textContent=_.text,y.onclick=()=>{if(!f||E!==v)return;k();let P=_.start,F=f.currentTime+.03,B=f.createBufferSource(),G=f.createBuffer(1,v.audio.length,24e3);G.copyToChannel(v.audio,0),B.buffer=G,B.connect(f.destination),B.start(F,P),g=[B],e.textContent="Stop",T(v,F-P)},[y,document.createTextNode(" ")]}))};e.onclick=async()=>{if(M||g.length){M?.abort(),k(),A();return}f??=new AudioContext({sampleRate:24e3}),f.resume(),M=new AbortController,e.textContent="Stop";let v=M.signal;try{let _=c??await m();if(await b,v.aborted)return;let y=performance.now(),P=0,F=0,B=0;o("s-first","\u2026");let G=()=>{let I=performance.now()-y;o("s-speed",`${Math.round(F/(I/1e3))}\xD7`),o("s-audio",`${F.toFixed(1)} s`)};U("s-speed").classList.add("live");let L=await _.speak(U("text").value,{voice:n.value,speed:Number(r.value),signal:v,onAudio:I=>{P||(P=performance.now()-y,o("s-first",ze(P)),B=f.currentTime+.05),R(I,B),B+=I.length/24e3,F+=I.length/24e3,G()}});U("s-speed").classList.remove("live"),G(),E=L,ve.warmRealtime=F/((performance.now()-y)/1e3),ve.warmFirstMs=P,a(),d(L),T(L,B-L.audio.length/24e3)}catch(_){v.aborted||o("s-first","Failed",String(_?.message??_)),k(),A()}}}var vt=dt();for(let a of document.querySelectorAll("[data-chart]")){let e={length:ft,timing:mt,quality:gt}[a.dataset.chart];e?.(a)}var R0=document.getElementById("listen-players");R0&&bt(R0);wt(vt);
