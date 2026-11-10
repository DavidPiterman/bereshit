// ===== engine: a tiny GPT trained live with tf.js (GPU), read out by a hand-written CPU forward pass =====
const CHARS = " אבגדהוזחטיךכלםמןנסעףפץצקרשת.,'\"-?!:׃";
const V = CHARS.length;
const C2I = {}; for (let i = 0; i < V; i++) C2I[CHARS[i]] = i;
const CFG = { d: 128, L: 3, H: 4, T: 64, B: 48 };
CFG.hid = CFG.d * 4;

function encodeText(s) {
  const out = new Uint8Array(s.length); let n = 0, lastSpace = true;
  for (let i = 0; i < s.length; i++) {
    let c = s[i];
    let id = C2I[c];
    if (id === undefined) id = 0;
    if (id === 0) { if (lastSpace) continue; lastSpace = true; } else lastSpace = false;
    out[n++] = id;
  }
  return out.slice(0, n);
}

const Engine = {
  vars: null, list: [], names: [], masks: [], opt: null, step: 0, causal: null, data: null, backend: '',
  async init() {
    try { await tf.setBackend('webgl'); await tf.ready(); } catch (e) { await tf.setBackend('cpu'); await tf.ready(); }
    this.backend = tf.getBackend();
    this.build();
  },
  build() {
    const { d, L, T, hid } = CFG;
    if (this.list.length) this.list.forEach(v => v.dispose());
    if (this.masks.length) this.masks.forEach(v => v.dispose());
    if (this.opt) this.opt.dispose();
    this.list = []; this.names = []; this.masks = [];
    const add = (name, t) => { const v = tf.variable(t, true, name + '_' + Math.random().toString(36).slice(2)); this.list.push(v); this.names.push(name); return v; };
    const rn = (shape, sd) => tf.randomNormal(shape, 0, sd);
    const P = { layers: [] };
    P.wte = add('wte', rn([V, d], 0.05));
    P.wpe = add('wpe', rn([T, d], 0.02));
    for (let l = 0; l < L; l++) {
      const y = {};
      y.g1 = add(`g1_${l}`, tf.ones([d])); y.b1n = add(`b1n_${l}`, tf.zeros([d]));
      y.wqkv = add(`wqkv_${l}`, rn([d, 3 * d], 0.04)); y.bqkv = add(`bqkv_${l}`, tf.zeros([3 * d]));
      y.wo = add(`wo_${l}`, rn([d, d], 0.04 / Math.sqrt(2 * L))); y.bo = add(`bo_${l}`, tf.zeros([d]));
      y.g2 = add(`g2_${l}`, tf.ones([d])); y.b2n = add(`b2n_${l}`, tf.zeros([d]));
      y.w1 = add(`w1_${l}`, rn([d, hid], 0.04)); y.b1 = add(`b1_${l}`, tf.zeros([hid]));
      y.w2 = add(`w2_${l}`, rn([hid, d], 0.04 / Math.sqrt(2 * L))); y.b2 = add(`b2_${l}`, tf.zeros([d]));
      P.layers.push(y);
      this.masks.push(tf.variable(tf.ones([hid]), false));
    }
    P.gf = add('gf', tf.ones([d])); P.bf = add('bf', tf.zeros([d]));
    P.wh = add('wh', rn([d, V], 0.04)); P.bh = add('bh', tf.zeros([V]));
    this.vars = P;
    if (this.causal) this.causal.dispose();
    const m = new Float32Array(T * T);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) m[i * T + j] = j <= i ? 0 : -1e9;
    this.causal = tf.tensor(m, [1, 1, T, T]);
    this.opt = tf.train.adam(2e-3, 0.9, 0.99, 1e-8);
    this.step = 0;
  },
  ln(x, g, b) {
    const mom = tf.moments(x, -1, true);
    return x.sub(mom.mean).div(mom.variance.add(1e-5).sqrt()).mul(g).add(b);
  },
  logits(xOH, B) {
    const { d, L, H, T } = CFG, P = this.vars, dh = d / H;
    let x = xOH.matMul(P.wte).reshape([B, T, d]).add(P.wpe).reshape([B * T, d]);
    for (let l = 0; l < L; l++) {
      const y = P.layers[l];
      let h = this.ln(x, y.g1, y.b1n);
      const qkv = h.matMul(y.wqkv).add(y.bqkv);
      const [q, k, v] = tf.split(qkv, 3, 1).map(t => t.reshape([B, T, H, dh]).transpose([0, 2, 1, 3]));
      const att = tf.softmax(tf.matMul(q, k, false, true).mul(1 / Math.sqrt(dh)).add(this.causal), -1);
      const a = tf.matMul(att, v).transpose([0, 2, 1, 3]).reshape([B * T, d]);
      x = x.add(a.matMul(y.wo).add(y.bo));
      h = this.ln(x, y.g2, y.b2n);
      const mm = tf.relu(h.matMul(y.w1).add(y.b1)).mul(this.masks[l]);
      x = x.add(mm.matMul(y.w2).add(y.b2));
    }
    x = this.ln(x, P.gf, P.bf);
    return x.matMul(P.wh).add(P.bh);
  },
  setData(u8) { this.data = u8; },
  lr() {
    const s = this.step;
    if (s < 25) return 2.5e-3 * (s + 1) / 25;
    const t = Math.min(1, (s - 25) / 2500);
    return 6e-4 + (2.5e-3 - 6e-4) * 0.5 * (1 + Math.cos(Math.PI * t));
  },
  async trainStep() {
    const { T, B } = CFG, data = this.data, N = data.length - T - 1;
    const xi = new Int32Array(B * T), yi = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const s = Math.floor(Math.random() * N);
      for (let t = 0; t < T; t++) { xi[b * T + t] = data[s + t]; yi[b * T + t] = data[s + t + 1]; }
    }
    this.opt.learningRate = this.lr();
    const lossT = tf.tidy(() => {
      const x = tf.oneHot(tf.tensor1d(xi, 'int32'), V).toFloat();
      const yOH = tf.oneHot(tf.tensor1d(yi, 'int32'), V).toFloat();
      return this.opt.minimize(() => {
        const lp = tf.logSoftmax(this.logits(x, B), -1);
        return lp.mul(yOH).sum(-1).mean().neg();
      }, true, this.list);
    });
    const v = (await lossT.data())[0];
    lossT.dispose();
    this.step++;
    return v;
  },
  async snapshot() {
    const flat = tf.tidy(() => tf.concat(this.list.map(v => v.reshape([-1]))));
    const all = await flat.data(); flat.dispose();
    const w = {}; let o = 0;
    this.list.forEach((v, i) => { const n = v.size; w[this.names[i]] = all.subarray(o, o + n); o += n; });
    return w;
  },
  setMask(l, arr) { const t = tf.tensor1d(arr); this.masks[l].assign(t); t.dispose(); }
};

// ---- CPU mind: same network, one character at a time, with cached keys/values ----
class Mind {
  constructor() {
    const { d, L, H, T, hid } = CFG;
    this.w = null; this.pos = 0; this.hist = [];
    this.K = []; this.Vc = []; this.acts = []; this.att = []; this.mask = [];
    for (let l = 0; l < L; l++) {
      this.K.push(new Float32Array(T * d)); this.Vc.push(new Float32Array(T * d));
      this.acts.push(new Float32Array(hid)); this.att.push(new Float32Array(H * T));
      this.mask.push(new Float32Array(hid).fill(1));
    }
    this.x = new Float32Array(d); this.h = new Float32Array(d); this.qkv = new Float32Array(3 * d);
    this.a = new Float32Array(d); this.sc = new Float32Array(T); this.logits = new Float32Array(V);
    this.probs = new Float32Array(V);
  }
  setWeights(w) { const first = !this.w; this.w = w; if (!first) this.replay(); }
  replay() { const h = this.hist.slice(-(CFG.T >> 1)); this.pos = 0; this.hist = []; for (const t of h) this.step(t); }
  reset() { this.pos = 0; this.hist = []; }
  _ln(src, dst, g, b) {
    const d = CFG.d; let m = 0; for (let i = 0; i < d; i++) m += src[i]; m /= d;
    let v = 0; for (let i = 0; i < d; i++) { const t = src[i] - m; v += t * t; } v = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) dst[i] = (src[i] - m) * v * g[i] + b[i];
  }
  step(tok) {
    const { d, L, H, T, hid } = CFG, w = this.w, dh = d / H, scale = 1 / Math.sqrt(dh);
    if (this.pos >= T) this.replay();
    const pos = this.pos, x = this.x, h = this.h, qkv = this.qkv, a = this.a, sc = this.sc;
    this.hist.push(tok);
    const wte = w.wte, wpe = w.wpe;
    for (let i = 0; i < d; i++) x[i] = wte[tok * d + i] + wpe[pos * d + i];
    for (let l = 0; l < L; l++) {
      this._ln(x, h, w['g1_' + l], w['b1n_' + l]);
      const Wq = w['wqkv_' + l], bq = w['bqkv_' + l], d3 = 3 * d;
      for (let j = 0; j < d3; j++) qkv[j] = bq[j];
      for (let i = 0; i < d; i++) { const hi = h[i], r = i * d3; for (let j = 0; j < d3; j++) qkv[j] += hi * Wq[r + j]; }
      const K = this.K[l], Vc = this.Vc[l], att = this.att[l];
      for (let i = 0; i < d; i++) { K[pos * d + i] = qkv[d + i]; Vc[pos * d + i] = qkv[2 * d + i]; }
      for (let hd = 0; hd < H; hd++) {
        const o = hd * dh; let mx = -1e30;
        for (let j = 0; j <= pos; j++) { let s = 0; const r = j * d + o; for (let i = 0; i < dh; i++) s += qkv[o + i] * K[r + i]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
        let z = 0; for (let j = 0; j <= pos; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
        for (let i = 0; i < dh; i++) a[o + i] = 0;
        for (let j = 0; j <= pos; j++) { const p = sc[j] / z; att[hd * T + j] = p; const r = j * d + o; for (let i = 0; i < dh; i++) a[o + i] += p * Vc[r + i]; }
      }
      const Wo = w['wo_' + l], bo = w['bo_' + l];
      for (let j = 0; j < d; j++) x[j] += bo[j];
      for (let i = 0; i < d; i++) { const ai = a[i], r = i * d; for (let j = 0; j < d; j++) x[j] += ai * Wo[r + j]; }
      this._ln(x, h, w['g2_' + l], w['b2n_' + l]);
      const W1 = w['w1_' + l], b1 = w['b1_' + l], m = this.acts[l], mk = this.mask[l];
      for (let j = 0; j < hid; j++) m[j] = b1[j];
      for (let i = 0; i < d; i++) { const hi = h[i], r = i * hid; for (let j = 0; j < hid; j++) m[j] += hi * W1[r + j]; }
      const W2 = w['w2_' + l], b2 = w['b2_' + l];
      for (let j = 0; j < d; j++) x[j] += b2[j];
      for (let i = 0; i < hid; i++) {
        let mi = m[i]; mi = mi > 0 ? mi * mk[i] : 0; m[i] = mi;
        if (mi !== 0) { const r = i * d; for (let j = 0; j < d; j++) x[j] += mi * W2[r + j]; }
      }
    }
    this._ln(x, h, w.gf, w.bf);
    const Wh = w.wh, bh = w.bh, lg = this.logits;
    for (let j = 0; j < V; j++) lg[j] = bh[j];
    for (let i = 0; i < d; i++) { const hi = h[i], r = i * V; for (let j = 0; j < V; j++) lg[j] += hi * Wh[r + j]; }
    this.pos++;
    return lg;
  }
  sample(temp) {
    const lg = this.logits, p = this.probs; let mx = -1e30;
    for (let j = 0; j < V; j++) if (lg[j] > mx) mx = lg[j];
    let z = 0; for (let j = 0; j < V; j++) { p[j] = Math.exp((lg[j] - mx) / temp); z += p[j]; }
    let r = Math.random() * z, k = 0;
    for (; k < V - 1; k++) { r -= p[k]; if (r <= 0) break; }
    this.lastP = p[k] / z;
    return k;
  }
}
