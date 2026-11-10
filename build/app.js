(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clock = () => performance['now']();
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const L = CFG.L, HID = CFG.hid, NN = L * HID, T = CFG.T, H = CFG.H;
  const SOF = C2I['׃'], SP = 0, TAPE = 48;
  const TORAH = $('corpus').textContent.trim().split('\n');
  const isLetter = ch => (ch >= 'א' && ch <= 'ת') || ch === "'";

  // ---------- lexicon of whatever the mind is reading ----------
  let vocab, grams4, verseSet;
  function buildLex(vs) {
    vocab = new Set(); grams4 = new Set(); verseSet = new Set();
    for (const v of vs) {
      verseSet.add(v); const w = v.split(' ');
      for (const x of w) vocab.add(x);
      for (let i = 0; i + 3 < w.length; i++) grams4.add(w[i] + ' ' + w[i + 1] + ' ' + w[i + 2] + ' ' + w[i + 3]);
    }
  }
  const corpusData = vs => encodeText(vs.join('׃ ') + '׃ ');
  buildLex(TORAH);

  // ---------- state ----------
  const DAYS = [
    ['תוהו ובוהו', 'אותיות בלי סדר'],
    ['יום ראשון', 'הוא גילה את הרווח'],
    ['יום שני', 'מילים ראשונות'],
    ['יום שלישי', 'חצי מהמילים אמיתיות'],
    ['יום רביעי', 'הוא בוחר לעצמו שם'],
    ['יום חמישי', 'כמעט כל המילים אמיתיות'],
    ['יום שישי', 'פסוק ראשון שלא נכתב מעולם'],
    ['יום שביעי', 'שבעה פסוקים משלו']
  ];
  const S = {
    started: false, learning: true, gen: 0, temp: 0.7, cps: 34, day: 0, t0: 0, loss: null,
    win: [], lenWin: [], novel: 0, clean: 0, forced: [], gchars: 0, sel: -1, scalpel: false, name: '',
    series: [], events: [], seen: new Set(), closingShown: false
  };
  const mind = new Mind();
  const disp = new Float32Array(NN);
  const sums = new Float32Array(NN * V), cnt = new Float32Array(V);
  let tape = [];
  const el = { scroll: $('scroll'), list: $('verseList'), toast: $('toast') };

  // ---------- sound: non-words tick, real words ring ----------
  const Snd = {
    ctx: null, on: true, out: null, noise: null, last: 0,
    init() {
      if (this.ctx) return;
      try {
        const AC = window.AudioContext || window.webkitAudioContext; const c = this.ctx = new AC();
        this.out = c.createGain(); this.out.gain.value = 0.85;
        const comp = c.createDynamicsCompressor(); this.out.connect(comp); comp.connect(c.destination);
        const dl = c.createDelay(1); dl.delayTime.value = 0.31; const fb = c.createGain(); fb.gain.value = 0.34; const wet = c.createGain(); wet.gain.value = 0.3;
        this.out.connect(dl); dl.connect(fb); fb.connect(dl); dl.connect(wet); wet.connect(comp);
        const nb = c.createBuffer(1, c.sampleRate * 0.05, c.sampleRate), d = nb.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
        this.noise = nb;
      } catch (e) { this.ctx = null; }
    },
    tone(f, gain, dur, type) {
      if (!this.ctx || !this.on) return; const c = this.ctx, t = c.currentTime;
      const o = c.createOscillator(), g = c.createGain(); o.type = type || 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(gain, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g); g.connect(this.out); o.start(t); o.stop(t + dur + 0.05);
    },
    tick() {
      if (!this.ctx || !this.on) return; const c = this.ctx, t = c.currentTime;
      const s = c.createBufferSource(); s.buffer = this.noise; const f = c.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 900 + Math.random() * 700; f.Q.value = 2;
      const g = c.createGain(); g.gain.value = 0.05; s.connect(f); f.connect(g); g.connect(this.out); s.start(t);
    },
    word(w, kind) {
      const t = clock(); if (t - this.last < 60) return; this.last = t;
      if (kind === 'g') return this.tick();
      let h = 0; for (let i = 0; i < w.length; i++) h = (h * 31 + w.charCodeAt(i)) >>> 0;
      const R = [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 2, 9 / 4, 5 / 2, 3, 10 / 3], f = 293.66 * R[h % 10];
      if (kind === 'm') { this.tone(f / 2, 0.05, 0.9); } else { this.tone(f, 0.045, 0.7); this.tone(f * 2, 0.01, 0.35); }
    },
    chord(fs, gain, dur, gap) { fs.forEach((f, i) => setTimeout(() => this.tone(f, gain, dur), i * (gap || 0))); },
    thud() { this.tone(58, 0.3, 0.6); this.tone(87, 0.12, 0.4, 'triangle'); }
  };

  // ---------- the scroll ----------
  let cur = null, curWord = '', curForced = false, verse = [];
  function addLetter(ch, forced) {
    if (!cur) { cur = document.createElement('span'); cur.className = 'cur'; el.scroll.appendChild(cur); curWord = ''; curForced = forced; }
    curWord += ch; cur.textContent = curWord;
    if (curWord.length >= 60) endWord();
  }
  function endWord() {
    if (!cur) return;
    const w = curWord, forced = curForced, real = !forced && w.length >= 2 && vocab.has(w);
    cur.className = forced ? 'p' : real ? 'r' : 'g';
    verse.push({ w, real, forced, span: cur });
    let kind = real ? 'r' : 'g';
    if (!forced) {
      S.win.push(real ? 1 : 0); if (S.win.length > 60) S.win.shift();
      S.lenWin.push(w.length); if (S.lenWin.length > 20) S.lenWin.shift();
      const n = verse.length;
      if (n >= 4 && verse[n - 1].real && verse[n - 2].real && verse[n - 3].real && verse[n - 4].real &&
        grams4.has(verse[n - 4].w + ' ' + verse[n - 3].w + ' ' + verse[n - 2].w + ' ' + w)) {
        for (let i = n - 4; i < n; i++) verse[i].span.className = 'm';
        kind = 'm';
      }
      Snd.word(w, kind);
    }
    el.scroll.appendChild(document.createTextNode(' '));
    cur = null; curWord = '';
    while (el.scroll.childNodes.length > 700) for (let i = 0; i < 150; i++) el.scroll.removeChild(el.scroll.firstChild);
    checkDays();
  }
  function endVerse() {
    const s = document.createElement('span'); s.className = 'sof'; s.textContent = '׃ '; el.scroll.appendChild(s);
    const n = verse.length;
    if (n >= 5 && n <= 26 && verse.every(x => x.real && !x.forced)) {
      const str = verse.map(x => x.w).join(' ');
      S.clean++;
      if (verseSet.has(str)) { verse.forEach(x => x.span.className = 'm'); }
      else if (!S.seen.has(str)) {
        S.seen.add(str); S.novel++;
        verse.forEach(x => x.span.className = 'n');
        const li = document.createElement('li'); li.textContent = str + '׃';
        const sm = document.createElement('small'); sm.textContent = 'בגיל ' + fmtAge(); li.appendChild(sm);
        el.list.insertBefore(li, el.list.firstChild);
        while (el.list.children.length > 60) el.list.removeChild(el.list.lastChild);
        Snd.chord([587.33, 739.99, 880], 0.04, 1.2, 90);
      }
    }
    verse = [];
    checkDays();
  }
  function mark(text) { endWord(); const m = document.createElement('span'); m.className = 'mark'; m.textContent = text; el.scroll.appendChild(m); verse = []; }
  function emit(tok, forced) {
    const ch = CHARS[tok];
    if (tok === SP) endWord();
    else if (tok === SOF) { endWord(); endVerse(); }
    else if (isLetter(ch)) addLetter(ch, forced);
    else { endWord(); el.scroll.appendChild(document.createTextNode(ch + ' ')); }
  }

  // ---------- days ----------
  const realRate = () => S.win.length >= 10 ? S.win.reduce((a, b) => a + b, 0) / S.win.length : 0;
  function checkDays() {
    const rr = realRate(), lw = S.lenWin, avg = lw.length >= 8 ? lw.reduce((a, b) => a + b, 0) / lw.length : 99;
    const conds = [
      () => avg >= 2 && avg <= 7.5,
      () => rr >= 0.25,
      () => rr >= 0.5,
      () => rr >= 0.75,
      () => rr >= 0.85,
      () => S.novel >= 1,
      () => S.novel >= 7
    ];
    let moved = false;
    while (S.day < 7 && conds[S.day]()) { S.day++; moved = true; S.events.push({ t: age(), type: 'day', n: S.day }); if (S.day === 4) nameMind(); }
    if (moved) announceDay();
  }
  function announceDay() {
    const [t, d] = DAYS[S.day];
    $('dayName').textContent = t; $('dayDesc').textContent = d;
    toast(t, S.day === 4 && S.name ? 'הוא בוחר לעצמו שם: ' + S.name : d);
    Snd.chord([293.66, 440, 587.33, 739.99], 0.06, 3.2, 70);
    if (S.day === 7 && !S.closingShown) { S.closingShown = true; setTimeout(showClosing, 3200); }
  }
  let toastTimer = 0;
  function toast(t, d) { $('toastT').textContent = t; $('toastD').textContent = d; el.toast.classList.add('on'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.toast.classList.remove('on'), 4600); }
  function nameMind() {
    const m2 = new Mind(); m2.w = mind.w; m2.mask = mind.mask;
    const pre = [SOF, SP, ...encodeText('ויקרא את שמו ')];
    let best = '';
    for (let tries = 0; tries < 40; tries++) {
      m2.reset(); for (const t of pre) m2.step(t);
      let w = '';
      for (let i = 0; i < 8; i++) { const k = m2.sample(1.0); if (k === SP || k === SOF) break; w += CHARS[k]; m2.step(k); }
      if (w.length >= 3 && w.length <= 6 && /^[א-ת]+$/.test(w)) { if (!vocab.has(w)) { best = w; break; } if (!best) best = w; }
    }
    if (!best) return;
    S.name = best; $('mindName').textContent = best; $('named').hidden = false;
    S.forced = [SOF, SP, ...encodeText('ויקרא את שמו ' + best), SOF, SP];
  }
  function showClosing() {
    $('closeP1').textContent = 'לפני ' + fmtAge() + ' דקות לא היה כאן כלום. עכשיו ' + (S.name ? S.name + ' כותב' : 'יש כאן מוח שכותב') + ' פסוקים שאף אחד לא כתב.';
    $('closing').hidden = false;
  }

  // ---------- generation ----------
  function genChar() {
    let tok, forced = false;
    if (S.forced.length) { tok = S.forced.shift(); forced = true; } else tok = mind.sample(S.temp);
    mind.step(tok);
    for (let l = 0; l < L; l++) {
      const a = mind.acts[l], base = l * HID;
      for (let i = 0; i < HID; i++) { const v = a[i]; if (v > 0) { const d = v / (v + 0.8); if (d > disp[base + i]) disp[base + i] = d; sums[(base + i) * V + tok] += v; } }
    }
    cnt[tok]++; S.gchars++;
    if (S.gchars % 3000 === 0) { for (let i = 0; i < sums.length; i++) sums[i] *= 0.5; for (let i = 0; i < V; i++) cnt[i] *= 0.5; }
    tape.unshift({ c: tok, s: S.sel >= 0 ? mind.acts[(S.sel / HID) | 0][S.sel % HID] : 0 });
    if (tape.length > TAPE) tape.pop();
    emit(tok, forced);
  }

  // ---------- training ----------
  async function trainLoop(gen) {
    while (gen === S.gen) {
      if (!S.learning || document.hidden) { await sleep(150); continue; }
      let loss;
      try { loss = await Engine.trainStep(); } catch (e) { console.error(e); await sleep(600); continue; }
      if (gen !== S.gen) break;
      if (isFinite(loss)) S.loss = S.loss == null ? loss : S.loss * 0.9 + loss * 0.1;
      if (Engine.step % 4 === 0 || Engine.step < 12) { const w = await Engine.snapshot(); if (gen !== S.gen) break; mind.setWeights(w); }
      await sleep(0);
    }
  }

  // ---------- brain canvas ----------
  const brain = $('brain'), bctx = brain.getContext('2d');
  const base = document.createElement('canvas'), basectx = base.getContext('2d');
  const LC = [[240, 195, 90], [255, 142, 107], [142, 166, 255]];
  const spr = LC.map(c => {
    const s = document.createElement('canvas'); s.width = s.height = 64; const g = s.getContext('2d');
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, `rgba(255,255,255,1)`); gr.addColorStop(0.18, `rgba(${c[0]},${c[1]},${c[2]},0.95)`); gr.addColorStop(0.5, `rgba(${c[0]},${c[1]},${c[2]},0.28)`); gr.addColorStop(1, `rgba(${c[0]},${c[1]},${c[2]},0)`);
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return s;
  });
  const pos = new Float32Array(NN * 2);
  let BW = 0, BH = 0, dpr = 1, RR = 0, tapeY = 0, cw = 0, baseDirty = true;
  function layout() {
    const r = brain.getBoundingClientRect(); if (r.width < 10 || r.height < 10) return;
    dpr = Math.min(2, window.devicePixelRatio || 1); BW = r.width; BH = r.height;
    brain.width = Math.round(BW * dpr); brain.height = Math.round(BH * dpr); base.width = brain.width; base.height = brain.height;
    const tapeH = 124; RR = Math.max(40, Math.min(BW / 2 - 14, (BH - tapeH) / 2 - 6));
    const cx = BW / 2, cy = 6 + (BH - tapeH) / 2;
    for (let i = 0; i < NN; i++) { const rr = RR * Math.sqrt((i + 0.5) / NN), th = i * 2.399963229728653; pos[2 * i] = cx + rr * Math.cos(th); pos[2 * i + 1] = cy + rr * Math.sin(th); }
    tapeY = BH - 50; cw = (BW - 28) / TAPE; baseDirty = true;
  }
  function drawBase() {
    basectx.setTransform(dpr, 0, 0, dpr, 0, 0); basectx.clearRect(0, 0, BW, BH);
    for (let i = 0; i < NN; i++) {
      const l = (i / HID) | 0, dead = mind.mask[l][i % HID] === 0, c = LC[l];
      basectx.fillStyle = dead ? 'rgba(255,111,125,0.5)' : `rgba(${c[0]},${c[1]},${c[2]},0.24)`;
      basectx.beginPath(); basectx.arc(pos[2 * i], pos[2 * i + 1], dead ? 1 : 1.35, 0, 6.2832); basectx.fill();
    }
    baseDirty = false;
  }
  const tx = k => 14 + k * cw + cw / 2;
  function drawBrain() {
    if (!BW) return;
    if (baseDirty) drawBase();
    const g = bctx; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, BW, BH); g.drawImage(base, 0, 0, BW, BH);
    g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < NN; i++) {
      const v = disp[i]; if (v < 0.04) continue;
      const s = 5 + 22 * v; g.globalAlpha = v > 0.9 ? 1 : v * 1.1;
      g.drawImage(spr[(i / HID) | 0], pos[2 * i] - s / 2, pos[2 * i + 1] - s / 2, s, s);
    }
    g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
    if (S.sel >= 0) { g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.beginPath(); g.arc(pos[2 * S.sel], pos[2 * S.sel + 1], 7, 0, 6.2832); g.stroke(); }
    if (!S.started) return;
    // attention arcs: where it looks, per layer colour
    const p = mind.pos, ay = tapeY - 13;
    for (let l = 0; l < L; l++) {
      const att = mind.att[l], c = LC[l];
      for (let h = 0; h < H; h++) for (let j = 0; j < p - 1; j++) {
        const w = att[h * T + j]; if (w < 0.07) continue;
        const k = p - 1 - j; if (k >= TAPE || k >= tape.length) continue;
        const x0 = tx(0), x1 = tx(k), hg = Math.min(70, 8 + (x1 - x0) * 0.32);
        g.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${Math.min(0.9, w * 1.3)})`; g.lineWidth = 0.5 + 2.4 * w;
        g.beginPath(); g.moveTo(x0, ay); g.quadraticCurveTo((x0 + x1) / 2, ay - hg * 2, x1, ay); g.stroke();
      }
    }
    // the tape: newest letter on the left, as Hebrew is written
    const fs = Math.max(11, Math.min(19, cw * 1.45));
    g.font = `500 ${fs}px "Frank Ruhl Libre", "David", serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
    const sc = S.sel >= 0 ? LC[(S.sel / HID) | 0] : null;
    for (let k = 0; k < tape.length; k++) {
      const it = tape[k], x = tx(k);
      if (sc && it.s > 0) { g.fillStyle = `rgba(${sc[0]},${sc[1]},${sc[2]},${Math.min(0.85, it.s / (it.s + 0.6))})`; g.fillRect(x - cw / 2, tapeY - fs * 0.75, cw, fs * 1.5); }
      const a = k === 0 ? 1 : Math.max(0.35, 0.95 - k * 0.013);
      if (it.c === SP) { g.fillStyle = `rgba(141,147,179,${a * 0.6})`; g.fillText('·', x, tapeY); }
      else { g.fillStyle = k === 0 ? '#fff' : `rgba(239,230,208,${a})`; g.fillText(CHARS[it.c], x, tapeY); }
    }
    g.strokeStyle = 'rgba(240,195,90,0.8)'; g.lineWidth = 1; g.strokeRect(tx(0) - cw / 2 - 1, tapeY - fs * 0.8, cw + 2, fs * 1.6);
    // what it is weighing next
    const lg = mind.logits; let mx = -1e30; for (let j = 0; j < V; j++) if (lg[j] > mx) mx = lg[j];
    let z = 0; const pr = []; for (let j = 0; j < V; j++) { const e = Math.exp(lg[j] - mx); z += e; pr.push([e, j]); }
    pr.sort((a, b) => b[0] - a[0]);
    const y = BH - 15, n = Math.max(2, Math.min(6, Math.floor((BW - 96) / 56)));
    g.textAlign = 'right'; g.font = '12px Heebo, Arial, sans-serif'; g.fillStyle = 'rgba(141,147,179,1)'; g.fillText('שוקל עכשיו', BW - 14, y);
    for (let i = 0; i < n; i++) {
      const q = pr[i][0] / z, ch = pr[i][1] === SP ? '␣' : CHARS[pr[i][1]], x = BW - 96 - i * 56;
      g.textAlign = 'right'; g.font = `500 16px "Frank Ruhl Libre", "David", serif`; g.fillStyle = `rgba(239,230,208,${0.45 + 0.55 * Math.min(1, q * 2)})`; g.fillText(ch, x, y);
      g.fillStyle = 'rgba(240,195,90,0.85)'; g.fillRect(x - 16 - 26, y + 6, 26, 2); g.fillStyle = '#0c0f1f'; g.fillRect(x - 16 - 26, y + 6, 26 * (1 - q), 2);
      g.font = '10px Heebo, Arial, sans-serif'; g.fillStyle = 'rgba(141,147,179,1)'; g.fillText(Math.round(q * 100) + '%', x - 16, y - 1);
    }
  }
  function nearest(x, y, rad) {
    let best = -1, bd = rad * rad;
    for (let i = 0; i < NN; i++) { const dx = pos[2 * i] - x, dy = pos[2 * i + 1] - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
    return best;
  }
  const chName = j => j === SP ? 'רווח' : j === SOF ? 'סוף פסוק' : CHARS[j];
  function neuronTip(i) {
    const l = (i / HID) | 0, n = i % HID;
    if (mind.mask[l][n] === 0) return `<b>נוירון ${l + 1}·${n + 1}</b><br>כבוי`;
    const arr = [];
    for (let j = 0; j < V; j++) if (cnt[j] >= 5) arr.push([sums[i * V + j] / cnt[j], j]);
    arr.sort((a, b) => b[0] - a[0]);
    const top = arr.filter(a => a[0] > 0.05).slice(0, 3).map(a => chName(a[1]));
    return `<b>נוירון ${l + 1}·${n + 1}</b><br>` + (top.length ? 'נדלק בעיקר כשהוא קורא: ' + top.join(' · ') : 'כמעט לא נדלק') + '<br><span style="color:var(--dim)">לחיצה: לעקוב אחריו ברצועת האותיות</span>';
  }
  let maskDirty = false, dragging = false;
  function cut(x, y) {
    let any = false;
    for (let i = 0; i < NN; i++) { const dx = pos[2 * i] - x, dy = pos[2 * i + 1] - y; if (dx * dx + dy * dy < 15 * 15) { const m = mind.mask[(i / HID) | 0]; if (m[i % HID] !== 0) { m[i % HID] = 0; any = true; } } }
    if (any) { baseDirty = true; maskDirty = true; }
  }
  function pushMasks() { if (!S.started) return; for (let l = 0; l < L; l++) Engine.setMask(l, mind.mask[l]); maskDirty = false; }
  const deadShare = () => { let d = 0; for (let l = 0; l < L; l++) for (let i = 0; i < HID; i++) if (mind.mask[l][i] === 0) d++; return d / NN; };
  brain.addEventListener('pointerdown', e => {
    if (S.scalpel) { dragging = true; brain.setPointerCapture(e.pointerId); cut(e.offsetX, e.offsetY); return; }
    const i = nearest(e.offsetX, e.offsetY, 12); S.sel = i === S.sel ? -1 : i; tape.forEach(t => t.s = 0);
  });
  brain.addEventListener('pointermove', e => {
    const tip = $('tip');
    if (S.scalpel) { tip.hidden = true; if (dragging) cut(e.offsetX, e.offsetY); return; }
    const i = nearest(e.offsetX, e.offsetY, 10);
    if (i < 0) { tip.hidden = true; return; }
    tip.innerHTML = neuronTip(i); tip.hidden = false;
    tip.style.left = Math.max(6, Math.min(BW - 226, e.offsetX - 110)) + 'px'; tip.style.top = Math.max(6, e.offsetY + 16) + 'px';
  });
  const endDrag = () => { if (dragging) { dragging = false; if (maskDirty) { pushMasks(); S.events.push({ t: age(), type: 'hurt' }); Snd.thud(); } } };
  brain.addEventListener('pointerup', endDrag); brain.addEventListener('pointercancel', endDrag);
  brain.addEventListener('pointerleave', () => { $('tip').hidden = true; });

  // ---------- chart ----------
  const chart = $('chart'), cctx = chart.getContext('2d');
  function drawChart() {
    const r = chart.getBoundingClientRect(); if (r.width < 10) return;
    const d = Math.min(2, window.devicePixelRatio || 1), W = r.width, Hh = r.height;
    if (chart.width !== Math.round(W * d)) { chart.width = Math.round(W * d); chart.height = Math.round(Hh * d); }
    const g = cctx; g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, Hh);
    const top = 22, bot = Hh - 6, tmax = Math.max(45, age() * 1.04), X = t => 6 + (t / tmax) * (W - 12);
    g.strokeStyle = 'rgba(43,49,87,0.9)'; g.lineWidth = 1;
    for (const f of [0, 0.5, 1]) { const y = bot - f * (bot - top); g.beginPath(); g.moveTo(6, y); g.lineTo(W - 6, y); g.stroke(); }
    for (const ev of S.events) {
      const x = X(ev.t);
      g.strokeStyle = ev.type === 'day' ? 'rgba(240,195,90,0.55)' : ev.type === 'hurt' ? 'rgba(255,111,125,0.9)' : ev.type === 'heal' ? 'rgba(111,211,194,0.9)' : 'rgba(239,230,208,0.6)';
      g.beginPath(); g.moveTo(x, top - 6); g.lineTo(x, bot); g.stroke();
      if (ev.type === 'day') { g.fillStyle = 'rgba(240,195,90,0.95)'; g.font = '700 12px "Frank Ruhl Libre", serif'; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('אבגדהוז'[ev.n - 1] + '׳', x + 3, top - 8); }
    }
    const line = (key, color, lo, hi) => {
      g.strokeStyle = color; g.lineWidth = 1.8; g.beginPath();
      S.series.forEach((p, i) => { const y = bot - Math.max(0, Math.min(1, (p[key] - lo) / (hi - lo))) * (bot - top); if (i) g.lineTo(X(p.t), y); else g.moveTo(X(p.t), y); });
      g.stroke();
    };
    if (S.series.length > 1) { line('bits', '#8ea6ff', 0, 5.3); line('rr', '#f0c35a', 0, 1); }
  }

  // ---------- loop ----------
  const age = () => S.started ? (clock() - S.t0) / 1000 : 0;
  const fmtAge = () => { const a = Math.floor(age()); return Math.floor(a / 60) + ':' + String(a % 60).padStart(2, '0'); };
  function updateStats() {
    $('sAge').textContent = fmtAge();
    $('sRead').textContent = (Engine.step * CFG.B * T).toLocaleString('en-US');
    $('sBits').textContent = (S.loss == null ? Math.log(V) / Math.LN2 : S.loss / Math.LN2).toFixed(2);
    $('sReal').textContent = Math.round(realRate() * 100) + '%';
    $('sNovel').textContent = S.novel;
  }
  let last = 0, acc = 0, lastUi = 0, lastChart = 0;
  let lastRaf = 0;
  function frame(ts) { requestAnimationFrame(frame); lastRaf = ts; tick(ts); }
  setInterval(() => { const ts = clock(); if (ts - lastRaf > 250) tick(ts); }, 100);
  function tick(ts) {
    const dt = Math.min(1, (ts - last) / 1000); last = ts;
    for (let i = 0; i < NN; i++) disp[i] *= 0.87;
    if (S.started && mind.w) {
      acc = Math.min(36, acc + dt * S.cps); let n = Math.floor(acc); acc -= n;
      while (n-- > 0) genChar();
    } else {
      for (let k = 0; k < 14; k++) disp[(Math.random() * NN) | 0] = 0.25 + Math.random() * 0.5;
    }
    drawBrain();
    if (S.started && ts - lastUi > 250) { updateStats(); lastUi = ts; }
    if (S.started && ts - lastChart > 500) {
      lastChart = ts;
      S.series.push({ t: age(), rr: realRate(), bits: S.loss == null ? 5.21 : S.loss / Math.LN2 });
      if (S.series.length > 2400) S.series = S.series.filter((_, i) => i % 2 === 0);
      drawChart();
    }
  }

  // ---------- controls ----------
  const tempName = t => t <= 0.35 ? 'מקובע' : t <= 0.6 ? 'זהיר' : t <= 0.95 ? 'צלול' : t <= 1.25 ? 'חופשי' : 'סלט מילים';
  $('temp').addEventListener('input', e => { S.temp = +e.target.value; $('tempOut').textContent = tempName(S.temp); });
  $('promptForm').addEventListener('submit', e => {
    e.preventDefault();
    const txt = $('prompt').value.replace(/[^א-ת' ]/g, ' ').replace(/ +/g, ' ').trim(); if (!txt || !S.started) return;
    S.forced = [SOF, SP, ...encodeText(txt), SP]; $('prompt').value = '';
  });
  $('bScalpel').addEventListener('click', e => { S.scalpel = !S.scalpel; e.currentTarget.setAttribute('aria-pressed', S.scalpel); $('brainbox').classList.toggle('scalpel', S.scalpel); });
  $('bStroke').addEventListener('click', () => {
    for (let l = 0; l < L; l++) for (let i = 0; i < HID; i++) if (mind.mask[l][i] !== 0 && Math.random() < 0.5) mind.mask[l][i] = 0;
    baseDirty = true; pushMasks(); S.events.push({ t: age(), type: 'hurt' }); Snd.thud();
    toast('פגיעה', Math.round(deadShare() * 100) + '% מהנוירונים כבויים. הוא ממשיך ללמוד עם מה שנשאר.');
  });
  $('bHeal').addEventListener('click', () => {
    for (let l = 0; l < L; l++) mind.mask[l].fill(1);
    baseDirty = true; pushMasks(); S.events.push({ t: age(), type: 'heal' }); Snd.chord([440, 554.37, 659.25], 0.05, 1.5, 60);
  });
  $('bLearn').addEventListener('click', e => { S.learning = !S.learning; e.currentTarget.setAttribute('aria-pressed', S.learning); e.currentTarget.textContent = S.learning ? 'לומד' : 'לא לומד'; });
  $('bSound').addEventListener('click', e => { Snd.on = !Snd.on; e.currentTarget.setAttribute('aria-pressed', Snd.on); });
  $('bCloseOk').addEventListener('click', () => { $('closing').hidden = true; });
  $('bFeed').addEventListener('click', () => { $('feedErr').textContent = ''; $('feed').hidden = false; });
  $('bFeedX').addEventListener('click', () => { $('feed').hidden = true; });
  function newDiet(vs, label) {
    buildLex(vs); Engine.setData(corpusData(vs)); S.win = []; S.lenWin = []; mark(label); S.events.push({ t: age(), type: 'feed' }); $('feed').hidden = true;
  }
  $('bFeedGo').addEventListener('click', () => {
    const vs = $('feedText').value.split(/[.!?;:\n\r]+/).map(s => s.replace(/[^א-ת' ]/g, ' ').replace(/ +/g, ' ').trim()).filter(s => s.length > 1);
    const n = vs.join(' ').length;
    if (n < 1500) { $('feedErr').textContent = 'יש כאן ' + n.toLocaleString('en-US') + ' אותיות בעברית. צריך לפחות 1,500.'; return; }
    newDiet(vs, 'חומר קריאה חדש');
  });
  $('bFeedTorah').addEventListener('click', () => newDiet(TORAH, 'חזרה לתורה'));
  $('bReborn').addEventListener('click', () => { if (S.started) birth(); });

  function birth() {
    S.gen++; Engine.build(); Engine.setData(corpusData(TORAH)); buildLex(TORAH);
    for (let l = 0; l < L; l++) { mind.mask[l].fill(1); }
    Object.assign(S, { day: 0, loss: null, win: [], lenWin: [], novel: 0, clean: 0, forced: [], gchars: 0, sel: -1, name: '', series: [], events: [], seen: new Set(), closingShown: false, learning: true });
    sums.fill(0); cnt.fill(0); tape = []; cur = null; curWord = ''; verse = [];
    el.scroll.textContent = ''; el.list.textContent = ''; $('named').hidden = true;
    $('dayName').textContent = DAYS[0][0]; $('dayDesc').textContent = DAYS[0][1];
    $('bLearn').setAttribute('aria-pressed', true); $('bLearn').textContent = 'לומד';
    baseDirty = true; mind.w = null;
    const gen = S.gen;
    Engine.snapshot().then(w => {
      if (gen !== S.gen) return;
      mind.reset(); mind.w = w; mind.step(SOF); mind.step(SP);
      S.t0 = clock(); S.started = true; trainLoop(gen);
    });
  }

  $('bGo').addEventListener('click', async () => {
    const b = $('bGo'), note = $('introNote'); b.disabled = true;
    if (typeof tf === 'undefined') { note.textContent = 'ספריית החישוב לא נטענה. צריך חיבור לאינטרנט בפתיחה הראשונה. רענן ונסה שוב.'; b.disabled = false; return; }
    Snd.init(); if (Snd.ctx && Snd.ctx.state === 'suspended') Snd.ctx.resume();
    note.textContent = 'מעיר את כרטיס המסך…';
    try {
      await Engine.init();
      if (Engine.backend !== 'webgl') { CFG.B = 8; }
      birth();
      $('intro').hidden = true;
      Snd.chord([146.83, 220, 293.66], 0.07, 4, 120);
    } catch (e) { console.error(e); note.textContent = 'משהו השתבש בהפעלה: ' + e.message; b.disabled = false; }
  });

  new ResizeObserver(() => { layout(); }).observe(brain);
  layout(); requestAnimationFrame(frame);
  window.__bereshit = { S, mind, Engine };
})();
