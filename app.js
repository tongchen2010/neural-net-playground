// Neural Net Playground — a small MLP implemented from scratch (forward pass,
// backpropagation, SGD) trained live in the browser on 2-D toy datasets, with
// the decision boundary drawn to a canvas as it learns. No ML library.

const REPO_URL = "https://github.com/tongchen2010/neural-net-playground";

// ---------- the network (verified by /tmp/test_net.js) ----------
const ACT = {
  tanh:    { f: (x) => Math.tanh(x),           df: (y) => 1 - y * y },
  relu:    { f: (x) => (x > 0 ? x : 0),        df: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (y) => y * (1 - y) },
};

function makeNet(sizes, act) {
  const layers = [];
  for (let l = 1; l < sizes.length; l++) {
    const inN = sizes[l - 1], outN = sizes[l];
    const scale = Math.sqrt(2 / (inN + outN));
    const W = Array.from({ length: outN }, () =>
      Array.from({ length: inN }, () => (Math.random() * 2 - 1) * scale));
    const b = Array.from({ length: outN }, () => 0);
    layers.push({ W, b });
  }
  return { layers, act };
}

function forwardActs(net, x) {
  const acts = [x];
  let a = x;
  const L = net.layers.length;
  for (let l = 0; l < L; l++) {
    const { W, b } = net.layers[l];
    const fn = l === L - 1 ? ACT.sigmoid : ACT[net.act];
    a = W.map((row, i) => fn.f(row.reduce((s, w, j) => s + w * a[j], 0) + b[i]));
    acts.push(a);
  }
  return acts;
}

const predict = (net, x) => forwardActs(net, x)[net.layers.length][0];

function trainStep(net, X, Y, lr) {
  const L = net.layers.length;
  const gW = net.layers.map((ly) => ly.W.map((r) => r.map(() => 0)));
  const gb = net.layers.map((ly) => ly.b.map(() => 0));
  let loss = 0, correct = 0;
  for (let n = 0; n < X.length; n++) {
    const acts = forwardActs(net, X[n]);
    const out = acts[L][0], y = Y[n], eps = 1e-7;
    loss += -(y * Math.log(out + eps) + (1 - y) * Math.log(1 - out + eps));
    if ((out >= 0.5 ? 1 : 0) === y) correct++;
    let delta = [out - y];
    for (let l = L - 1; l >= 0; l--) {
      const W = net.layers[l].W, aPrev = acts[l];
      for (let i = 0; i < W.length; i++) {
        gb[l][i] += delta[i];
        for (let j = 0; j < W[i].length; j++) gW[l][i][j] += delta[i] * aPrev[j];
      }
      if (l > 0) {
        const df = ACT[net.act].df, dPrev = new Array(aPrev.length).fill(0);
        for (let j = 0; j < aPrev.length; j++) {
          let s = 0;
          for (let i = 0; i < W.length; i++) s += delta[i] * W[i][j];
          dPrev[j] = s * df(aPrev[j]);
        }
        delta = dPrev;
      }
    }
  }
  const m = X.length;
  for (let l = 0; l < L; l++) {
    const ly = net.layers[l];
    for (let i = 0; i < ly.W.length; i++) {
      ly.b[i] -= (lr * gb[l][i]) / m;
      for (let j = 0; j < ly.W[i].length; j++) ly.W[i][j] -= (lr * gW[l][i][j]) / m;
    }
  }
  return { loss: loss / m, acc: correct / m };
}

// ---------- datasets in [-1, 1] ----------
const rnd = (a, b) => a + Math.random() * (b - a);
const DATASETS = {
  gauss(n = 220) { const X = [], Y = []; for (let i = 0; i < n; i++) { const c = i % 2, m = c ? 0.45 : -0.45; X.push([m + rnd(-0.22, 0.22), m + rnd(-0.22, 0.22)]); Y.push(c); } return { X, Y }; },
  circle(n = 240) { const X = [], Y = []; for (let i = 0; i < n; i++) { const c = i % 2, r = c ? rnd(0, 0.4) : rnd(0.6, 0.95), t = rnd(0, 2 * Math.PI); X.push([r * Math.cos(t), r * Math.sin(t)]); Y.push(c); } return { X, Y }; },
  xor(n = 240) { const X = [], Y = []; for (let i = 0; i < n; i++) { const x = rnd(-1, 1), y = rnd(-1, 1); X.push([x, y]); Y.push((x > 0) !== (y > 0) ? 1 : 0); } return { X, Y }; },
  moons(n = 240) { const X = [], Y = []; for (let i = 0; i < n; i++) { const c = i % 2, t = rnd(0, Math.PI), nx = rnd(-0.06, 0.06), ny = rnd(-0.06, 0.06); if (c) X.push([Math.cos(t) - 0.4 + nx, -Math.sin(t) + 0.18 + ny]); else X.push([Math.cos(t) + 0.4 + nx, Math.sin(t) - 0.18 + ny]); Y.push(c); } return { X, Y }; },
  spiral(n = 240) { const X = [], Y = []; const per = n / 2; for (let c = 0; c < 2; c++) for (let i = 0; i < per; i++) { const r = i / per * 0.95, t = c * Math.PI + i / per * 3.2 + rnd(-0.18, 0.18); X.push([r * Math.cos(t), r * Math.sin(t)]); Y.push(c); } return { X, Y }; },
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const board = $("board"), lossCurve = $("loss-curve");
const dsSel = $("dataset"), archSel = $("arch"), actSel = $("act"), lr = $("lr"), lrVal = $("lr-val");
const playBtn = $("play-btn"), stepBtn = $("step-btn"), resetBtn = $("reset-btn");
const epochEl = $("epoch"), lossEl = $("loss"), accEl = $("acc");
$("repo-link").href = REPO_URL;

const ctx = board.getContext("2d");
const lctx = lossCurve.getContext("2d");
const SIZE = 380, DOMAIN = 1.12, RES = 58;
let net, data, playing = false, epoch = 0, lossHist = [], raf = null;

function dpr() { return window.devicePixelRatio || 1; }
function setupCanvas() {
  const r = dpr();
  board.width = SIZE * r; board.height = SIZE * r;
  ctx.setTransform(r, 0, 0, r, 0, 0);
  lossCurve.width = lossCurve.clientWidth * r; lossCurve.height = 80 * r;
  lctx.setTransform(r, 0, 0, r, 0, 0);
}
const toPx = (v) => ((v / DOMAIN) * 0.5 + 0.5) * SIZE;

function archSizes() {
  const hidden = archSel.value ? archSel.value.split(",").map(Number) : [];
  return [2, ...hidden, 1];
}

function rebuild() {
  data = DATASETS[dsSel.value]();
  net = makeNet(archSizes(), actSel.value);
  epoch = 0; lossHist = [];
  epochEl.textContent = "0"; lossEl.textContent = "—"; accEl.textContent = "—";
  draw(); drawLoss();
}

function reinit() {
  net = makeNet(archSizes(), actSel.value);
  epoch = 0; lossHist = [];
  epochEl.textContent = "0"; lossEl.textContent = "—"; accEl.textContent = "—";
  draw(); drawLoss();
}

function colorFor(p) {
  const o = [232, 131, 58], b = [90, 156, 255];
  const r = Math.round(o[0] + (b[0] - o[0]) * p);
  const g = Math.round(o[1] + (b[1] - o[1]) * p);
  const bl = Math.round(o[2] + (b[2] - o[2]) * p);
  return `rgba(${r},${g},${bl},0.5)`;
}

function draw() {
  // decision-boundary heatmap
  const cell = SIZE / RES;
  for (let gx = 0; gx < RES; gx++) {
    for (let gy = 0; gy < RES; gy++) {
      const x = ((gx + 0.5) / RES * 2 - 1) * DOMAIN;
      const y = -((gy + 0.5) / RES * 2 - 1) * DOMAIN;
      ctx.fillStyle = colorFor(predict(net, [x, y]));
      ctx.fillRect(gx * cell, gy * cell, cell + 1, cell + 1);
    }
  }
  // data points
  for (let i = 0; i < data.X.length; i++) {
    const [x, y] = data.X[i];
    ctx.beginPath();
    ctx.arc(toPx(x), toPx(-y), 4, 0, 2 * Math.PI);
    ctx.fillStyle = data.Y[i] ? "#5a9cff" : "#e8833a";
    ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.stroke();
  }
}

function drawLoss() {
  const w = lossCurve.clientWidth, h = 80;
  lctx.clearRect(0, 0, w, h);
  if (lossHist.length < 2) return;
  const show = lossHist.slice(-200);
  const max = Math.max(...show), min = Math.min(...show);
  const span = max - min || 1;
  lctx.beginPath();
  show.forEach((v, i) => {
    const px = (i / (show.length - 1)) * w;
    const py = h - ((v - min) / span) * (h - 10) - 5;
    i ? lctx.lineTo(px, py) : lctx.moveTo(px, py);
  });
  lctx.strokeStyle = "#6ea8fe"; lctx.lineWidth = 2; lctx.stroke();
}

function tick(steps) {
  let m = null;
  for (let s = 0; s < steps; s++) { m = trainStep(net, data.X, data.Y, parseFloat(lr.value)); epoch++; lossHist.push(m.loss); }
  epochEl.textContent = String(epoch);
  lossEl.textContent = m.loss.toFixed(3);
  accEl.textContent = (m.acc * 100).toFixed(1) + "%";
  draw(); drawLoss();
}

function loop() {
  if (!playing) return;
  tick(3);
  raf = requestAnimationFrame(loop);
}

function play() {
  playing = !playing;
  playBtn.textContent = playing ? "⏸ Pause" : "▶ Train";
  if (playing) loop(); else if (raf) cancelAnimationFrame(raf);
}

// ---------- events ----------
dsSel.addEventListener("change", () => { stop(); rebuild(); });
archSel.addEventListener("change", () => { stop(); reinit(); });
actSel.addEventListener("change", () => { stop(); reinit(); });
lr.addEventListener("input", () => (lrVal.textContent = lr.value));
playBtn.addEventListener("click", play);
stepBtn.addEventListener("click", () => { if (!playing) tick(5); });
resetBtn.addEventListener("click", () => { stop(); reinit(); });
window.addEventListener("resize", () => { setupCanvas(); draw(); drawLoss(); });

function stop() {
  playing = false; playBtn.textContent = "▶ Train";
  if (raf) cancelAnimationFrame(raf);
}

// ---------- boot ----------
setupCanvas();
rebuild();
