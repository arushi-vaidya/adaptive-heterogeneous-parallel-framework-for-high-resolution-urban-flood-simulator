'use strict';
/* Adaptive Heterogeneous Flood Framework — UI logic.
   All data comes from the local server (scripts/ui.py), which runs the C++ solver. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const setText = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
const num = v => (v === null || v === undefined || v === '' || !Number.isFinite(Number(v))) ? null : Number(v);
const clamp01 = v => v < 0 ? 0 : v > 1 ? 1 : v;
const WET = 0.05; // m: flooded-cell threshold used for extent metrics
const BACKEND_META = {
  serial: {label: 'Serial', sub: 'CPU baseline', icon: 'i-timer', color: '#9333ea'},
  openmp: {label: 'OpenMP', sub: 'Shared memory', icon: 'i-chip', color: '#ea580c'},
  mpi: {label: 'MPI', sub: 'Distributed', icon: 'i-net', color: '#2563eb'},
  cuda: {label: 'CUDA', sub: 'NVIDIA GPU', icon: 'i-gpu', color: '#16a34a'},
  openacc: {label: 'OpenACC', sub: 'Directive GPU', icon: 'i-doc', color: '#0891b2'},
};
// Defaults mirror src/simulation/Scenarios.cpp (infiltration in mm/h).
const SCENARIO_DEFAULTS = {
  'urban-district': {rows: 128, cols: 128, duration: 120, manning: .035, infiltration: 7.2, boundary: 'outflow', storm: '120 mm/h for 60 s, then 40 mm/h',
    params: [{field: 'surge_height', label: 'River surge height', min: 0, max: 6, step: .1, value: 3.4, unit: 'm'}]},
  'dam-breach': {rows: 96, cols: 96, duration: 60, manning: .04, infiltration: 0, boundary: 'outflow', storm: 'no rain',
    params: [{field: 'reservoir_depth', label: 'Reservoir level', min: .5, max: 6, step: .1, value: 3, unit: 'm'},
             {field: 'breach_width', label: 'Breach width', min: .05, max: 1, step: .05, value: .2, unit: '× dam', fmt: v => `${Math.round(v * 100)}% of dam`}]},
  'catchment': {rows: 96, cols: 96, duration: 900, manning: .05, infiltration: 5, boundary: 'outflow', storm: '200 mm/h for 600 s', params: []},
  'localized-refinement': {rows: 64, cols: 64, duration: .6, manning: .03, infiltration: 0, boundary: 'closed', storm: 'no rain',
    params: [{field: 'pond_depth', label: 'Initial pond depth', min: .1, max: 3, step: .05, value: .5, unit: 'm'}]},
  'dam-break': {rows: 96, cols: 96, duration: 4, manning: .03, infiltration: 0, boundary: 'closed', storm: 'no rain',
    params: [{field: 'reservoir_depth', label: 'Reservoir depth behind dam', min: .2, max: 5, step: .1, value: 1, unit: 'm'}]},
  'rain-drain': {rows: 96, cols: 96, duration: 60, manning: .03, infiltration: 3.6, boundary: 'outflow', storm: '60 mm/h for 30 s', params: []},
};
const rainWord = v => v === 0 ? 'none' : v < 10 ? 'light' : v < 50 ? 'moderate' : v < 100 ? 'heavy' : 'extreme';
function syncParamOutputs() {
  const d = SCENARIO_DEFAULTS[$('#scenario').value];
  (d?.params || []).forEach(p => { const el = document.getElementById(`sp-${p.field}`); if (el) setText(`sp-${p.field}-out`, p.fmt ? p.fmt(Number(el.value)) : `${Number(el.value).toFixed(p.step < .1 ? 2 : 1)} ${p.unit}`); });
  const ri = Number($('#rainIntensity').value), rd = Number($('#rainDuration').value);
  setText('rainIntensityOut', `${ri} mm/h · ${rainWord(ri)}`);
  setText('rainDurationOut', `${rd} s · ${(ri * rd / 3600).toFixed(1)} mm total`);
  setText('manningOut', Number($('#manning').value).toFixed(3));
  setText('infiltrationOut', `${Number($('#infiltration').value).toFixed(1)} mm/h`);
  $('#stormFields').hidden = !$('#customStorm').checked;
  setText('stormNote', $('#customStorm').checked ? 'Constant-intensity storm from t = 0.' : `Scenario storm: ${d?.storm || '—'}.`);
}
function applyParamDefaults() {
  const d = SCENARIO_DEFAULTS[$('#scenario').value]; if (!d) return;
  $('#manning').value = d.manning; $('#infiltration').value = d.infiltration; $('#boundary').value = d.boundary; $('#customStorm').checked = false;
  $('#scenarioParams').innerHTML = (d.params || []).map(p => `<div class="range"><div class="rl"><label for="sp-${p.field}">${p.label}</label><output id="sp-${p.field}-out"></output></div>
    <input type="range" id="sp-${p.field}" name="${p.field}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}"></div>`).join('');
  $$('#scenarioParams input').forEach(el => el.addEventListener('input', syncParamOutputs));
  syncParamOutputs();
}
['rainIntensity', 'rainDuration', 'manning', 'infiltration', 'customStorm'].forEach(id => document.getElementById(id).addEventListener('input', syncParamOutputs));
$('#resetParams').addEventListener('click', applyParamDefaults);
const fmtSec = s => s === null ? '—' : s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(1)} ms` : `${(s * 1e6).toFixed(0)} µs`;
const fmtSci = v => v === null ? '—' : v === 0 ? '0' : Math.abs(v) >= 1e-2 ? v.toFixed(3) : v.toExponential(1);
const fmtBig = v => v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : `${Math.round(v)}`;

/* ---------------------------------------------------------------- data */
function parseCsvGrid(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const rows = lines.length, cols = rows ? lines[0].split(',').length : 0;
  const data = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const parts = lines[r].split(',');
    for (let c = 0; c < cols; c++) { const v = Number(parts[c]); data[r * cols + c] = Number.isFinite(v) ? v : 0; }
  }
  return {rows, cols, data};
}
async function fetchGrid(url) {
  const res = await fetch(url, {cache: 'no-store'});
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return parseCsvGrid(await res.text());
}
const optionalGrid = url => fetchGrid(url).catch(() => null);
async function getJson(url, opts) {
  const res = await fetch(url, {cache: 'no-store', ...opts});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.error || `HTTP ${res.status}`), {data});
  return data;
}

/* ------------------------------------------------------- terrain analysis */
// Classify the bed into buildings / ground and compute a hillshade so maps
// read like the urban GIS rasters in the slides.
function analyseTerrain(t) {
  const {rows, cols, data} = t, n = rows * cols;
  // local minimum over a (2R+1)^2 window via separable min filters
  const R = Math.max(2, Math.round(Math.min(rows, cols) / 40));
  const tmp = new Float32Array(n), localMin = new Float32Array(n);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    let m = Infinity; for (let k = -R; k <= R; k++) { const cc = Math.min(cols - 1, Math.max(0, c + k)); m = Math.min(m, data[r * cols + cc]); }
    tmp[r * cols + c] = m;
  }
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    let m = Infinity; for (let k = -R; k <= R; k++) { const rr = Math.min(rows - 1, Math.max(0, r + k)); m = Math.min(m, tmp[rr * cols + c]); }
    localMin[r * cols + c] = m;
  }
  const building = new Uint8Array(n), height = new Float32Array(n);
  let zmin = Infinity, zmax = -Infinity, gmin = Infinity, gmax = -Infinity, bmax = 0;
  for (let i = 0; i < n; i++) {
    const h = data[i] - localMin[i]; zmin = Math.min(zmin, data[i]); zmax = Math.max(zmax, data[i]);
    if (h > 2.5) { building[i] = 1; height[i] = h; bmax = Math.max(bmax, h); }
    else { gmin = Math.min(gmin, data[i]); gmax = Math.max(gmax, data[i]); }
  }
  // ground surface with buildings removed, for shading and hypsometric tint
  const ground = new Float32Array(n);
  for (let i = 0; i < n; i++) ground[i] = building[i] ? localMin[i] : data[i];
  const shade = new Float32Array(n);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const z = (rr, cc) => ground[Math.min(rows - 1, Math.max(0, rr)) * cols + Math.min(cols - 1, Math.max(0, cc))];
    const dx = (z(r, c + 1) - z(r, c - 1)) / 2, dy = (z(r + 1, c) - z(r - 1, c)) / 2;
    shade[r * cols + c] = Math.max(-1, Math.min(1, (-dx - dy) * 1.6));
  }
  // building shadows cast to the south-east
  const shadow = new Float32Array(n);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (building[r * cols + c]) continue;
    for (let k = 1; k <= 2; k++) {
      const rr = r - k, cc = c - k;
      if (rr >= 0 && cc >= 0 && building[rr * cols + cc]) { shadow[r * cols + c] = k === 1 ? .28 : .16; break; }
    }
  }
  const hasRelief = zmax - zmin > 1e-6;
  return {rows, cols, bed: data, building, height, ground, shade, shadow, zmin, zmax, gmin, gmax, bmax, hasRelief,
          hasBuildings: bmax > 0};
}

/* ------------------------------------------------------------- palettes */
const WATER = [[0, [214, 232, 252]], [.18, [147, 197, 253]], [.4, [59, 130, 246]], [.7, [29, 78, 216]], [1, [23, 37, 84]]];
const HEAT = [[0, [30, 64, 175]], [.25, [6, 182, 212]], [.5, [250, 204, 21]], [.75, [249, 115, 22]], [1, [220, 38, 38]]];
const HYPSO = [[0, [74, 160, 98]], [.35, [134, 190, 96]], [.65, [212, 196, 120]], [1, [186, 140, 96]]];
function ramp(stops, t) {
  t = clamp01(t);
  for (let i = 1; i < stops.length; i++) if (t <= stops[i][0]) {
    const [t0, a] = stops[i - 1], [t1, b] = stops[i], u = (t - t0) / (t1 - t0 || 1);
    return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  }
  return stops[stops.length - 1][1];
}
const cssRamp = (stops, dir = '90deg') => `linear-gradient(${dir},${[0, .2, .4, .6, .8, 1].map(t => { const c = ramp(stops, t); return `rgb(${c.map(Math.round)})`; }).join(',')})`;
const WATER_REAL = [[0, [96, 168, 196]], [.5, [57, 114, 153]], [1, [18, 60, 110]]];
const waterT = (d, scale) => Math.pow(clamp01(d / scale), .6);
function niceDepthScale(max) { for (const s of [.1, .25, .5, 1, 2, 2.5, 5, 10]) if (max <= s) return s; return Math.ceil(max); }

/* --------------------------------------------------------- map renderer */
// Paints a raster (one pixel per cell) into an offscreen canvas, then scales it.
function paintRaster(canvas, rows, cols, pixel) {
  const off = document.createElement('canvas'); off.width = cols; off.height = rows;
  const octx = off.getContext('2d'), img = octx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const p = pixel(r, c, r * cols + c), o = (r * cols + c) * 4;
    img.data[o] = p[0]; img.data[o + 1] = p[1]; img.data[o + 2] = p[2]; img.data[o + 3] = p[3] ?? 255;
  }
  octx.putImageData(img, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.save(); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = rows < canvas.height / 6 ? false : true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height); ctx.restore();
  return ctx;
}
function basePixel(terrain, i) {
  if (!terrain) return [236, 239, 243];
  if (terrain.building[i]) {
    const t = terrain.bmax ? terrain.height[i] / terrain.bmax : 0;
    const v = 238 - 34 * t; return [v, v + 2, v + 6];
  }
  let v = 226 + terrain.shade[i] * 18 - terrain.shadow[i] * 120;
  return [v - 3, v, v + 3];
}
function mixWater(base, d, scale) {
  if (d <= WET * .2) return base;
  const t = waterT(d, scale), w = ramp(WATER, t), a = d < WET ? .35 : .55 + .42 * t;
  return [base[0] * (1 - a) + w[0] * a, base[1] * (1 - a) + w[1] * a, base[2] * (1 - a) + w[2] * a];
}
function drawFlood(canvas, depth, terrain, scale) {
  return paintRaster(canvas, depth.rows, depth.cols, (r, c, i) => {
    const base = basePixel(terrain, i);
    if (terrain && terrain.building[i]) return base;
    return mixWater(base, depth.data[i], scale);
  });
}
function drawTerrain(canvas, terrain, depth) {
  const {rows, cols} = terrain || depth;
  return paintRaster(canvas, rows, cols, (r, c, i) => {
    if (!terrain || !terrain.hasRelief) return [226, 232, 240];
    if (terrain.building[i]) { const t = terrain.height[i] / (terrain.bmax || 1); const v = 245 - 70 * t; return [v, v, v + 4]; }
    const g = (terrain.ground[i] - terrain.gmin) / ((terrain.gmax - terrain.gmin) || 1);
    const h = ramp(HYPSO, g), s = 1 + terrain.shade[i] * .25 - terrain.shadow[i];
    return [h[0] * s, h[1] * s, h[2] * s];
  });
}
const LEVEL_FILL = [null, [124, 58, 237, .28], [234, 88, 12, .38], [220, 38, 38, .45]];
function drawMesh(canvas, depth, terrain, refine, scale) {
  const ctx = drawFlood(canvas, depth, terrain, scale);
  if (!refine) return ctx;
  const {rows, cols, data} = refine, W = canvas.width / cols, H = canvas.height / rows;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const L = Math.round(data[r * cols + c]); if (L <= 0) continue;
    const f = LEVEL_FILL[Math.min(L, 3)]; ctx.fillStyle = `rgba(${f[0]},${f[1]},${f[2]},${f[3]})`;
    ctx.fillRect(c * W, r * H, W + .5, H + .5);
  }
  ctx.lineWidth = 1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const L = Math.round(data[r * cols + c]); if (L <= 0) continue;
    const k = 2 ** L; if (W / k < 2.2) continue;
    ctx.strokeStyle = L >= 2 ? 'rgba(154,52,18,.45)' : 'rgba(91,33,182,.4)';
    ctx.beginPath();
    for (let s = 1; s < k; s++) { ctx.moveTo(c * W + s * W / k, r * H); ctx.lineTo(c * W + s * W / k, (r + 1) * H); ctx.moveTo(c * W, r * H + s * H / k); ctx.lineTo((c + 1) * W, r * H + s * H / k); }
    ctx.stroke();
  }
  // base-cell grid when it is coarse enough to be legible
  if (W >= 5) {
    ctx.strokeStyle = 'rgba(15,23,42,.14)'; ctx.beginPath();
    for (let c = 0; c <= cols; c++) { ctx.moveTo(c * W, 0); ctx.lineTo(c * W, canvas.height); }
    for (let r = 0; r <= rows; r++) { ctx.moveTo(0, r * H); ctx.lineTo(canvas.width, r * H); }
    ctx.stroke();
  }
  return ctx;
}
// Mirrors MPI_Dims_create + block partition used by src/mpi/MpiSolver.cpp.
function mpiDims(p) { if (p === 4) return [2, 2]; if (p === 2) return [2, 1]; return [p, 1]; }
function splits(n, parts) { const out = [0]; const base = Math.floor(n / parts), rem = n % parts; for (let i = 0; i < parts; i++) out.push(out[i] + base + (i < rem ? 1 : 0)); return out; }
const RANK_COLORS = ['#8b5cf6', '#3b82f6', '#22c55e', '#f97316', '#ec4899', '#14b8a6', '#eab308', '#64748b'];
function partitionFor(summary) {
  if (!summary) return null;
  if (summary.backend === 'mpi') { const [pr, pc] = mpiDims(Number(summary.processes) || 1); return {kind: 'MPI rank', pr, pc}; }
  if (summary.backend === 'openmp') return {kind: 'OpenMP thread', pr: Number(summary.threads) || 1, pc: 1};
  if (summary.backend === 'cuda' || summary.backend === 'openacc') return {kind: 'GPU', pr: 1, pc: 1, gpu: true};
  return {kind: 'Serial', pr: 1, pc: 1};
}
function drawPartition(canvas, depth, terrain, scale, part) {
  const ctx = drawFlood(canvas, depth, terrain, scale);
  if (!part) return ctx;
  const rs = splits(depth.rows, part.pr), cs = splits(depth.cols, part.pc);
  const sx = canvas.width / depth.cols, sy = canvas.height / depth.rows;
  let k = 0;
  for (let i = 0; i < part.pr; i++) for (let j = 0; j < part.pc; j++, k++) {
    const x = cs[j] * sx, y = rs[i] * sy, w = (cs[j + 1] - cs[j]) * sx, h = (rs[i + 1] - rs[i]) * sy;
    const col = RANK_COLORS[k % RANK_COLORS.length];
    ctx.fillStyle = col + '30'; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.setLineDash([10, 6]); ctx.strokeRect(x + 1.5, y + 1.5, w - 3, h - 3); ctx.setLineDash([]);
    if (part.pr * part.pc > 1) { // halo band
      ctx.fillStyle = col + '55';
      if (i > 0) ctx.fillRect(x, y, w, Math.max(2, sy));
      if (j > 0) ctx.fillRect(x, y, Math.max(2, sx), h);
    }
    const label = part.gpu ? 'GPU · one thread per cell' : part.pr * part.pc === 1 ? 'Single domain' : `${part.kind} ${k}`;
    ctx.font = '700 15px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(255,255,255,.92)'; ctx.beginPath(); ctx.roundRect(x + 12, y + 12, tw + 20, 28, 14); ctx.fill();
    ctx.fillStyle = col; ctx.fillText(label, x + 22, y + 31);
  }
  return ctx;
}

/* ------------------------------------------------------ realistic city renderer */
// Renders solver output as an aerial view: static city layer (land use,
// hillshade, rooftops, sun shadows) + animated water (depth absorption,
// free-surface lighting, ripples advected by the computed velocity, shoreline
// foam), flow tracers integrated through the solver's velocity field, and rain
// while the scenario hyetograph is active. All water geometry is the solver's.
const LU = {PAVE: 0, ROAD: 1, BUILDING: 2, GRASS: 3, RIVER: 4, OPEN: 5, FARM: 6, DAM: 7};
// Same layout as src/simulation/Scenarios.cpp, for runs that do not export landuse.csv (adaptive).
function defaultLandUse(scenario, rows, cols) {
  const lu = new Uint8Array(rows * cols);
  if (scenario === 'dam-break') { for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) lu[r * cols + c] = c < Math.floor(cols / 2) ? LU.RIVER : LU.FARM; return lu; }
  if (scenario === 'rain-drain') return lu.fill(LU.FARM);
  if (scenario === 'localized-refinement') {
    lu.fill(LU.GRASS); const r0 = Math.floor(rows / 3), r1 = Math.min(rows, r0 + Math.floor(rows / 5)), c0 = Math.floor(cols / 3), c1 = Math.min(cols, c0 + Math.floor(cols / 5));
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) lu[r * cols + c] = LU.RIVER; return lu;
  }
  return null;
}
const CROPS = [[122, 150, 74], [150, 160, 82], [186, 170, 100], [104, 136, 66], [160, 132, 92], [134, 158, 90]];
const RAIN_TABLE = { // mirrors src/simulation/Scenarios.cpp (time s, mm/h)
  'urban-district': [[0, 120], [60, 40], [150, 0]], 'rain-drain': [[0, 60], [30, 0]], 'flat-basin': [[0, 30]], 'catchment': [[0, 200], [600, 0]],
};
function rainAt(scenario, t, custom) { const tab = custom || RAIN_TABLE[scenario]; if (!tab) return 0; let v = 0; for (const [t0, i] of tab) if (t >= t0) v = i; return v; }
function hash2(a, b) { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177 | 0; return ((h ^ (h >>> 16)) >>> 0) / 4294967295; }
// Tileable wave texture (sum of integer-wavenumber sines) for open-water surfaces.
const WAVE_N = 128, WAVES = (() => {
  const a = new Float32Array(WAVE_N * WAVE_N), waves = [];
  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let k = 0; k < 9; k++) waves.push({kx: Math.round((rnd() - .5) * 14), ky: Math.round((rnd() - .5) * 14) || 1, p: rnd() * 6.283, a: .4 + rnd()});
  let m = 0;
  for (let y = 0; y < WAVE_N; y++) for (let x = 0; x < WAVE_N; x++) {
    let v = 0; for (const w of waves) v += w.a * Math.sin(6.283 * (w.kx * x + w.ky * y) / WAVE_N + w.p);
    a[y * WAVE_N + x] = v; m = Math.max(m, Math.abs(v));
  }
  for (let i = 0; i < a.length; i++) a[i] /= m; return a;
})();
const ROOFS = [[214, 214, 208], [196, 190, 180], [172, 100, 78], [128, 136, 148], [232, 230, 224], [160, 150, 136], [186, 120, 96], [150, 160, 166]];

class CityRenderer {
  constructor(canvas, {animate = true, particles = 1400} = {}) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.animate = animate; this.nParticles = particles;
    this.P = canvas.width; this.W2 = Math.round(this.P / 2);
    this.water = document.createElement('canvas'); this.water.width = this.water.height = this.W2;
    this.wctx = this.water.getContext('2d'); this.wimg = this.wctx.createImageData(this.W2, this.W2);
    this.trail = document.createElement('canvas'); this.trail.width = this.trail.height = this.P; this.tctx = this.trail.getContext('2d');
    this.playing = false; this.speed = 1; this.simT = 0; this.animT = 0; this.showWater = true; this.onTime = null;
    this.visible = true; this.last = 0; this.rain = [];
    if (animate) {
      new IntersectionObserver(es => { this.visible = es[0].isIntersecting; if (this.visible) this.kick(); }).observe(canvas);
    }
  }
  setData({rows, cols, cell, bed, landUse, frames, scenario, scale, rainfall}) {
    Object.assign(this, {rows, cols, cell: cell || 5, bed, scenario, scale: scale || 1, rainfall});
    landUse = landUse || defaultLandUse(scenario, rows, cols);
    const n = rows * cols;
    this.lu = new Uint8Array(n);
    this.height = new Float32Array(n);
    // land use: exported by the solver for the urban scenario; otherwise derived from the bed
    const terr = bed ? analyseTerrain({rows, cols, data: bed}) : null;
    for (let i = 0; i < n; i++) {
      if (landUse) this.lu[i] = landUse[i];
      else this.lu[i] = terr && terr.building[i] ? LU.BUILDING : LU.OPEN;
      if (terr && (this.lu[i] === LU.BUILDING || this.lu[i] === LU.DAM)) this.height[i] = Math.max(terr.height[i], 3);
    }
    this.ground = terr ? terr.ground : new Float32Array(n);
    this.frames = frames; this.times = frames.map(f => f.t);
    this.duration = this.times[this.times.length - 1] - this.times[0];
    this.simT = this.times[this.times.length - 1];
    this.cur = {h: new Float32Array(n), u: new Float32Array(n), v: new Float32Array(n), nx: new Float32Array(n), ny: new Float32Array(n), shore: new Float32Array(n)};
    this.buildBase(); this.blend(this.simT); this.seedParticles(); this.tctx.clearRect(0, 0, this.P, this.P);
    this.draw(); this.kick();
  }
  sampleIdx(u, v) { const c = Math.min(this.cols - 1, Math.max(0, Math.floor(u))), r = Math.min(this.rows - 1, Math.max(0, Math.floor(v))); return r * this.cols + c; }
  bilinear(a, u, v) {
    const {rows, cols} = this; u -= .5; v -= .5;
    const c0 = Math.max(0, Math.min(cols - 1, Math.floor(u))), r0 = Math.max(0, Math.min(rows - 1, Math.floor(v)));
    const c1 = Math.min(cols - 1, c0 + 1), r1 = Math.min(rows - 1, r0 + 1), fu = Math.min(1, Math.max(0, u - c0)), fv = Math.min(1, Math.max(0, v - r0));
    return (a[r0 * cols + c0] * (1 - fu) + a[r0 * cols + c1] * fu) * (1 - fv) + (a[r1 * cols + c0] * (1 - fu) + a[r1 * cols + c1] * fu) * fv;
  }
  buildBase() {
    const {P, rows, cols, lu, height, cell} = this, img = new ImageData(P, P), d = img.data;
    const sx = cols / P, sy = rows / P;
    let gmin = Infinity, gmax = -Infinity;
    for (let i = 0; i < lu.length; i++) if (lu[i] !== LU.BUILDING) { gmin = Math.min(gmin, this.ground[i]); gmax = Math.max(gmax, this.ground[i]); }
    const span = Math.max(1e-6, gmax - gmin);
    const isB = (r, c) => r >= 0 && c >= 0 && r < rows && c < cols && lu[r * cols + c] === LU.BUILDING;
    const maxShadow = 4.5; // cells
    for (let py = 0; py < P; py++) for (let px = 0; px < P; px++) {
      const u = (px + .5) * sx, v = (py + .5) * sy, c = Math.min(cols - 1, Math.floor(u)), r = Math.min(rows - 1, Math.floor(v)), i = r * cols + c;
      const cls = lu[i], noise = hash2(px, py) - .5, o = (py * P + px) * 4;
      let col;
      if (cls === LU.BUILDING) {
        // rooftop: colour per building (keyed on its height), parapet edge, rooftop units
        const key = Math.round(height[i] * 10), roof = ROOFS[key % ROOFS.length];
        const fu = u - c, fv = v - r;
        const edge = (!isB(r, c - 1) && fu < .22) || (!isB(r, c + 1) && fu > .78) || (!isB(r - 1, c) && fv < .22) || (!isB(r + 1, c) && fv > .78);
        const lit = (!isB(r - 1, c) && fv < .22) || (!isB(r, c - 1) && fu < .22);
        const k = edge ? (lit ? 1.12 : .78) : 1 + noise * .05;
        col = [roof[0] * k, roof[1] * k, roof[2] * k];
        if (!edge && hash2(c * 7 + key, r * 13) < .06 && Math.abs(fu - .5) < .22 && Math.abs(fv - .5) < .22) col = col.map(x => x * .82);
      } else {
        const g = (this.bilinear(this.ground, u, v) - gmin) / span;
        if (cls === LU.ROAD) {
          const kerb = [[0, -1], [0, 1], [-1, 0], [1, 0]].some(([dr, dc]) => { const rr = r + dr, cc = c + dc; return rr >= 0 && cc >= 0 && rr < rows && cc < cols && lu[rr * cols + cc] !== LU.ROAD && lu[rr * cols + cc] !== LU.RIVER && Math.abs((dr ? v - r : u - c) - (dr > 0 || dc > 0 ? 1 : 0)) < .14; });
          const base = 78 + noise * 10; col = kerb ? [168, 166, 160] : [base, base + 3, base + 8];
        } else if (cls === LU.GRASS) {
          const m = hash2(Math.floor(px / 3), Math.floor(py / 3)) * 18; col = [92 + m * .6 + noise * 10, 138 + m + noise * 12, 78 + noise * 8];
        } else if (cls === LU.RIVER) {
          col = [118 + noise * 8, 108 + noise * 8, 86 + noise * 6];
        } else if (cls === LU.DAM) {
          const crest = Math.abs((u - c) - .5) < .18; col = crest ? [214, 210, 200] : [178 + noise * 8, 176 + noise * 8, 168 + noise * 6];
        } else if (cls === LU.FARM) {
          // fields of ~10 cells with crop rows; hedgerows along field edges
          const F = Math.max(6, Math.round(cols / 10)), fc = Math.floor(u / F), fr = Math.floor(v / F), crop = CROPS[Math.floor(hash2(fc, fr) * CROPS.length)];
          const along = hash2(fr, fc) < .5 ? u : v, stripe = Math.sin(along * Math.PI * 2 * 1.6) * .06;
          const hedge = (u / F - fc) < .04 || (v / F - fr) < .04;
          col = hedge ? [70 + noise * 10, 98 + noise * 12, 58] : [crop[0] * (1 + stripe) + noise * 8, crop[1] * (1 + stripe) + noise * 8, crop[2] * (1 + stripe) + noise * 6];
        } else if (cls === LU.PAVE) {
          const tile = ((px >> 2) + (py >> 2)) & 1 ? 4 : 0; col = [200 + tile + noise * 8, 196 + tile + noise * 8, 188 + noise * 6];
        } else {
          col = [196 + noise * 10 - g * 20, 196 + noise * 10 - g * 6, 176 + noise * 8 - g * 22];
        }
        // hillshade of the ground surface
        const gx = this.bilinear(this.ground, u + .7, v) - this.bilinear(this.ground, u - .7, v), gy = this.bilinear(this.ground, u, v + .7) - this.bilinear(this.ground, u, v - .7);
        let shade = 1 + Math.max(-.25, Math.min(.25, (-gx - gy) * 1.4 / (cell / 5)));
        // sun from the north-west at ~45°: a building of height H shades H metres
        for (let k = .5; k <= maxShadow; k += .5) {
          const rr = Math.floor(v - k), cc = Math.floor(u - k);
          if (rr < 0 || cc < 0) break;
          if ((lu[rr * cols + cc] === LU.BUILDING || lu[rr * cols + cc] === LU.DAM) && height[rr * cols + cc] > k * cell) { shade *= .62; break; }
        }
        col = col.map(x => x * shade);
      }
      d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
    }
    // tree canopies in parks
    this.base = document.createElement('canvas'); this.base.width = this.base.height = P;
    const bctx = this.base.getContext('2d'); bctx.putImageData(img, 0, 0);
    const pxPerCell = P / cols;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = r * cols + c; if (lu[i] !== LU.GRASS || hash2(c, r * 3 + 1) > .22) continue;
      const x = (c + hash2(c, r)) * pxPerCell, y = (r + hash2(r, c)) * pxPerCell, rad = pxPerCell * (.7 + hash2(c * 3, r) * .9);
      bctx.fillStyle = 'rgba(20,40,20,.25)'; bctx.beginPath(); bctx.arc(x + rad * .45, y + rad * .45, rad, 0, 7); bctx.fill();
      const g = bctx.createRadialGradient(x - rad * .3, y - rad * .3, rad * .1, x, y, rad);
      g.addColorStop(0, '#6fa557'); g.addColorStop(1, '#2f5f2c'); bctx.fillStyle = g; bctx.beginPath(); bctx.arc(x, y, rad, 0, 7); bctx.fill();
    }
  }
  blend(t) {
    const {times, frames, cur, rows, cols, lu} = this;
    let k = 0; while (k < times.length - 2 && times[k + 1] < t) k++;
    const a = frames[k], b = frames[Math.min(k + 1, frames.length - 1)];
    const f = b === a ? 0 : Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t || 1)));
    this.frameIndex = f > .5 ? k + 1 : k;
    const n = rows * cols, cell = this.cell;
    for (let i = 0; i < n; i++) {
      const h = lu[i] === LU.BUILDING ? 0 : a.h[i] * (1 - f) + b.h[i] * f;
      cur.h[i] = h;
      if (h > .02 && a.hu) { cur.u[i] = (a.hu[i] * (1 - f) + b.hu[i] * f) / h / cell; cur.v[i] = (a.hv[i] * (1 - f) + b.hv[i] * f) / h / cell; }
      else { cur.u[i] = 0; cur.v[i] = 0; }
    }
    // free-surface normals from eta = h + z
    const bed = this.bed;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const i = r * cols + c; if (cur.h[i] <= .01) { cur.nx[i] = cur.ny[i] = 0; continue; }
      const eta = j => (cur.h[j] > .01 ? cur.h[j] + (bed ? bed[j] : 0) : cur.h[i] + (bed ? bed[i] : 0));
      const L = c > 0 ? r * cols + c - 1 : i, R = c < cols - 1 ? r * cols + c + 1 : i, U = r > 0 ? i - cols : i, D = r < rows - 1 ? i + cols : i;
      cur.nx[i] = (eta(R) - eta(L)) * .5; cur.ny[i] = (eta(D) - eta(U)) * .5;
      const dry = j => j !== i && cur.h[j] <= .012 && lu[j] !== LU.BUILDING;
      cur.shore[i] = dry(L) || dry(R) || dry(U) || dry(D) ? 1 : 0;
    }
    this.simT = t;
  }
  renderWater() {
    const {W2, cur, rows, cols, lu} = this, d = this.wimg.data, sx = cols / W2, sy = rows / W2, at = this.animT, scale = this.scale;
    for (let py = 0; py < W2; py++) for (let px = 0; px < W2; px++) {
      const o = (py * W2 + px) * 4, u = (px + .5) * sx, v = (py + .5) * sy;
      if (lu[this.sampleIdx(u, v)] === LU.BUILDING) { d[o + 3] = 0; continue; }
      const h = this.bilinear(cur.h, u, v);
      if (h < .012) { d[o + 3] = 0; continue; }
      const vu = this.bilinear(cur.u, u, v), vv = this.bilinear(cur.v, u, v), spd = Math.hypot(vu, vv);
      const t = Math.pow(Math.min(1, h / scale), .55);
      // water colour: turquoise in the shallows, deep blue where deep
      let R = 96 - 78 * t, G = 168 - 108 * t, B = 196 - 86 * t;
      const nx = this.bilinear(cur.nx, u, v) * 3, ny = this.bilinear(cur.ny, u, v) * 3;
      const ph1 = (u - vu * at * 2.5) * 5.1 + (v - vv * at * 2.5) * 1.9, ph2 = (u - vu * at * 2.5) * -2.3 + (v - vv * at * 2.5) * 6.7;
      const ripple = (Math.sin(ph1 + at * 1.3) + .6 * Math.sin(ph2 - at * 1.7)) * (.05 + Math.min(.25, spd * .35));
      const ox = (at * 6) | 0, oy = (at * 4) | 0;
      const wave = WAVES[((py + oy) & (WAVE_N - 1)) * WAVE_N + ((px + ox) & (WAVE_N - 1))] * .5 + WAVES[((py * 2 - oy) & (WAVE_N - 1)) * WAVE_N + ((px * 2 + ox) & (WAVE_N - 1))] * .3;
      const light = .5 - (nx + ny) + ripple + wave * (.06 + .1 * t);
      const k = .82 + .36 * Math.min(1, Math.max(0, light));
      R *= k; G *= k; B *= k;
      const spec = Math.max(0, light - .62) * 260; R += spec; G += spec; B += spec;
      let alpha = Math.min(.94, .3 + (1 - Math.exp(-h / .35)) * .7);
      // foam only along the wet/dry edge; thin sheet flow elsewhere is a translucent sheen
      const shore = this.bilinear(cur.shore, u, v);
      if (shore > .05) { const foam = shore * Math.max(0, 1 - h / .25) * .32; R += (255 - R) * foam; G += (255 - G) * foam; B += (255 - B) * foam; }
      if (h < .08) alpha *= .55 + h / .08 * .45;
      d[o] = R; d[o + 1] = G; d[o + 2] = B; d[o + 3] = alpha * 255;
    }
    this.wctx.putImageData(this.wimg, 0, 0);
  }
  seedParticles() {
    this.wet = []; for (let i = 0; i < this.cur.h.length; i++) if (this.cur.h[i] > .08) this.wet.push(i);
    this.parts = Array.from({length: this.nParticles}, () => this.spawn({}));
  }
  spawn(p) {
    if (!this.wet.length) { p.dead = true; return p; }
    const i = this.wet[(Math.random() * this.wet.length) | 0];
    p.x = i % this.cols + Math.random(); p.y = Math.floor(i / this.cols) + Math.random(); p.age = Math.random() * 80; p.dead = false; return p;
  }
  stepParticles(dtSim) {
    const ctx = this.tctx, s = this.P / this.cols;
    ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,.09)'; ctx.fillRect(0, 0, this.P, this.P);
    ctx.globalCompositeOperation = 'source-over'; ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.1; ctx.beginPath();
    for (const p of this.parts) {
      if (p.dead) { if (Math.random() < .02) this.spawn(p); continue; }
      const h = this.bilinear(this.cur.h, p.x, p.y);
      const vu = this.bilinear(this.cur.u, p.x, p.y), vv = this.bilinear(this.cur.v, p.x, p.y);
      if (h < .03 || ++p.age > 160 || Math.hypot(vu, vv) < .002) { this.spawn(p); continue; }
      const nx = p.x + vu * dtSim, ny = p.y + vv * dtSim;
      if (nx < 0 || ny < 0 || nx >= this.cols || ny >= this.rows || this.lu[this.sampleIdx(nx, ny)] === LU.BUILDING) { this.spawn(p); continue; }
      ctx.moveTo(p.x * s, p.y * s); ctx.lineTo(nx * s, ny * s); p.x = nx; p.y = ny;
    }
    ctx.stroke();
  }
  drawRain(dtReal) {
    const mmh = rainAt(this.scenario, this.simT, this.rainfall); if (mmh <= 0) { this.rain.length = 0; return; }
    const want = Math.round(mmh * 1.6), ctx = this.ctx, P = this.P;
    while (this.rain.length < want) this.rain.push({x: Math.random() * P, y: Math.random() * P, l: 8 + Math.random() * 10});
    this.rain.length = Math.min(this.rain.length, want);
    ctx.strokeStyle = 'rgba(225,235,245,.38)'; ctx.lineWidth = 1; ctx.beginPath();
    for (const d of this.rain) { d.y += dtReal * 900; d.x += dtReal * 180; if (d.y > P) { d.y -= P; d.x = Math.random() * P; } if (d.x > P) d.x -= P; ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - d.l * .2, d.y - d.l); }
    ctx.stroke();
  }
  draw(dtReal = 0) {
    const ctx = this.ctx; ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.base, 0, 0, this.canvas.width, this.canvas.height);
    if (this.showWater) {
      this.renderWater(); ctx.drawImage(this.water, 0, 0, this.canvas.width, this.canvas.height);
      if (this.animate) ctx.drawImage(this.trail, 0, 0);
    }
    if (this.animate && this.showWater) this.drawRain(dtReal);
    if (this.overlay) this.overlay(ctx);
  }
  kick() { if (this.animate && !this.raf && this.frames) this.raf = requestAnimationFrame(ts => this.tick(ts)); }
  tick(ts) {
    this.raf = null; if (!this.visible || !this.frames) return;
    const dt = Math.min(.05, this.last ? (ts - this.last) / 1000 : 0); this.last = ts; this.animT += dt;
    const simRate = Math.max(1, this.duration / 14) * this.speed; // whole run ≈ 14 s at 1×
    if (this.playing && this.frames.length > 1) {
      let t = this.simT + dt * simRate;
      const end = this.times[this.times.length - 1];
      if (t >= end) { t = end; this.playing = false; this.onPlayState?.(false); }
      this.blend(t); if (this.wet.length < 20 || Math.random() < .03) this.wet = this.wetCells();
      this.onTime?.(t);
    }
    if (this.showWater) this.stepParticles(dt * Math.max(1, this.duration / 14) * 1.5);
    this.draw(dt); this.kick();
  }
  wetCells() { const w = []; for (let i = 0; i < this.cur.h.length; i++) if (this.cur.h[i] > .08) w.push(i); return w; }
  seek(t) { this.blend(t); this.wet = this.wetCells(); this.tctx.clearRect(0, 0, this.P, this.P); this.onTime?.(t); this.draw(); }
  play() { if (this.simT >= this.times[this.times.length - 1] - 1e-9) this.seek(this.times[0]); this.playing = true; this.onPlayState?.(true); this.kick(); }
  pause() { this.playing = false; this.onPlayState?.(false); }
}
async function loadFrames(prefix) {
  try {
    const meta = await getJson(`${prefix}frames.json`);
    const res = await fetch(`${prefix}frames.bin`, {cache: 'no-store'}); if (!res.ok) return null;
    const buf = new Float32Array(await res.arrayBuffer()), n = meta.rows * meta.cols;
    if (buf.length !== meta.times.length * 3 * n) return null;
    return {meta, frames: meta.times.map((t, k) => ({t, h: buf.subarray((3 * k) * n, (3 * k + 1) * n), hu: buf.subarray((3 * k + 1) * n, (3 * k + 2) * n), hv: buf.subarray((3 * k + 2) * n, (3 * k + 3) * n)}))};
  } catch { return null; }
}
async function loadLandUse(url) { const g = await optionalGrid(url); return g ? Uint8Array.from(g.data) : null; }

/* ------------------------------------------------------------ hero stack */
function heroLayer(canvas, kind, depth, terrain, speed) {
  const scale = niceDepthScale(maxOf(depth.data));
  if (kind === 'flood') return drawFlood(canvas, depth, terrain, scale);
  if (kind === 'elevation') return drawTerrain(canvas, terrain, depth);
  if (kind === 'urban') return paintRaster(canvas, terrain.rows, terrain.cols, (r, c, i) => {
    if (terrain.building[i]) { const t = terrain.height[i] / terrain.bmax; const v = 250 - 80 * t; return [v, v, v + 3]; }
    const d = depth.data[i]; if (d > .4 && terrain.shade[i] > -2) { const w = ramp(WATER, .45); return [w[0] * .5 + 60, w[1] * .5 + 70, w[2] * .5 + 90]; }
    const v = 92 + terrain.shade[i] * 20 - terrain.shadow[i] * 80; return [v, v + 4, v + 10];
  });
  if (kind === 'activity') {
    let vmax = 0; for (const v of speed) vmax = Math.max(vmax, v);
    return paintRaster(canvas, depth.rows, depth.cols, (r, c, i) => {
      const t = Math.pow(clamp01(speed[i] / (vmax * .55 || 1)), .55);
      return ramp(HEAT, t);
    });
  }
}
function maxOf(a) { let m = 0; for (const v of a) if (v > m) m = v; return m; }
async function loadShowcase(attempt = 0) {
  const [depth, bed, hu, hv] = await Promise.all(['depth', 'terrain', 'hu', 'hv'].map(n => optionalGrid(`/showcase/${n}.csv`)));
  if (!depth || !bed) { // the server computes it on first start
    if (attempt < 30) setTimeout(() => loadShowcase(attempt + 1), 1500);
    else setText('stackCap', 'Solver output unavailable — build build-mpi first');
    return;
  }
  const terrain = analyseTerrain(bed);
  const speed = new Float32Array(depth.data.length);
  for (let i = 0; i < speed.length; i++) { const h = depth.data[i]; speed[i] = h > .05 && hu && hv ? Math.hypot(hu.data[i], hv.data[i]) / h : 0; }
  // smooth the speed field so the activity layer reads as a continuous heat map
  for (let pass = 0; pass < 3; pass++) {
    const src = speed.slice(), R = 2, {rows, cols} = depth;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      let sum = 0, n = 0;
      for (let dr = -R; dr <= R; dr++) for (let dc = -R; dc <= R; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && cc >= 0 && rr < rows && cc < cols) { sum += src[rr * cols + cc]; n++; }
      }
      speed[r * cols + c] = sum / n;
    }
  }
  const [landUse, fr] = await Promise.all([loadLandUse('/showcase/landuse.csv'), loadFrames('/showcase/')]);
  const frames = fr ? fr.frames : [{t: 120, h: depth.data, hu: hu?.data, hv: hv?.data}];
  const scale = niceDepthScale(maxOf(depth.data));
  $$('#heroStack canvas').forEach(c => {
    const kind = c.dataset.layer;
    if (kind !== 'flood' && kind !== 'urban') { heroLayer(c, kind, depth, terrain, speed); return; }
    c.width = c.height = 360;
    const r = new CityRenderer(c, {animate: kind === 'flood', particles: 500});
    r.setData({rows: depth.rows, cols: depth.cols, cell: 5, bed: bed.data, landUse, frames, scenario: 'urban-district', scale});
    if (kind === 'urban') { r.showWater = false; r.draw(); return; }
    // loop the real time-lapse in the hero
    r.onPlayState = playing => { if (!playing) setTimeout(() => r.play(), 2500); };
    if (frames.length > 1) r.play();
  });
  showcaseData = {depth, bed, landUse, frames, scale};
  setText('stackCap', `Real solver output · urban district ${depth.rows}×${depth.cols} cells · t = 120 s`);
  // compact copy for the problem-statement slide
  const mini = $('#problemStack');
  if (mini && !mini.children.length) {
    const kinds = [['activity', '#2563eb', 'Rainfall Input'], ['urban', '#334155', 'Urban Terrain'], ['elevation', '#16a34a', 'Elevation & Land Use'], ['flood', '#4f46e5', 'Flood Simulation']];
    kinds.forEach(([k, color, label], idx) => {
      const layer = document.createElement('div'); layer.className = 'layer'; layer.style.top = `${40 + idx * 92}px`; layer.style.zIndex = 10 - idx;
      const cv = document.createElement('canvas'); cv.width = cv.height = 260; layer.appendChild(cv); mini.appendChild(layer);
      if (k === 'flood' || k === 'urban') {
        const r = new CityRenderer(cv, {animate: false});
        r.setData({rows: depth.rows, cols: depth.cols, cell: 5, bed: bed.data, landUse, frames: [frames[frames.length - 1]], scenario: 'urban-district', scale});
        if (k === 'urban') { r.showWater = false; r.draw(); }
      } else heroLayer(cv, k, depth, terrain, speed);
      const lab = document.createElement('div'); lab.className = 'stack-label'; lab.style.top = `${70 + idx * 92}px`; lab.style.color = color; lab.innerHTML = `<i></i>${label}`; mini.appendChild(lab);
    });
  }
}
function buildRain(el, n = 34) {
  for (let i = 0; i < n; i++) { const s = document.createElement('span'); s.style.left = `${Math.random() * 100}%`; s.style.animationDuration = `${.7 + Math.random() * .6}s`; s.style.animationDelay = `${-Math.random()}s`; el.appendChild(s); }
}
function buildSkyline() {
  const g = $('#skyline-buildings'), t = $('#skyline-trees'); if (!g) return;
  let x = 0, seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  let svg = '';
  while (x < 1440) { const w = 18 + rnd() * 34, h = 30 + rnd() * (x > 380 && x < 1100 ? 120 : 70); svg += `<rect x="${x.toFixed(0)}" y="${(152 - h).toFixed(0)}" width="${w.toFixed(0)}" height="${h.toFixed(0)}" rx="1.5"/>`; x += w + 4 + rnd() * 10; }
  g.innerHTML = svg;
  let trees = ''; for (let i = 0; i < 60; i++) { const cx = rnd() * 1440, r = 6 + rnd() * 10; trees += `<circle cx="${cx.toFixed(0)}" cy="${(150 - r * .4).toFixed(0)}" r="${r.toFixed(1)}"/>`; }
  t.innerHTML = trees;
}

/* ------------------------------------------------------------- lab state */
let showcaseData = null, labRenderer = null, predRenderer = null, refRenderer = null;
let caps = {}, capInfo = {}, labData = null, mapMode = 'flood', currentSummary = null, polling = false;

async function loadCapabilities() {
  try { capInfo = await getJson('/api/capabilities'); caps = capInfo.backends || {}; } catch { caps = {}; }
  const available = name => caps[name]?.status === 'AVAILABLE' && (name !== 'mpi' || capInfo.mpiLauncherAvailable);
  $$('.tech').forEach(t => {
    const name = t.dataset.backend, ok = available(name);
    t.querySelector('.dot').className = `dot ${ok ? 'on' : 'off'}`;
    t.querySelector('small').textContent = ok ? 'available' : name === 'cuda' ? 'needs NVIDIA GPU' : name === 'openacc' ? 'needs nvc++ / -fopenacc' : 'unavailable';
    t.title = caps[name]?.detail || '';
  });
  const pick = $('#backendPick'); pick.innerHTML = '';
  for (const [name, m] of Object.entries(BACKEND_META)) {
    const ok = available(name), b = document.createElement('button');
    b.type = 'button'; b.className = `bk k-${name}`; b.dataset.backend = name; b.disabled = !ok;
    b.setAttribute('aria-pressed', 'false');
    b.title = ok ? caps[name]?.detail || '' : `Unavailable: ${caps[name]?.detail || 'not built'}`;
    b.innerHTML = `<svg><use href="#${m.icon}"/></svg><span>${m.label}<small>${ok ? m.sub : 'not on this host'}</small></span>`;
    b.addEventListener('click', () => selectBackend(name));
    pick.appendChild(b);
  }
  selectBackend(available('openmp') ? 'openmp' : 'serial');
}
function selectBackend(name) {
  $('#backend').value = name;
  $$('#backendPick .bk').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.backend === name)));
  updateForm();
}
function updateForm() {
  const backend = $('#backend').value, scenario = $('#scenario').value, adaptive = $('#adaptive');
  $('#threadsField').hidden = backend !== 'openmp';
  $('#ranksField').hidden = backend !== 'mpi';
  const rainy = ['urban-district', 'catchment'].includes(scenario);
  const adaptiveAllowed = ['serial', 'openmp', 'mpi'].includes(backend) && !rainy;
  adaptive.disabled = !adaptiveAllowed; if (!adaptiveAllowed) adaptive.checked = false;
  $('#adaptiveFields').hidden = !adaptive.checked;
  $('#dynamicBalance').disabled = backend !== 'mpi'; if (backend !== 'mpi') $('#dynamicBalance').checked = false;
  const note = [];
  if (rainy) note.push('Rain wets every cell here, so adaptive refinement is disabled for this scenario.');
  else if (!adaptiveAllowed) note.push('Adaptive mesh runs use the Serial, OpenMP or MPI backends.');
  if (adaptive.checked) note.push(`Accuracy reference: uniform grid at ${2 ** Number($('#maxLevel').value)}× resolution.`);
  else if ($('#reference').checked) note.push('Accuracy reference: uniform grid at 2× resolution (block-averaged).');
  $('#formNote').textContent = note.join(' ');
}
function applyScenarioDefaults() {
  const d = SCENARIO_DEFAULTS[$('#scenario').value]; if (!d) return;
  $('#rows').value = d.rows; $('#cols').value = d.cols; $('#duration').value = d.duration; applyParamDefaults(); updateForm();
}
$$('.seg').forEach(seg => seg.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('button', seg).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  document.getElementById(seg.dataset.target).value = b.dataset.v;
}));
$('#scenario').addEventListener('change', applyScenarioDefaults);
['adaptive', 'reference', 'maxLevel'].forEach(id => document.getElementById(id).addEventListener('change', updateForm));

function showMessage(text, kind = '') { const m = $('#runMessage'); m.hidden = !text; m.textContent = text; m.className = `msg ${kind}`; }
function setBadge(state) {
  const b = $('#runBadge'), map = {RUNNING: 'run', PASS: 'pass', FAIL: 'fail'};
  b.className = `badge ${map[state] || 'idle'}`; b.textContent = state === 'PASS' ? 'VALIDATED' : state || 'IDLE';
}
const STAGES = ['input', 'decomposition', 'adaptive', 'solver', 'output'];
const STAGE_ORDER = {input: 0, decomposition: 1, adaptive: 2, solver: 3, output: 4, done: 5};
function highlightStage(stage, adaptive) {
  const idx = STAGE_ORDER[stage] ?? -1;
  $$('.stage').forEach(el => {
    const s = el.dataset.stage, i = STAGE_ORDER[s];
    const skipped = s === 'adaptive' && !adaptive;
    el.classList.toggle('live', s === stage || (stage === 'solver' && s === 'decomposition' && false));
    el.classList.toggle('done', !skipped && i < idx);
  });
}

$('#runForm').addEventListener('submit', async e => {
  e.preventDefault();
  const form = new FormData($('#runForm'));
  form.set('adaptive', $('#adaptive').checked ? 'true' : 'false');
  form.set('dynamic_load_balancing', $('#dynamicBalance').checked ? 'true' : 'false');
  form.set('reference', $('#reference').checked ? 'true' : 'false');
  if (!$('#customStorm').checked) { form.delete('rain_intensity'); form.delete('rain_duration'); }
  $('#runButton').disabled = true; showMessage('');
  try {
    await getJson('/run', {method: 'POST', body: new URLSearchParams(form)});
    setBadge('RUNNING'); $('#progress').hidden = false;
    pollRun($('#adaptive').checked);
  } catch (err) { showMessage(err.message, 'err'); $('#runButton').disabled = false; }
});

async function pollRun(adaptive) {
  if (polling) return; polling = true;
  const started = performance.now();
  const bar = $('#progress .bar');
  try {
    for (;;) {
      const st = await getJson('/status');
      if (st.running) {
        const p = st.progress || {};
        bar.classList.toggle('indet', !p.step);
        bar.querySelector('i').style.width = p.step ? `${(p.step / p.total) * 100}%` : '30%';
        setText('progressText', p.message || 'Running solver…');
        setText('progressStep', `${((performance.now() - started) / 1000).toFixed(1)} s`);
        highlightStage(p.stage, adaptive);
        await new Promise(r => setTimeout(r, 600));
        continue;
      }
      bar.classList.remove('indet'); bar.querySelector('i').style.width = '100%';
      highlightStage('done', adaptive);
      setTimeout(() => $$('.stage').forEach(el => el.classList.remove('done')), 6000);
      if (st.error || !st.summary) { setBadge('FAIL'); showMessage(st.error || 'The solver finished without a result.', 'err'); }
      else {
        setBadge(st.state); setText('progressText', 'Completed');
        showMessage(st.summary.validation_pass ? 'Run complete — matches the serial reference and conserves mass.' : 'Run complete, but a validation check failed.', st.summary.validation_pass ? 'ok' : 'err');
        await loadRunOutput(st.summary);
      }
      break;
    }
  } catch (err) { showMessage(err.message, 'err'); }
  finally { polling = false; $('#runButton').disabled = false; }
}

/* ----------------------------------------------------------- lab output */
async function loadRunOutput(summary) {
  currentSummary = summary;
  const uniform = !summary.adaptive;
  const [depth, bed, refine, ref, landUse, fr] = await Promise.all([
    optionalGrid('/depth.csv'), optionalGrid('/terrain.csv'),
    uniform ? null : optionalGrid('/refinement.csv'), optionalGrid('/reference-depth.csv'),
    loadLandUse('/landuse.csv'), uniform ? loadFrames('/') : null]);
  if (!depth) return;
  const bedOk = bed && bed.rows === depth.rows && bed.cols === depth.cols;
  const terrain = bedOk ? analyseTerrain(bed) : null;
  const scale = niceDepthScale(Math.max(maxOf(depth.data), ref ? maxOf(ref.data) : 0));
  const cell = num(summary.cell_size_m) || 5;
  const finalT = num(summary.stats?.time_s) ?? 0;
  const frames = fr && fr.meta.rows === depth.rows ? fr.frames : [{t: finalT, h: depth.data, hu: null, hv: null}];
  labData = {depth, terrain, refine, ref, scale, summary, landUse, frames, cell, bed: bedOk ? bed.data : null};
  $('#labEmpty').hidden = true;
  if (!labRenderer) {
    labRenderer = new CityRenderer($('#labMap'));
    labRenderer.onTime = t => { setText('simTime', `t = ${t.toFixed(1)} / ${labRenderer.times[labRenderer.times.length - 1].toFixed(0)} s`); if (!scrubbing) $('#scrub').value = Math.round((t - labRenderer.times[0]) / (labRenderer.duration || 1) * 1000); };
    labRenderer.onPlayState = on => $('#playBtn').classList.toggle('on', on);
  }
  labRenderer.pause();
  labRenderer.setData({rows: depth.rows, cols: depth.cols, cell, bed: labData.bed, landUse, frames, scenario: summary.scenario, scale, rainfall: summary.rainfall});
  labRenderer.onTime(labRenderer.simT);
  $('#playback').hidden = frames.length < 2;
  renderLab(); renderMetrics(summary, depth); renderOutcomes(summary, depth, ref, terrain, scale);
  if (frames.length > 1 && mapMode === 'flood') labRenderer.play();
}
let scrubbing = false;
$('#playBtn').addEventListener('click', () => { if (!labRenderer) return; labRenderer.playing ? labRenderer.pause() : labRenderer.play(); });
$('#scrub').addEventListener('input', e => { if (!labRenderer) return; scrubbing = true; labRenderer.pause(); labRenderer.seek(labRenderer.times[0] + e.target.value / 1000 * labRenderer.duration); scrubbing = false; });
$('#speedSeg').addEventListener('click', e => { const b = e.target.closest('button'); if (!b || !labRenderer) return; $$('#speedSeg button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); labRenderer.speed = Number(b.dataset.v); });

function meshOverlay(refine) {
  return ctx => {
    if (!refine) return;
    const {rows, cols, data} = refine, W = ctx.canvas.width / cols, H = ctx.canvas.height / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const L = Math.round(data[r * cols + c]); if (L <= 0) continue;
      const f = LEVEL_FILL[Math.min(L, 3)]; ctx.fillStyle = `rgba(${f[0]},${f[1]},${f[2]},${f[3]})`; ctx.fillRect(c * W, r * H, W + .5, H + .5);
      const k = 2 ** L; if (W / k < 2.2) continue;
      ctx.strokeStyle = L >= 2 ? 'rgba(255,237,213,.55)' : 'rgba(237,233,254,.55)'; ctx.lineWidth = 1; ctx.beginPath();
      for (let s = 1; s < k; s++) { ctx.moveTo(c * W + s * W / k, r * H); ctx.lineTo(c * W + s * W / k, (r + 1) * H); ctx.moveTo(c * W, r * H + s * H / k); ctx.lineTo((c + 1) * W, r * H + s * H / k); }
      ctx.stroke();
    }
    if (W >= 5) { ctx.strokeStyle = 'rgba(255,255,255,.18)'; ctx.beginPath(); for (let c = 0; c <= cols; c++) { ctx.moveTo(c * W, 0); ctx.lineTo(c * W, ctx.canvas.height); } for (let r = 0; r <= rows; r++) { ctx.moveTo(0, r * H); ctx.lineTo(ctx.canvas.width, r * H); } ctx.stroke(); }
  };
}
function partitionOverlay(depth, part) {
  return ctx => {
    if (!part) return;
    const rs = splits(depth.rows, part.pr), cs = splits(depth.cols, part.pc), sx = ctx.canvas.width / depth.cols, sy = ctx.canvas.height / depth.rows;
    let k = 0;
    for (let i = 0; i < part.pr; i++) for (let j = 0; j < part.pc; j++, k++) {
      const x = cs[j] * sx, y = rs[i] * sy, w = (cs[j + 1] - cs[j]) * sx, h = (rs[i + 1] - rs[i]) * sy, col = RANK_COLORS[k % RANK_COLORS.length];
      ctx.fillStyle = col + '2e'; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.setLineDash([10, 6]); ctx.strokeRect(x + 1.5, y + 1.5, w - 3, h - 3); ctx.setLineDash([]);
      if (part.pr * part.pc > 1) { ctx.fillStyle = col + '66'; if (i > 0) ctx.fillRect(x, y, w, Math.max(2, sy)); if (j > 0) ctx.fillRect(x, y, Math.max(2, sx), h); }
      const label = part.gpu ? 'GPU · one thread per cell' : part.pr * part.pc === 1 ? 'Single domain' : `${part.kind} ${k}`;
      ctx.font = '700 15px Inter, system-ui, sans-serif'; const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(255,255,255,.94)'; ctx.beginPath(); ctx.roundRect(x + 12, y + 12, tw + 20, 28, 14); ctx.fill();
      ctx.fillStyle = col; ctx.fillText(label, x + 22, y + 31);
    }
  };
}
function renderLab() {
  if (!labData || !labRenderer) return;
  const {depth, terrain, refine, scale, summary} = labData, r = labRenderer;
  const legend = $('#labLegend'); legend.hidden = false;
  const depthLegend = `Water depth (m)<div class="legend-bar" style="background:${cssRamp(WATER_REAL)}"></div><div class="legend-scale"><span>0</span><span>${scale / 2}</span><span>${scale}</span></div>`;
  const LEGEND_LU = {
    'urban-district': '<span><i style="background:#4e5258"></i>Road</span><span><i style="background:#c8c4bc"></i>Pavement</span><span><i style="background:#5f8a4e"></i>Park</span><span><i style="background:#c4bfb6;border:1px solid #9a948a"></i>Building roof</span>',
    'dam-break': '<span><i style="background:#766c56"></i>Reservoir bed</span><span><i style="background:#93a052"></i>Farmland</span>',
    'rain-drain': '<span><i style="background:#93a052"></i>Farmland catchment</span>',
    'localized-refinement': '<span><i style="background:#766c56"></i>Pond basin</span><span><i style="background:#5f8a4e"></i>Park</span>',
    'dam-breach': '<span><i style="background:#766c56"></i>Reservoir / stream bed</span><span><i style="background:#b2b0a8"></i>Dam</span><span><i style="background:#93a052"></i>Farmland</span><span><i style="background:#c4bfb6;border:1px solid #9a948a"></i>Houses</span><span><i style="background:#4e5258"></i>Road</span>',
    'catchment': '<span><i style="background:#766c56"></i>Stream bed</span><span><i style="background:#93a052"></i>Farmland</span><span><i style="background:#5f8a4e"></i>Upland pasture</span><span><i style="background:#c4bfb6;border:1px solid #9a948a"></i>Houses</span><span><i style="background:#4e5258"></i>Road</span>'};
  const lu = LEGEND_LU[summary.scenario] ? `<div class="legend-items">${LEGEND_LU[summary.scenario]}</div>` : '';
  r.showWater = mapMode !== 'terrain'; r.overlay = null;
  if (mapMode === 'flood') { legend.innerHTML = depthLegend + lu + (r.frames.length > 1 ? '<div class="legend-items"><span><i style="background:#fff;border:1px solid #94a3b8;height:2px"></i>Flow tracers</span></div>' : ''); }
  else if (mapMode === 'terrain') {
    legend.innerHTML = lu ? `Land use${lu}${terrain?.hasBuildings ? `<div class="legend-scale" style="margin-top:4px"><span>buildings 6–${terrain.bmax.toFixed(0)} m tall</span></div>` : ''}` : (terrain?.hasRelief ? 'Ground relief (hillshade)' : summary.adaptive ? 'Adaptive runs export depth and refinement only' : 'Flat bed (no relief)');
  } else if (mapMode === 'mesh') {
    r.overlay = meshOverlay(refine);
    legend.innerHTML = refine ? `Refinement level<div class="legend-items"><span><i style="background:transparent;border:1px solid #cbd5e1"></i>L0 base cell</span><span><i style="background:rgba(124,58,237,.45)"></i>L1 · 2× finer</span><span><i style="background:rgba(234,88,12,.55)"></i>L2 · 4× finer</span></div>` : 'Uniform grid — enable adaptive mesh refinement to see patches';
  } else {
    const part = partitionFor(summary); r.overlay = partitionOverlay(depth, part);
    legend.innerHTML = summary.backend === 'mpi' ? `MPI ${part.pr}×${part.pc} Cartesian blocks<div class="legend-items"><span><i style="background:rgba(0,0,0,.25)"></i>Halo exchange rows/cols</span></div>` : summary.backend === 'openmp' ? `OpenMP static schedule<br><span style="font-weight:500">${part.pr} row chunks</span>` : `${BACKEND_META[summary.backend]?.label} decomposition`;
  }
  r.draw();
  const s = summary, cell = labData.cell;
  setText('mapMetaLeft', `${depth.rows} × ${depth.cols} cells · ${cell} m cells · ${(depth.cols * cell / 1000).toFixed(2)} × ${(depth.rows * cell / 1000).toFixed(2)} km`);
  setText('mapMetaRight', r.frames.length > 1 ? `${r.frames.length} solver frames` : (s.backend === 'mpi' ? 'final state (MPI runs record no time-lapse)' : 'final state'));
}
$$('.map-tabs button').forEach(b => b.addEventListener('click', () => {
  mapMode = b.dataset.mode; $$('.map-tabs button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); renderLab();
}));
$('#labMapWrap').addEventListener('mousemove', e => {
  const tip = $('#labTip'); if (!labData) return;
  const rect = $('#labMap').getBoundingClientRect(), {depth, terrain, refine} = labData;
  const c = Math.floor((e.clientX - rect.left) / rect.width * depth.cols), r = Math.floor((e.clientY - rect.top) / rect.height * depth.rows);
  if (r < 0 || c < 0 || r >= depth.rows || c >= depth.cols) { tip.hidden = true; return; }
  const i = r * depth.cols + c, h = labRenderer ? labRenderer.cur.h[i] : depth.data[i];
  const lines = [`cell (${r}, ${c}) · t ${labRenderer ? labRenderer.simT.toFixed(1) : '—'} s`, `depth ${h.toFixed(3)} m`];
  if (terrain) lines.push(`bed ${terrain.bed[i].toFixed(2)} m${labData.landUse ? ' · ' + ['pavement', 'road', 'building', 'park', 'river', 'open'][labData.landUse[i]] : ''}`);
  if (refine) lines.push(`refinement L${Math.round(refine.data[i])}`);
  tip.innerHTML = lines.join('<br>'); tip.hidden = false;
  const wrap = $('#labMapWrap').getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX - wrap.left, wrap.width - 170)}px`; tip.style.top = `${e.clientY - wrap.top}px`;
});
$('#labMapWrap').addEventListener('mouseleave', () => { $('#labTip').hidden = true; });

function renderMetrics(s, depth) {
  const meta = BACKEND_META[s.backend] || {label: s.backend};
  setText('mBackend', `${meta.label}${s.adaptive ? ' · AMR' : ''}`);
  setText('mBackendH', s.backend === 'openmp' ? `${s.threads} threads` : s.backend === 'mpi' ? `${s.processes} ranks${s.dynamic_load_balancing ? ' · dynamic balancing' : ''}` : meta.sub || '');
  setText('mRuntime', fmtSec(num(s.backend_runtime))); setText('mRuntimeH', `serial ${s.adaptive ? 'adaptive ' : ''}reference: ${fmtSec(num(s.serial_runtime))}`);
  setText('mSpeedup', num(s.speedup) === null ? '—' : `${num(s.speedup).toFixed(2)}×`); setText('mEfficiency', `parallel efficiency ${num(s.efficiency) === null ? '—' : (num(s.efficiency) * 100).toFixed(0) + '%'}`);
  setText('mDepth', `${maxOf(depth.data).toFixed(2)} m`);
  const v = num(s.stats?.max_velocity_m_s ?? s.stats?.maximum_velocity); setText('mVelocity', `max velocity ${v === null ? '—' : v.toFixed(2) + ' m/s'}`);
  let wet = 0; for (const d of depth.data) if (d > WET) wet++;
  const cell = num(s.cell_size_m) || 5, area = wet * cell * cell;
  setText('mArea', area >= 1e6 ? `${(area / 1e6).toFixed(2)} km²` : `${(area / 1e4).toFixed(2)} ha`);
  setText('mCells', `${wet.toLocaleString()} of ${depth.data.length.toLocaleString()} cells`);
  const res = num(s.stats?.mass_residual_m3 ?? s.conservation_residual_m3);
  setText('mResidual', res === null ? '—' : `${fmtSci(res)} m³`);
  setText('mValid', s.validation_pass ? '✓ conservation + serial match' : '✗ check failed');
  setText('vMaxErr', s.adaptive ? 'n/a (AMR)' : `${fmtSci(num(s.max_error))} m`);
}

/* -------------------------------------------------------------- outcomes */
function renderOutcomes(s, depth, ref, terrain, scale) {
  const o = s.outcomes || {};
  const label = `${BACKEND_META[s.backend]?.label || s.backend}${s.adaptive ? ' adaptive' : ''} · ${s.scenario} · ${depth.rows}×${depth.cols}`;
  const prm = s.parameters || {}, bits = [];
  if (prm['rain-intensity']) bits.push(`rain ${Number(prm['rain-intensity'])} mm/h${prm['rain-duration'] ? ' for ' + Number(prm['rain-duration']) + ' s' : ''}`);
  if (prm.manning) bits.push(`n = ${Number(prm.manning)}`);
  if (prm['surge-height']) bits.push(`surge ${Number(prm['surge-height'])} m`);
  if (prm['reservoir-depth']) bits.push(`reservoir ${Number(prm['reservoir-depth'])} m`);
  if (prm['pond-depth']) bits.push(`pond ${Number(prm['pond-depth'])} m`);
  setText('outcomeSource', `Computed from the latest Simulation Lab run (${label}${bits.length ? ' · ' + bits.join(', ') : ''}).`);
  // 1 accuracy
  if (o.available) { setText('kIoU', `${(o.iou * 100).toFixed(1)}%`); setText('kIoUH', `Flood extent IoU vs ${o.factor}× resolution reference (depth > ${o.wet_threshold_m} m)`); }
  else { setText('kIoU', '—'); setText('kIoUH', `Accuracy reference ${o.reason || 'not computed'}`); }
  // 2 speedup
  const sp = num(s.speedup); setText('kSpeed', sp === null ? '—' : `${sp.toFixed(1)}×`);
  setText('kSpeedH', `Faster than the CPU serial baseline (${BACKEND_META[s.backend]?.label}${s.backend === 'openmp' ? `, ${s.threads} threads` : s.backend === 'mpi' ? `, ${s.processes} ranks` : ''})`);
  // 3 scalability
  if (s.adaptive && num(s.uniform_fine_cells) && num(s.stats?.active_leaf_cells)) {
    const ratio = s.uniform_fine_cells / s.stats.active_leaf_cells;
    setText('kScale', `${ratio.toFixed(1)}×`); setText('kScaleH', `Fewer cells than a uniform grid at the finest level (${fmtBig(s.stats.active_leaf_cells)} vs ${fmtBig(s.uniform_fine_cells)})`);
  } else if (num(s.throughput_cell_updates_per_s)) {
    setText('kScale', fmtBig(s.throughput_cell_updates_per_s)); setText('kScaleH', `Cell updates per second (${(s.steps || 0).toLocaleString()} timesteps × ${(depth.rows * depth.cols).toLocaleString()} cells)`);
  } else { setText('kScale', '—'); }
  // 4 error
  if (o.available) { setText('kErr', `${o.relative_error_pct < 10 ? '< ' + Math.max(1, Math.ceil(o.relative_error_pct)) : o.relative_error_pct.toFixed(1)}%`); setText('kErrH', `Mean relative depth error (${o.relative_error_pct.toFixed(2)}%), RMSE ${fmtSci(o.rmse_m)} m`); }
  else setText('kErr', '—');
  // maps
  const common = {rows: depth.rows, cols: depth.cols, cell: labData.cell, bed: labData.bed, landUse: labData.landUse, scenario: s.scenario, scale};
  predRenderer ||= new CityRenderer($('#predMap'), {animate: false});
  predRenderer.setData({...common, frames: [{t: 0, h: depth.data, hu: null, hv: null}]}); $('#predEmpty').hidden = true;
  if (ref && ref.rows === depth.rows && ref.cols === depth.cols) {
    refRenderer ||= new CityRenderer($('#refMap'), {animate: false});
    refRenderer.setData({...common, frames: [{t: 0, h: ref.data, hu: null, hv: null}]}); $('#refEmpty').hidden = true; setText('refCaption', `Reference (${o.factor}× resolution)`);
  } else { $('#refEmpty').hidden = false; $('#refMap').getContext('2d').clearRect(0, 0, 420, 420); }
  $('#vbar').style.background = cssRamp(WATER_REAL, '0deg');
  $('#vscale').innerHTML = [1, .75, .5, .25, 0].map(f => `<span>${(scale * f).toFixed(scale < 1 ? 2 : 1)}</span>`).join('');
  drawHistogram(o);
}

/* ---------------------------------------------------------------- charts */
const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs = {}, parent) { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; }
function niceTicks(lo, hi, n = 5) { const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n || 1))); const err = (hi - lo) / n / step; const m = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1; const s = m * step, out = []; for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-12; v += s) out.push(+v.toFixed(10)); return out; }
function tipFor(box) { let t = box.querySelector('.chart-tip'); if (!t) { t = document.createElement('div'); t.className = 'chart-tip'; t.hidden = true; box.appendChild(t); } return t; }

function drawHistogram(o) {
  const box = $('#histBox');
  if (!o.available || !o.histogram) { box.innerHTML = `<div class="empty-chart">${o.reason ? 'Accuracy reference ' + o.reason : 'Error distribution appears after a run with the accuracy reference enabled.'}</div>`; setText('histNote', ''); return; }
  const {edges, counts} = o.histogram, W = 420, H = 270, m = {l: 46, r: 10, t: 14, b: 40};
  box.innerHTML = '';
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Histogram of absolute water depth error'}, box);
  const cmax = Math.max(...counts, 1), yt = niceTicks(0, cmax, 4), ymax = yt[yt.length - 1] || cmax;
  const x = v => m.l + (v - edges[0]) / (edges[edges.length - 1] - edges[0]) * (W - m.l - m.r), y = v => H - m.b - v / ymax * (H - m.t - m.b);
  for (const t of yt) { el('line', {x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: '#e6ecf5'}, svg); el('text', {x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#64748b'}, svg).textContent = t; }
  const tip = tipFor(box);
  counts.forEach((c, i) => {
    const x0 = x(edges[i]) + 1, x1 = x(edges[i + 1]) - 1, h = H - m.b - y(c);
    if (c > 0) { const r = Math.min(4, (x1 - x0) / 2, h); el('path', {d: `M${x0},${H - m.b}V${y(c) + r}Q${x0},${y(c)} ${x0 + r},${y(c)}H${x1 - r}Q${x1},${y(c)} ${x1},${y(c) + r}V${H - m.b}Z`, fill: '#60a5fa', stroke: '#3b82f6', 'stroke-width': .8}, svg); }
    const hit = el('rect', {x: x(edges[i]), y: m.t, width: x(edges[i + 1]) - x(edges[i]), height: H - m.t - m.b, fill: 'transparent'}, svg);
    hit.addEventListener('mousemove', e => { tip.hidden = false; tip.innerHTML = `<b>${c.toLocaleString()} cells</b><br>|Δh| ${edges[i].toFixed(3)}–${edges[i + 1].toFixed(3)} m`; const b = box.getBoundingClientRect(); tip.style.left = `${Math.min(e.clientX - b.left + 10, b.width - 150)}px`; tip.style.top = `${e.clientY - b.top - 50}px`; });
    hit.addEventListener('mouseleave', () => { tip.hidden = true; });
  });
  el('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: '#94a3b8'}, svg);
  for (const t of niceTicks(edges[0], edges[edges.length - 1], 4)) el('text', {x: x(t), y: H - m.b + 16, 'text-anchor': 'middle', 'font-size': 11, fill: '#64748b'}, svg).textContent = t;
  el('text', {x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: '#334155'}, svg).textContent = 'Absolute Water Depth Error (m)';
  const yl = el('text', {x: 12, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', 'font-size': 12, fill: '#334155', transform: `rotate(-90 12 ${(H - m.b + m.t) / 2})`}, svg); yl.textContent = 'Frequency';
  const sb = document.createElement('div'); sb.className = 'statbox'; sb.innerHTML = `Mean Error: ${o.mean_abs_error_m.toFixed(3)} m<br>Std. Dev.: ${o.std_abs_error_m.toFixed(3)} m`; box.appendChild(sb);
  setText('histNote', `Cells flooded (> ${o.wet_threshold_m} m) in either the model or the ${o.factor}× reference: ${(o.flooded_cells_model + o.flooded_cells_reference).toLocaleString()} counted. Precision ${(o.precision * 100).toFixed(1)}%, recall ${(o.recall * 100).toFixed(1)}%.`);
}

function drawPerf(data) {
  const box = $('#perfBox'), legend = $('#perfLegend'), pts = data.points || [];
  if (!pts.length) { legend.innerHTML = ''; box.innerHTML = `<div class="empty-chart">${data.running ? 'Measuring… the first points appear in a few seconds.' : 'Run a scaling sweep to measure runtime against domain size for every available backend.'}</div>`; return; }
  const series = {}; for (const p of pts) (series[p.backend] ||= []).push(p);
  const order = Object.keys(BACKEND_META).filter(k => series[k]);
  legend.innerHTML = order.map(k => `<span><i style="background:${BACKEND_META[k].color}"></i>${BACKEND_META[k].label}${k === 'openmp' ? ` (${data.threads} thr)` : k === 'mpi' ? ` (${data.processes} ranks)` : ''}</span>`).join('');
  const W = 460, H = 280, m = {l: 54, r: 70, t: 12, b: 42};
  box.innerHTML = '';
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Simulation time versus domain size, log scale'}, box);
  const xs = pts.map(p => p.domain_km2), ys = pts.map(p => p.solver_seconds);
  const xmax = Math.max(...xs) * 1.05, lymin = Math.floor(Math.log10(Math.min(...ys))), lymax = Math.ceil(Math.log10(Math.max(...ys)));
  const x = v => m.l + v / xmax * (W - m.l - m.r), y = v => H - m.b - (Math.log10(v) - lymin) / ((lymax - lymin) || 1) * (H - m.t - m.b);
  for (let e = lymin; e <= lymax; e++) { el('line', {x1: m.l, x2: W - m.r, y1: y(10 ** e), y2: y(10 ** e), stroke: '#e6ecf5'}, svg); const t = el('text', {x: m.l - 8, y: y(10 ** e) + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#64748b'}, svg); t.innerHTML = `10<tspan dy="-5" font-size="8">${e}</tspan>`; }
  for (const t of niceTicks(0, xmax, 5)) el('text', {x: x(t), y: H - m.b + 16, 'text-anchor': 'middle', 'font-size': 11, fill: '#64748b'}, svg).textContent = t;
  el('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: '#94a3b8'}, svg);
  el('text', {x: (W - m.r + m.l) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: '#334155'}, svg).textContent = 'Domain Size (km²)';
  el('text', {x: 13, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', 'font-size': 12, fill: '#334155', transform: `rotate(-90 13 ${(H - m.b + m.t) / 2})`}, svg).textContent = 'Simulation Time (s)';
  const all = [];
  for (const k of order) {
    const s = series[k].sort((a, b) => a.domain_km2 - b.domain_km2), col = BACKEND_META[k].color;
    el('path', {d: s.map((p, i) => `${i ? 'L' : 'M'}${x(p.domain_km2)},${y(p.solver_seconds)}`).join(''), fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-dasharray': k === 'serial' ? '5 4' : ''}, svg);
    for (const p of s) { el('circle', {cx: x(p.domain_km2), cy: y(p.solver_seconds), r: 4.5, fill: col, stroke: '#fff', 'stroke-width': 2}, svg); all.push({...p, col}); }
    const last = s[s.length - 1]; el('text', {x: x(last.domain_km2) + 9, y: y(last.solver_seconds) + 4, 'font-size': 11, 'font-weight': 600, fill: '#334155'}, svg).textContent = BACKEND_META[k].label;
  }
  const tip = tipFor(box), cross = el('line', {y1: m.t, y2: H - m.b, stroke: '#94a3b8', 'stroke-dasharray': '3 3', visibility: 'hidden'}, svg);
  const hit = el('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}, svg);
  hit.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect(), vx = (e.clientX - r.left) / r.width * W;
    const sizes = [...new Set(all.map(p => p.domain_km2))]; const near = sizes.reduce((a, b) => Math.abs(x(b) - vx) < Math.abs(x(a) - vx) ? b : a);
    const at = all.filter(p => p.domain_km2 === near).sort((a, b) => b.solver_seconds - a.solver_seconds);
    cross.setAttribute('x1', x(near)); cross.setAttribute('x2', x(near)); cross.setAttribute('visibility', 'visible');
    const serial = at.find(p => p.backend === 'serial');
    tip.innerHTML = `<b>${near.toFixed(2)} km²</b> · ${at[0].rows}×${at[0].cols}<br>` + at.map(p => `<span style="color:${p.col}">●</span> ${BACKEND_META[p.backend].label}: ${fmtSec(p.solver_seconds)}${serial && p !== serial ? ` (${(serial.solver_seconds / p.solver_seconds).toFixed(1)}×)` : ''}`).join('<br>');
    tip.hidden = false; const b = box.getBoundingClientRect(); tip.style.left = `${Math.min(e.clientX - b.left + 12, b.width - 200)}px`; tip.style.top = `${Math.max(0, e.clientY - b.top - 70)}px`;
  });
  hit.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
  const done = data.complete ? 'complete' : `${pts.length}/${data.planned} runs`;
  setText('perfNote', `Measured on this machine: ${data.scenario}, ${data.duration_s} s simulated, ${data.cell_size_m} m cells (${done}). Log-scale runtime.${data.errors?.length ? ` ${data.errors.length} run(s) failed.` : ''}`);
}
async function loadSweep() {
  try { const d = await getJson('/api/sweep'); drawPerf(d); $('#sweepButton').disabled = d.running; $('#sweepButton').textContent = d.running ? 'Measuring…' : 'Run scaling sweep'; if (d.running) setTimeout(loadSweep, 1500); } catch {}
}
$('#sweepButton').addEventListener('click', async () => {
  $('#sweepButton').disabled = true;
  try { await getJson('/sweep', {method: 'POST', body: new URLSearchParams({scenario: 'urban-district'})}); loadSweep(); }
  catch (err) { $('#sweepButton').disabled = false; setText('perfNote', err.message); }
});

async function loadValidation() {
  try {
    const v = await getJson('/api/validation');
    setText('vTests', v.testLogAvailable ? `${v.passedTests}/${v.totalTests} passed` : 'not run');
    setText('vTestsH', v.testLogModified ? `CTest · ${new Date(v.testLogModified * 1000).toLocaleDateString()}` : 'run ctest --test-dir build-mpi');
    setText('vMass', v.maxMassBalanceResidualM3 === null ? '—' : `${fmtSci(v.maxMassBalanceResidualM3)} m³`);
    setText('vIface', v.maxInterfaceResidual === null ? '—' : fmtSci(v.maxInterfaceResidual));
  } catch {}
}

/* ----------------------------------------------------------------- boot */
function initReveal() {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), {threshold: .08});
  $$('.reveal').forEach(e => io.observe(e));
}
function initNav() {
  const links = $$('.nav a[href^="#"]'), targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`)); }), {rootMargin: '-45% 0px -50% 0px'});
  targets.forEach(t => io.observe(t));
}
async function boot() {
  buildSkyline(); buildRain($('#heroRain')); initReveal(); initNav();
  loadShowcase(); loadValidation(); loadSweep();
  await loadCapabilities(); applyParamDefaults(); updateForm();
  try {
    const st = await getJson('/status');
    if (st.running) { $('#progress').hidden = false; setBadge('RUNNING'); $('#runButton').disabled = true; pollRun(false); }
    else if (st.summary) { setBadge(st.state); await loadRunOutput(st.summary); }
  } catch {}
}
boot();
