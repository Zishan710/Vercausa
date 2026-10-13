"use strict";

const SIGNALS = [
  ["air_temperature", "Air temp", "K", 1],
  ["process_temperature", "Process temp", "K", 1],
  ["rotational_speed", "Speed", "rpm", 0],
  ["torque", "Torque", "Nm", 1],
  ["tool_wear", "Tool wear", "min", 0],
];
const LABEL = Object.fromEntries(SIGNALS.map(([k, l, u]) => [k, { label: l, unit: u }]));
LABEL.product_variant = { label: "Product variant", unit: "" };
LABEL.power = { label: "Power", unit: "W" };
LABEL.temperature_difference = { label: "Temp difference", unit: "K" };
const MODE_NAMES = { TWF: "Tool wear failure", HDF: "Heat dissipation failure", PWF: "Power failure", OSF: "Overstrain failure" };
const MODE_COLOR = { TWF: "#ffbe4d", HDF: "#ff8a3d", PWF: "#5aa8ff", OSF: "#ff6ec7" };
const CYAN = "#2fd9f4", VIOLET = "#8b7bff", RED = "#ff4d61", GREEN = "#35e0a1";
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const NODES = [
  ["feed", "IoT feed", "sensor stream", "M3 12h4l2.5-6 5 12 2.5-6h4"],
  ["detect", "Detection", "rule checks", "M12 3.5 21 19H3z M12 10v4 M12 16.6v.4"],
  ["kg", "Knowledge Graph", "digital twin", "M6 6.5a1.5 1.5 0 1 0 0 .01 M18 6.5a1.5 1.5 0 1 0 0 .01 M12 18a1.5 1.5 0 1 0 0 .01 M7 7.5l4 9 M17 7.5l-4 9 M7.5 6.5h9"],
  ["causal", "Causal model", "root cause", "M3 12h9 M9 7l5 5-5 5 M18 5v14"],
  ["sim", "Simulation", "what-if", "M4 7h9 M17 7h3 M4 17h4 M12 17h8 M15 4.5v5 M10 14.5v5"],
  ["reason", "Reasoning", "GraphRAG · next", "M4 5h16v10H9l-5 4z"],
  ["action", "Action", "AI agents · next", "M4 12.5l4.5 4.5L20 6"],
];

let meta = null;
let state = null;
let seen = new Set();
let rendered = new Set();
let openEvent = null;
let lastPosition = 0;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const svgEl = (tag, attrs) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const fmt = (v, d = 1) => Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (p) => (p >= 0.995 ? "100%" : p < 0.001 ? "0%" : p < 0.01 ? (p * 100).toFixed(1) + "%" : Math.round(p * 100) + "%");
const time = (ts) => (ts ? ts.replace("T", " ").slice(0, 16) : "—");
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}
const control = (action, value) => post("/api/control", { action, value }).catch((e) => console.error(e));

/* ---------- number tween ---------- */

function tween(node, to, digits = 1, suffix = "") {
  if (to === null || to === undefined) { node.textContent = "—"; node._v = null; return; }
  const from = node._v ?? to;
  node._v = to;
  if (REDUCED || from === to) { node.textContent = fmt(to, digits) + suffix; return; }
  const t0 = performance.now(), dur = 450;
  cancelAnimationFrame(node._raf);
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    node.textContent = fmt(from + (to - from) * e, digits) + suffix;
    if (k < 1) node._raf = requestAnimationFrame(step);
  };
  node._raf = requestAnimationFrame(step);
}

/* ---------- pipeline flow ---------- */

const particles = [];
let dashOffset = 0;

function buildPipeline() {
  const wrap = $("nodes");
  NODES.forEach(([key, name, sub, icon], i) => {
    const n = el("div", "node" + (i >= 5 ? " future" : ""));
    n.id = "node-" + key;
    const dot = el("div", "dot");
    const s = svgEl("svg", { viewBox: "0 0 24 24", "stroke-linecap": "round", "stroke-linejoin": "round" });
    s.append(svgEl("path", { d: icon }));
    dot.append(s);
    n.append(dot, el("div", "nm", name), el("div", "sb", sub));
    wrap.append(n);
  });
  requestAnimationFrame(flowLoop);
}

function nodeCenters() {
  const c = $("flow").getBoundingClientRect();
  return NODES.map(([key]) => {
    const r = $("node-" + key).querySelector(".dot").getBoundingClientRect();
    return { x: r.left + r.width / 2 - c.left, y: r.top + r.height / 2 - c.top };
  });
}

function flash(i, cls) {
  const n = $("node-" + NODES[i][0]);
  n.classList.remove("hit", "hit-bad", "hit-sim");
  void n.offsetWidth;
  n.classList.add(cls);
  clearTimeout(n._t);
  n._t = setTimeout(() => n.classList.remove(cls), cls === "hit" ? 260 : 900);
}

function pulse(from, to, kind) {
  if (REDUCED) { for (let i = from + 1; i <= to; i++) flash(i, kind === "bad" ? "hit-bad" : kind === "sim" ? "hit-sim" : "hit"); return; }
  if (particles.length > 28) particles.shift();
  const speed = kind === "normal" ? clamp(0.9 + (state ? state.speed : 2) * 0.12, 1, 4) : 1.6;
  particles.push({ seg: from, end: to, t: 0, kind, speed, color: kind === "bad" ? RED : kind === "sim" ? VIOLET : CYAN });
  flash(from, kind === "bad" ? "hit-bad" : kind === "sim" ? "hit-sim" : "hit");
}

let lastFrame = performance.now();
function flowLoop(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const c = $("flow");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pts = nodeCenters();

  dashOffset -= dt * 22;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1], future = i >= 4;
    const g = ctx.createLinearGradient(a.x, 0, b.x, 0);
    g.addColorStop(0, future ? "rgba(125,160,255,0.12)" : "rgba(47,217,244,0.35)");
    g.addColorStop(1, future ? "rgba(125,160,255,0.08)" : "rgba(139,123,255,0.35)");
    ctx.strokeStyle = g;
    ctx.lineWidth = 2;
    ctx.setLineDash(future ? [3, 7] : [8, 10]);
    ctx.lineDashOffset = future ? 0 : dashOffset;
    ctx.beginPath(); ctx.moveTo(a.x + 22, a.y); ctx.lineTo(b.x - 22, b.y); ctx.stroke();
  }
  ctx.setLineDash([]);

  for (let k = particles.length - 1; k >= 0; k--) {
    const p = particles[k];
    p.t += dt * p.speed * 2.2;
    while (p.t >= 1 && p.seg < p.end) {
      p.t -= 1; p.seg += 1;
      flash(p.seg, p.kind === "bad" ? "hit-bad" : p.kind === "sim" ? "hit-sim" : "hit");
      if (p.seg >= p.end) break;
    }
    if (p.seg >= p.end) { particles.splice(k, 1); continue; }
    const a = pts[p.seg], b = pts[p.seg + 1];
    const x = a.x + (b.x - a.x) * p.t, y = a.y + (b.y - a.y) * p.t;
    const trail = ctx.createLinearGradient(x - 46, y, x, y);
    trail.addColorStop(0, "rgba(0,0,0,0)");
    trail.addColorStop(1, p.color);
    ctx.strokeStyle = trail; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(Math.max(a.x, x - 46), y); ctx.lineTo(x, y); ctx.stroke();
    ctx.shadowColor = p.color; ctx.shadowBlur = 16; ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, p.kind === "normal" ? 3.2 : 4.6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }
  requestAnimationFrame(flowLoop);
}

/* ---------- machine rig ---------- */

function buildRig() {
  const teeth = $("teeth");
  for (let i = 0; i < 12; i++) {
    teeth.append(svgEl("path", { d: "M 95 52 L 105 52 L 108 60 L 92 60 Z", transform: `rotate(${i * 30} 100 100)`, fill: CYAN }));
  }
}

function wearColor(f) {
  const stops = [[47, 217, 244], [255, 190, 77], [255, 77, 97]];
  const x = clamp(f, 0, 1) * 2, i = Math.min(1, Math.floor(x)), k = x - i;
  const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * k));
  return `rgb(${c.join(",")})`;
}

let lastSpin = 0;
function updateRig(live, alert) {
  const rig = $("rig");
  rig.classList.toggle("on", !!live);
  rig.classList.toggle("fault", !!alert);
  if (!live) {
    $("rig-rpm").textContent = "—"; $("rig-wear").textContent = "—";
    $("teeth").querySelectorAll("path").forEach((p) => p.setAttribute("fill", "#56657f"));
    return;
  }
  const rpm = live.values.rotational_speed;
  const spin = clamp((1.4 * 1500) / rpm, 0.35, 3);
  if (Math.abs(spin - lastSpin) / (lastSpin || 1) > 0.05) { $("cutter").style.setProperty("--spin", spin.toFixed(2) + "s"); lastSpin = spin; }
  tween($("rig-rpm"), rpm, 0);
  $("rig-wear").textContent = fmt(live.values.tool_wear, 0) + " min";
  const col = alert ? RED : wearColor(live.values.tool_wear / 240);
  $("teeth").querySelectorAll("path").forEach((p) => p.setAttribute("fill", col));
}

/* ---------- gauges ---------- */

const ARC_LEN = Math.PI * 55;
function buildGauges() {
  const wrap = $("gauges");
  for (const [key, label, unit] of SIGNALS) {
    const g = el("div", "gauge");
    const svg = svgEl("svg", { viewBox: "0 0 140 92" });
    const defs = svgEl("defs", {});
    const lg = svgEl("linearGradient", { id: "g-" + key, x1: "0", y1: "0", x2: "1", y2: "0" });
    lg.append(svgEl("stop", { offset: "0", "stop-color": "#2fd9f4" }), svgEl("stop", { offset: "1", "stop-color": "#8b7bff" }));
    defs.append(lg);
    svg.append(defs);
    for (let i = 0; i <= 10; i++) {
      const a = Math.PI - (i / 10) * Math.PI;
      const r1 = 63, r2 = i % 5 === 0 ? 69 : 66;
      svg.append(svgEl("line", { x1: 70 + r1 * Math.cos(a), y1: 80 - r1 * Math.sin(a), x2: 70 + r2 * Math.cos(a), y2: 80 - r2 * Math.sin(a), stroke: "rgba(135,151,177,0.45)", "stroke-width": 1.2 }));
    }
    svg.append(svgEl("path", { d: "M 15 80 A 55 55 0 0 1 125 80", fill: "none", stroke: "rgba(255,255,255,0.07)", "stroke-width": 10, "stroke-linecap": "round" }));
    const arc = svgEl("path", { d: "M 15 80 A 55 55 0 0 1 125 80", fill: "none", stroke: `url(#g-${key})`, "stroke-width": 10, "stroke-linecap": "round", "stroke-dasharray": `0 ${ARC_LEN}`, class: "arc", id: "arc-" + key });
    svg.append(arc);
    const val = el("div", "gauge-value", "—");
    val.id = "val-" + key;
    g.append(svg, val, el("div", "gauge-unit", unit), el("div", "gauge-label", label));
    wrap.append(g);
  }
}

function updateGauges(values) {
  for (const [key, , , digits] of SIGNALS) {
    const r = meta.ranges[key];
    const v = values ? values[key] : null;
    tween($("val-" + key), v, digits);
    const frac = v === null ? 0 : clamp((v - r.min) / (r.max - r.min), 0.015, 1);
    $("arc-" + key).setAttribute("stroke-dasharray", `${(frac * ARC_LEN).toFixed(1)} ${ARC_LEN}`);
  }
}

/* ---------- physics meters ---------- */

const METERS = {};
function buildMeter(id, name, unit, lo, hi, safeLo, safeHi, digits) {
  const m = $(id);
  const top = el("div", "meter-top");
  const nm = el("span", "meter-name");
  nm.append(el("b", null, name), document.createTextNode(" · physics"));
  const val = el("span", "meter-value", "—");
  top.append(nm, val);
  const track = el("div", "meter-track");
  const safe = el("div", "meter-safe");
  const pos = (x) => ((clamp(x, lo, hi) - lo) / (hi - lo)) * 100;
  safe.style.left = pos(safeLo) + "%";
  safe.style.width = pos(safeHi) - pos(safeLo) + "%";
  const mk = el("div", "meter-marker");
  mk.style.left = "-100px";
  track.append(safe, mk);
  const scale = el("div", "meter-scale");
  scale.append(el("span", null, fmt(lo, 0)), el("span", null, `safe ${fmt(safeLo, digits)}${safeHi < hi ? "–" + fmt(safeHi, 0) : "+"} ${unit}`), el("span", null, fmt(hi, 0)));
  m.append(top, track, scale);
  METERS[id] = { m, val, mk, pos, safeLo, safeHi, unit, digits };
}

function updateMeter(id, value) {
  const M = METERS[id];
  M.m.classList.toggle("bad", value !== null && (value < M.safeLo || value > M.safeHi));
  tween(M.val, value, M.digits, " " + M.unit);
  M.mk.style.left = value === null ? "-100px" : `calc(${M.pos(value)}% - 2px)`;
}

/* ---------- power chart ---------- */

function drawSpark(history) {
  const c = $("spark");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 120;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  const lo = 0, hi = 12000, y = (v) => h - 8 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * (h - 16);
  const T = meta.thresholds;
  ctx.fillStyle = "rgba(53,224,161,0.07)";
  ctx.fillRect(0, y(T.power_max), w, y(T.power_min) - y(T.power_max));
  ctx.strokeStyle = "rgba(53,224,161,0.35)"; ctx.setLineDash([4, 6]); ctx.lineWidth = 1;
  for (const v of [T.power_min, T.power_max]) { ctx.beginPath(); ctx.moveTo(0, y(v)); ctx.lineTo(w, y(v)); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = "#56657f"; ctx.font = "10.5px JetBrains Mono, monospace";
  ctx.fillText("9,000 W", 4, y(T.power_max) - 4);
  ctx.fillText("3,500 W", 4, y(T.power_min) + 13);
  if (!history.length) return;
  const n = 90, step = w / (n - 1), off = n - history.length;
  const xs = history.map((_, i) => (off + i) * step);
  const area = ctx.createLinearGradient(0, 0, 0, h);
  area.addColorStop(0, "rgba(47,217,244,0.35)"); area.addColorStop(1, "rgba(47,217,244,0)");
  ctx.beginPath();
  history.forEach((p, i) => (i ? ctx.lineTo(xs[i], y(p.power)) : ctx.moveTo(xs[i], y(p.power))));
  ctx.lineTo(xs[xs.length - 1], h); ctx.lineTo(xs[0], h); ctx.closePath();
  ctx.fillStyle = area; ctx.fill();
  const line = ctx.createLinearGradient(0, 0, w, 0);
  line.addColorStop(0, CYAN); line.addColorStop(1, VIOLET);
  ctx.strokeStyle = line; ctx.lineWidth = 2; ctx.shadowColor = CYAN; ctx.shadowBlur = 8;
  ctx.beginPath();
  history.forEach((p, i) => (i ? ctx.lineTo(xs[i], y(p.power)) : ctx.moveTo(xs[i], y(p.power))));
  ctx.stroke();
  history.forEach((p, i) => {
    if (p.power < T.power_min || p.power > T.power_max) {
      ctx.shadowColor = RED; ctx.shadowBlur = 14; ctx.fillStyle = RED;
      ctx.beginPath(); ctx.arc(xs[i], y(p.power), 4, 0, Math.PI * 2); ctx.fill();
    }
  });
  const last = history[history.length - 1];
  ctx.shadowColor = "#fff"; ctx.shadowBlur = 16; ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(xs[xs.length - 1] - 1, y(last.power), 4, 0, Math.PI * 2); ctx.fill();
  ctx.shadowBlur = 0;
}

/* ---------- risks ---------- */

const RISK = {};
function buildRisks() {
  const wrap = $("risks");
  for (const [code, name] of Object.entries(meta.modes)) {
    const row = el("div", "risk-row");
    const nm = el("span", "risk-name");
    const dot = el("i");
    dot.style.setProperty("--c", MODE_COLOR[code]);
    nm.append(dot, document.createTextNode(name.replace(" Failure", "")));
    const bar = el("div", "risk-bar");
    const fill = el("div", "risk-fill");
    bar.append(fill);
    const p = el("span", "risk-pct", "—");
    row.append(nm, bar, p);
    wrap.append(row);
    RISK[code] = { fill, p };
  }
}

function updateRisks(risk) {
  for (const code of Object.keys(RISK)) {
    const v = risk ? risk[code] : 0;
    const R = RISK[code];
    R.fill.className = "risk-fill" + (v >= 0.5 ? " high" : v >= 0.05 ? " mid" : "");
    R.fill.style.width = (risk ? Math.max(v * 100, 1.5) : 0) + "%";
    R.p.textContent = risk ? pct(v) : "—";
  }
}

/* ---------- alerts ---------- */

function updateAlerts(events) {
  $("alert-count").textContent = state.event_count;
  const wrap = $("alerts");
  const ids = events.map((e) => e.id).join("|");
  if (wrap._ids === ids) return;
  wrap._ids = ids;
  wrap.replaceChildren();
  if (!events.length) { wrap.append(el("p", "empty-note", "No failures detected yet.")); return; }
  for (const e of events) {
    const card = el("button", "alert-card " + e.mechanism + (rendered.has(e.id) ? "" : " enter"));
    rendered.add(e.id);
    const top = el("div", "alert-top");
    const left = el("span");
    left.append(el("span", "chip " + e.mechanism, e.mechanism), document.createTextNode(MODE_NAMES[e.mechanism]));
    top.append(left, el("span", "muted mono", time(e.timestamp).slice(11)));
    card.append(top, el("div", "alert-summary", e.summary), el("div", "alert-reason", e.reason), el("div", "alert-go", "Investigate →"));
    card.addEventListener("click", () => openDrawer(e));
    wrap.append(card);
  }
}

/* ---------- main render ---------- */

const LINK_LABEL = {
  live: "LIVE · MQTT",
  waiting: "MQTT · waiting for the machine",
  offline: "MQTT · machine offline",
  "broker-down": "MQTT · broker unreachable",
};
let lastRejected = 0;

function updateSource(s) {
  const mqtt = s.source.mode === "mqtt";
  document.querySelectorAll(".replay-only").forEach((e) => { e.hidden = mqtt; });
  $("source-badge").hidden = !mqtt;
  $("progress").hidden = mqtt;
  $("node-feed").querySelector(".sb").textContent = mqtt ? "MQTT stream" : "sensor stream";
  if (!mqtt) return;
  $("source-badge").className = "source-badge " + s.source.link;
  $("source-text").textContent = LINK_LABEL[s.source.link];
  $("source-counts").textContent = `${s.source.received.toLocaleString("en-IN")} received · ${s.source.rejected} rejected`;
  if (s.source.rejected > lastRejected && s.source.last_rejection) {
    const note = $("feed-note");
    note.textContent = "✗ Rejected a reading from the machine: " + s.source.last_rejection;
    note.hidden = false;
    clearTimeout(note._t);
    note._t = setTimeout(() => { note.hidden = true; }, 6000);
  }
  lastRejected = s.source.rejected;
}

function render(s) {
  state = s;
  const live = s.live;
  const mqtt = s.source.mode === "mqtt";
  const offline = !!live && s.status === "offline";
  const alert = !!live && s.status === "alert";
  updateSource(s);
  $("empty").hidden = !!live;
  $("empty-text").replaceChildren(...(mqtt
    ? [document.createTextNode("Waiting for the machine to send data over MQTT."), el("br"),
       document.createTextNode("Start it with "), el("strong", null, "python scripts/publish_machine.py")]
    : [document.createTextNode("Press "), el("strong", null, "Play"), document.createTextNode(" to start the machine feed,"), el("br"),
       document.createTextNode("or "), el("strong", null, "Jump to next failure"), document.createTextNode(".")]));
  $("clock").textContent = live ? time(live.timestamp) : "—";
  $("position").textContent = mqtt
    ? `${s.position.toLocaleString("en-IN")} received`
    : `${s.position.toLocaleString("en-IN")} / ${s.total.toLocaleString("en-IN")}`;
  if (!mqtt) $("progress-fill").style.width = (s.position / s.total) * 100 + "%";
  $("play").textContent = s.playing ? "❚❚ Pause" : "▶ Play";
  $("speed").value = String(s.speed);
  $("autopause").checked = s.pause_on_failure;
  document.body.classList.toggle("alarm", alert);

  const st = $("status");
  st.className = "status " + (!live ? "idle" : offline ? "offline" : alert ? "alert" : "running");
  $("status-text").textContent = !live ? "Waiting for data" : offline
    ? "OFFLINE · no data from the machine"
    : alert ? "ALERT · " + s.active_alerts.map((m) => MODE_NAMES[m]).join(" + ")
    : "Running normally";

  updateRig(offline ? null : live, alert);
  updateGauges(live ? live.values : null);
  updateMeter("meter-power", live ? live.derived.power : null);
  updateMeter("meter-tdiff", live ? live.derived.temperature_difference : null);
  drawSpark(s.history);
  updateRisks(live ? live.risk : null);
  updateAlerts(s.events);

  const fresh = s.new_event_ids.filter((id) => !seen.has(id));
  if (s.position > lastPosition) pulse(0, 3, fresh.length ? "bad" : "normal");
  lastPosition = s.position;
  s.events.forEach((e) => seen.add(e.id));
  if (fresh.length && !openEvent) {
    const e = s.events.find((x) => x.id === fresh[0]);
    $("toast-title").textContent = `Failure detected: ${MODE_NAMES[e.mechanism]}` + (fresh.length > 1 ? ` (+${fresh.length - 1} more)` : "");
    $("toast-body").textContent = e.summary;
    $("toast-open").onclick = () => openDrawer(e);
    $("toast").hidden = false;
  }
}

/* ---------- investigation drawer ---------- */

function drawPath(e) {
  const svg = $("d-path");
  svg.replaceChildren();
  const color = MODE_COLOR[e.mechanism];
  const top = e.causes[0] ? e.causes[0].signal : null;
  const signals = [...new Set(e.path.map((p) => p.signal))];
  const vias = [...new Set(e.path.map((p) => p.via).filter(Boolean))];
  const defs = svgEl("defs", {});
  const lg = svgEl("linearGradient", { id: "pg", x1: "0", y1: "0", x2: "1", y2: "0" });
  lg.append(svgEl("stop", { offset: "0", "stop-color": VIOLET }), svgEl("stop", { offset: "1", "stop-color": color }));
  defs.append(lg);
  svg.append(defs);
  const box = (x, y, w, text, style) => {
    const fills = { strong: "url(#pg)", mode: color, derived: "#1a2744", signal: "#131e34" };
    svg.append(svgEl("rect", { x, y, width: w, height: 30, rx: 9, fill: fills[style], stroke: style === "signal" || style === "derived" ? "rgba(125,160,255,0.3)" : "none" }));
    const t = svgEl("text", { x: x + w / 2, y: y + 19.5, "text-anchor": "middle", "font-size": 12, fill: style === "mode" ? "#0b1020" : "#e7eef9", "font-family": "Sora, -apple-system, Arial", "font-weight": style === "signal" ? 500 : 700 });
    t.textContent = text;
    svg.append(t);
  };
  const line = (x1, y1, x2, y2, strong) => {
    svg.append(svgEl("path", { d: `M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`, fill: "none",
      stroke: strong ? color : "rgba(125,160,255,0.28)", "stroke-width": strong ? 2.5 : 1.5, class: strong ? "flow" : "" }));
  };
  const colX = [4, 205, 400];
  const ys = (n, i) => 75 - (n - 1) * 21 + i * 42 - 15;
  const modeY = 60;
  const viaY = {};
  vias.forEach((v, i) => { viaY[v] = ys(vias.length, i); });
  signals.forEach((s, i) => {
    const y = ys(signals.length, i);
    const via = e.path.find((p) => p.signal === s).via;
    const strong = s === top;
    if (via) line(154, y + 15, colX[1], viaY[via] + 15, strong);
    else line(154, y + 15, colX[2], modeY + 15, strong);
  });
  vias.forEach((v) => line(colX[1] + 150, viaY[v] + 15, colX[2], modeY + 15, e.path.some((p) => p.via === v && p.signal === top)));
  signals.forEach((s, i) => box(colX[0], ys(signals.length, i), 150, LABEL[s].label, s === top ? "strong" : "signal"));
  vias.forEach((v) => box(colX[1], viaY[v], 150, LABEL[v].label + " (physics)", "derived"));
  box(colX[2], modeY, 132, { TWF: "Tool wear failure", HDF: "Heat failure", PWF: "Power failure", OSF: "Overstrain failure" }[e.mechanism], "mode");
}

function quickActions(e) {
  const v = e.values, acts = [];
  const causes = e.causes.map((c) => c.signal);
  if (causes.includes("torque")) {
    if (e.derived.power < meta.thresholds.power_min) acts.push(["Increase torque 20%", { torque: v.torque * 1.2 }]);
    else acts.push(["Reduce torque 20%", { torque: v.torque * 0.8 }]);
  }
  if (causes.includes("tool_wear")) acts.push(["Replace the tool", { tool_wear: 0 }]);
  if (causes.includes("rotational_speed") && e.mechanism === "HDF") acts.push(["Raise speed to 1,500 rpm", { rotational_speed: 1500 }]);
  return acts;
}

let simTimer = null;
let actionValues = null;
function currentInterventions() {
  if (actionValues) return { ...actionValues };
  const out = {};
  document.querySelectorAll("#d-sliders input[data-touched]").forEach((inp) => { out[inp.dataset.signal] = Number(inp.value); });
  return out;
}

function setSliders(e, values) {
  actionValues = Object.keys(values).length ? { ...values } : null;
  document.querySelectorAll("#d-sliders input").forEach((inp) => {
    const s = inp.dataset.signal;
    inp.value = s in values ? values[s] : e.values[s];
    delete inp.dataset.touched;
    inp.dispatchEvent(new Event("input"));
  });
}

function buildSliders(e) {
  const wrap = $("d-sliders");
  wrap.replaceChildren();
  for (const c of e.causes) {
    if (c.signal === "product_variant") continue;
    const r = meta.ranges[c.signal];
    const digits = c.signal === "rotational_speed" || c.signal === "tool_wear" ? 0 : 1;
    const row = el("div", "slider-row");
    const inp = el("input");
    Object.assign(inp, { type: "range", min: r.min, max: r.max, step: (r.max - r.min) / 400, value: e.values[c.signal] });
    inp.dataset.signal = c.signal;
    inp.setAttribute("aria-label", LABEL[c.signal].label);
    const val = el("div", "slider-val");
    const show = () => {
      const shown = actionValues && c.signal in actionValues ? actionValues[c.signal]
        : inp.dataset.touched ? Number(inp.value) : e.values[c.signal];
      val.replaceChildren(document.createTextNode(`${fmt(shown, digits)} ${LABEL[c.signal].unit}`), el("small", null, `was ${fmt(e.values[c.signal], digits)}`));
    };
    inp.addEventListener("input", (ev) => {
      if (ev.isTrusted) {
        if (actionValues) {
          document.querySelectorAll("#d-sliders input").forEach((o) => { if (o.dataset.signal in actionValues) o.dataset.touched = "1"; });
          actionValues = null;
        }
        inp.dataset.touched = "1";
      }
      show();
      clearTimeout(simTimer);
      simTimer = setTimeout(() => runSim(e), 140);
    });
    show();
    row.append(el("span", null, LABEL[c.signal].label), inp, val);
    wrap.append(row);
  }
}

async function runSim(e) {
  const res = $("d-result");
  const interventions = currentInterventions();
  if (!Object.keys(interventions).length) {
    res.replaceChildren(el("p", "hint-text", "Move a slider or pick an action above to test a change."));
    return;
  }
  let r;
  try { r = await post("/api/simulate", { event_id: e.id, interventions }); }
  catch (err) { res.replaceChildren(el("p", "bad", err.message)); return; }
  pulse(3, 4, "sim");
  const m = r.mechanism;
  const pair = el("div", "big-pair");
  const big = (label, before, after) => {
    const b = el("div", "big");
    const v = el("div", "val");
    const a = el("span", after < 0.5 ? "good" : "bad");
    v.append(el("span", null, pct(before)), el("span", "arrow", "→"), a);
    b.append(el("div", "label", label), v);
    const to = Math.round(after * 1000) / 10;
    tween(a, Math.round(before * 1000) / 10, 0, "%");
    tween(a, to, to > 0 && to < 1 ? 1 : 0, "%");
    return b;
  };
  pair.append(big(`This ${MODE_NAMES[m].toLowerCase()}`, r.risk_before[m], r.risk_after[m]), big("Any failure", r.any_before, r.any_after));
  const parts = [pair];
  const phys = [];
  for (const d of ["power", "temperature_difference"]) {
    if (Math.abs(r.derived_after[d] - r.derived_before[d]) > 1e-6) {
      phys.push(`${LABEL[d].label}: ${fmt(r.derived_before[d], d === "power" ? 0 : 2)} → ${fmt(r.derived_after[d], d === "power" ? 0 : 2)} ${LABEL[d].unit}`);
    }
  }
  if (phys.length) parts.push(el("div", "phys-change", phys.join("   ·   ")));
  let verdict, good;
  if (r.prevents_this && !r.still_failing.length) { verdict = "✓ This change prevents the failure" + (r.other_failures_before.length ? " and fixes every other active failure too" : ""); good = true; }
  else if (r.prevents_this) { verdict = `Prevents this failure, but ${r.still_failing.map((x) => MODE_NAMES[x].toLowerCase()).join(" and ")} is still likely`; good = false; }
  else if (r.risk_before[m] < 0.5 && r.any_after < r.any_before) { verdict = "Lowers the failure risk"; good = true; }
  else { verdict = "✗ The failure is still likely with this change"; good = false; }
  parts.push(el("div", "verdict " + (good ? "good" : "bad"), verdict));
  if (!r.trustworthy) parts.push(el("div", "trust", "⚠ Low confidence: this combination is outside the data the model learnt from, so treat the answer as a guess."));
  res.replaceChildren(...parts);
}

function openDrawer(e) {
  openEvent = e;
  actionValues = null;
  $("toast").hidden = true;
  $("d-chip").className = "chip " + e.mechanism;
  $("d-chip").textContent = e.mechanism;
  $("d-title").textContent = MODE_NAMES[e.mechanism];
  $("d-meta").textContent = `${time(e.timestamp)} · reading ${e.row.toLocaleString("en-IN")} · product variant ${e.values.product_variant}`;
  $("d-summary").textContent = e.summary;
  $("d-reason").textContent = "Rule triggered: " + e.reason;

  const causes = $("d-causes");
  causes.replaceChildren();
  const fills = [];
  for (const c of e.causes) {
    const row = el("div", "cause-row");
    const name = el("div", "cause-name", LABEL[c.signal].label);
    name.append(el("small", null, c.signal === "product_variant" ? `variant ${e.values.product_variant}` : `${fmt(c.value, c.signal === "rotational_speed" || c.signal === "tool_wear" ? 0 : 1)} ${LABEL[c.signal].unit} · ${c.direction} risk`));
    const bar = el("div", "cause-bar");
    const fill = el("div", "cause-fill");
    fill.style.setProperty("--c", MODE_COLOR[e.mechanism]);
    bar.append(fill);
    fills.push([fill, c.share]);
    const p = el("span", "cause-pct", "0%");
    row.append(name, bar, p);
    causes.append(row);
    tween(p, Math.round(c.share * 100), 0, "%");
  }
  requestAnimationFrame(() => requestAnimationFrame(() => fills.forEach(([f, s]) => { f.style.width = (s * 100).toFixed(1) + "%"; })));
  drawPath(e);

  const acts = $("d-actions");
  acts.replaceChildren();
  for (const [label, values] of quickActions(e)) {
    const b = el("button", "btn glow", label);
    b.addEventListener("click", () => setSliders(e, values));
    acts.append(b);
  }
  const back = el("button", "btn", "Undo changes");
  back.addEventListener("click", () => setSliders(e, {}));
  acts.append(back);

  buildSliders(e);
  runSim(e);
  $("drawer").hidden = false;
  $("scrim").hidden = false;
  pulse(2, 3, "bad");
}

function closeDrawer() {
  openEvent = null;
  $("drawer").hidden = true;
  $("scrim").hidden = true;
}

/* ---------- wiring ---------- */

async function init() {
  meta = await (await fetch("/api/meta")).json();
  $("machine").textContent = meta.machine;
  buildPipeline();
  buildRig();
  buildGauges();
  const T = meta.thresholds;
  buildMeter("meter-power", "Power", "W", 0, 12000, T.power_min, T.power_max, 0);
  buildMeter("meter-tdiff", "Temperature difference", "K", 7, 13, T.temperature_difference_min, 13, 2);
  buildRisks();
  $("play").onclick = () => control(state && state.playing ? "pause" : "play");
  $("next").onclick = () => control("next_failure");
  $("reset").onclick = () => { seen = new Set(); rendered = new Set(); lastPosition = 0; lastRejected = 0; closeDrawer(); $("toast").hidden = true; control("reset"); };
  $("speed").onchange = (ev) => control("speed", Number(ev.target.value));
  $("autopause").onchange = (ev) => control("pause_on_failure", ev.target.checked ? 1 : 0);
  $("d-close").onclick = closeDrawer;
  $("scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeDrawer(); });
  window.addEventListener("resize", () => state && drawSpark(state.history));

  render(await (await fetch("/api/state")).json());
  const src = new EventSource("/api/stream");
  src.onmessage = (m) => render(JSON.parse(m.data));
}

init();
