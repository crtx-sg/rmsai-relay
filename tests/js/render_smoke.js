// Render the companion app's pure HTML builders against a stub DOM (no browser, no LiveKit).
// Usage: node tests/js/render_smoke.js <app.js> <events.json> <info.json> <row.json>
// Prints one JSON object of checks; tests/test_app_render.py asserts on it.
const fs = require("fs"), vm = require("vm");
const [appPath, metricsPath, infoPath, rowPath] = process.argv.slice(2);

const els = {};
const mk = (id) => els[id] || (els[id] = {
  id, innerHTML: "", textContent: "", value: "", options: [], children: [], dataset: {},
  add(o) { this.options.push(o); }, addEventListener() {}, appendChild(c) { this.children.push(c); },
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
});
const ctx = {
  console: { log() {}, warn() {}, error() {} }, Option: function (t, v) { this.text = t; this.value = v; },
  window: { LivekitClient: {} }, setTimeout, clearTimeout, Date,
  document: {
    documentElement: { dataset: {} },
    getElementById: mk, querySelector: (q) => mk(q), querySelectorAll: () => [],
    createElement: () => ({ className: "", dataset: {}, innerHTML: "" }),
  },
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(appPath, "utf8"), ctx);

ctx.renderEvents(JSON.parse(fs.readFileSync(metricsPath)), true);  // the Events tab, labelled mode
const models = mk("perf-models").innerHTML;
const rows = mk("#perf-events tbody").innerHTML;
const info = ctx.renderInfo(JSON.parse(fs.readFileSync(infoPath)));
vm.runInContext(`applyMessage(state, ${fs.readFileSync(rowPath)}); render();`, ctx);
const worklistRow = mk("rows").children.map((c) => c.innerHTML).join("");

// theme toggle: no localStorage in this sandbox, so the toggle must still work (session only)
const before = vm.runInContext("currentTheme()", ctx);
vm.runInContext("toggleTheme()", ctx);
const after = ctx.document.documentElement.dataset.theme;

// Simulate tab: run separators + the results panel
const runs = vm.runInContext(`(() => {
  const rs = [{ event_id: "a", ts: 5, run: "r2-1402-n2-fp0" }, { event_id: "b", ts: 9 },
              { event_id: "c", ts: 7, run: "r2-1402-n2-fp0" }, { event_id: "d", ts: 1 }];
  const order = groupByRun(rs, "ts");
  return { order: order.map((r) => r.event_id),
           seps: order.map((r, i) => runSep(order[i - 1], r, 8)).filter(Boolean).length,
           label: runLabel("r3-1402-n10-fp20"), cli: runLabel(undefined) };
})()`, ctx);
ctx.renderSimulation({ run: "r1-0905-n2-fp50", summary: { expected_calls: 1, tp_calls: 1, fp_suppressed: 1 },
  events: [{ patient: "PT935761", ground_truth: "NORMAL_SINUS", event_type: "NORMAL_SINUS",
             confidence: 0.85, gate: "false_positive", call: false }] });
const sim = mk("sim-summary").innerHTML + mk("#sim-events tbody").innerHTML;

// Simulate history: one labelled bar per run + totals and rates
ctx.renderSimHistory({ runs: [{ run: "r5-1102-n4-fp50", hospitals: { h1: 1, h2: 3 },
  summary: { expected_calls: 2, tp_calls: 2, fp_suppressed: 2 } }],
  totals: { expected_calls: 2, tp_calls: 2, fp_suppressed: 2 },
  rates: { rang: { value: 1, low: 0.34, high: 1, k: 2, n: 2 }, fp_filtered: null } });
const hist = mk("sim-rates").innerHTML + mk("sim-totals").innerHTML + mk("#sim-history tbody").innerHTML;

// Header: which hospital, with links to the others' apps
ctx.renderHospitalBadge({ hospital_id: "h1",
  hospital_apps: { h1: "http://localhost:8080", h2: "http://localhost:8090" } });
const badge = mk("hospital-badge").innerHTML;

const all = models + rows + info + worklistRow + sim + badge + hist;
process.stdout.write(JSON.stringify({
  models, rows, info, worklistRow, runs, sim, badge, hist, title: ctx.document.title,
  datasetOptions: mk("f-dataset").options.map((o) => o.value),
  theme: { before, after, button: mk("theme-btn").textContent },
  leaks: all.includes("${") || all.includes("undefined"),
}));
