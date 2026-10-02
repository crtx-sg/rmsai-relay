// Render the companion app's pure HTML builders against a stub DOM (no browser, no LiveKit).
// Usage: node tests/js/render_smoke.js <app.js> <metrics.json> <info.json> <row.json>
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

ctx.renderPerf(JSON.parse(fs.readFileSync(metricsPath)));
const models = mk("perf-models").innerHTML;
const rows = mk("#perf-events tbody").innerHTML;
const info = ctx.renderInfo(JSON.parse(fs.readFileSync(infoPath)));
vm.runInContext(`applyMessage(state, ${fs.readFileSync(rowPath)}); render();`, ctx);
const worklistRow = mk("rows").children.map((c) => c.innerHTML).join("");

// theme toggle: no localStorage in this sandbox, so the toggle must still work (session only)
const before = vm.runInContext("currentTheme()", ctx);
vm.runInContext("toggleTheme()", ctx);
const after = ctx.document.documentElement.dataset.theme;

const all = models + rows + info + worklistRow;
process.stdout.write(JSON.stringify({
  models, rows, info, worklistRow,
  datasetOptions: mk("f-dataset").options.map((o) => o.value),
  theme: { before, after, button: mk("theme-btn").textContent },
  leaks: all.includes("${") || all.includes("undefined"),
}));
