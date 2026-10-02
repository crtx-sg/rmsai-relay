// Drive the companion app's session-expiry path against a fake gateway (no browser, no LiveKit).
// Usage: node tests/js/reauth_smoke.js <app.js>
// Prints one JSON object of observations; tests/test_app_reauth.py asserts on it.
const fs = require("fs"), vm = require("vm");

const els = {};
const listeners = {};  // id -> {event: [fn]}
const mk = (id) => els[id] || (els[id] = {
  id, innerHTML: "", textContent: "", value: "", options: [], children: [], dataset: {},
  returnValue: "", open: false,
  add(o) { this.options.push(o); },
  addEventListener(ev, fn) { ((listeners[id] ||= {})[ev] ||= []).push(fn); },
  removeEventListener(ev, fn) { const l = (listeners[id] || {})[ev] || []; const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1); },
  appendChild(c) { this.children.push(c); }, focus() {},
  showModal() { this.open = true; shown += 1; },
  close(v) { this.open = false; this.returnValue = v || ""; fire(id, "close", {}); },
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
});
const fire = (id, ev, e) => [...(((listeners[id] || {})[ev]) || [])].forEach((fn) => fn(e));
let shown = 0;

// Fake gateway: tokens "old" (expired) and "new"; PIN 1234.
const calls = [];
let sessionCalls = 0;
const fetch = async (path, opts) => {
  const body = JSON.parse(opts.body);
  calls.push({ path, token: body.session || null });
  const reply = (status, json) => ({ ok: status < 400, status, json: async () => json });
  if (path === "/session") {
    sessionCalls += 1;
    return body.pin === "1234" ? reply(200, { token: "new" }) : reply(401, {});
  }
  return body.session === "new" ? reply(200, { url: "/artifact/t", ok: true }) : reply(401, {});
};

const ctx = {
  console: { log() {}, warn() {}, error() {} }, fetch, setTimeout, clearTimeout, Date, Promise,
  window: { LivekitClient: {} },
  document: {
    documentElement: { dataset: {} },
    getElementById: mk, querySelector: (q) => mk(q), querySelectorAll: () => [],
    createElement: () => ({ className: "", dataset: {}, innerHTML: "" }),
  },
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
vm.runInContext(`session = { token: "old", room: "rmsai-inbox-h1" };`, ctx);

const tick = () => new Promise((r) => setTimeout(r, 0));
const submit = async (pin, value = "ok") => {
  mk("reauth-pin").value = pin;
  fire("reauth-form", "submit", { submitter: { value }, preventDefault() {} });
  await tick(); await tick();
};

(async () => {
  const out = {};

  // 1. two concurrent calls hit 401 -> ONE prompt; a wrong PIN keeps it open; the right PIN retries both
  const a = ctx.sessionPost("/artifact-link", { event_id: "e1", kind: "report" });
  const b = ctx.sessionPost("/event-info", { event_id: "e1" });
  await tick(); await tick();
  out.promptsShown = shown;
  await submit("0000");
  out.wrongPinMessage = mk("reauth-err").textContent;
  out.openAfterWrongPin = mk("reauth").open;
  await submit("1234");
  const [ra, rb] = await Promise.all([a, b]);
  out.statuses = [ra.status, rb.status];
  out.sessionCalls = sessionCalls;
  out.retriedWithNewToken = calls.filter((c) => c.path !== "/session").map((c) => c.token);
  out.tokenNow = vm.runInContext("session.token", ctx);

  // 2. expired again, clinician cancels -> the caller gets the 401 back, no extra /session call
  vm.runInContext(`session.token = "old";`, ctx);
  const before = sessionCalls;
  const c = ctx.sessionPost("/ack", { event_id: "e1" });
  await tick(); await tick();
  mk("reauth").close("cancel");
  out.cancelStatus = (await c).status;
  out.sessionCallsOnCancel = sessionCalls - before;

  process.stdout.write(JSON.stringify(out));
})();
