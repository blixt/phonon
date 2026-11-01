// Runs the benchmark: serves the repository and plans each device's runs. Open
// `/bench/?device=NAME` on a device (over https, or localhost: WebGPU needs a secure page).
// First, one first visit per runtime: a fresh page that downloads the model (use a new browser
// profile, so nothing is cached). Then 10 rounds of cold runs and 10 rounds of warm runs. Each
// round runs every runtime once, in an order that rotates by round, so slow drifts in machine
// load spread over all runtimes. Each warm round uses its own seed. Results go to
// bench/results/NAME.jsonl, and every warm clip of an --audio device to
// bench/results/audio/NAME/RUNTIME/ID~ROUND.wav.
// Usage: node bench/run.mjs [--port 7893] [--runs 10] [--runtimes ours,gradiumWebgpu,...] [--audio mac]
import { appendFileSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { serve } from "./serve.mjs";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i < 0 ? fallback : process.argv[i + 1];
};
const here = dirname(new URL(import.meta.url).pathname);
const root = join(here, "..");
const runs = Number(arg("--runs", "10"));
const runtimes = arg("--runtimes", "ours,gradiumWebgpu,gradiumCpu,kokoro").split(",");
const audioDevices = new Set(arg("--audio", "mac").split(","));
const { phrases, cold } = JSON.parse(readFileSync(join(here, "phrases.json"), "utf8"));
const coldPhrase = phrases.filter((p) => p.id === cold);
const out = join(here, "results");
mkdirSync(out, { recursive: true });

/** Each device's plan and where it stands. */
const devices = new Map();
function plan(device) {
  const tasks = [];
  const order = (r) => runtimes.map((_, i) => runtimes[(i + r) % runtimes.length]);
  for (const runtime of runtimes) tasks.push({ runtime, kind: "cold", run: 0, runs, phrases: coldPhrase, prime: true });
  for (let r = 1; r <= runs; r++)
    for (const runtime of order(r)) tasks.push({ runtime, kind: "cold", run: r, runs, phrases: coldPhrase });
  for (let r = 1; r <= runs; r++)
    for (const runtime of order(r))
      tasks.push({ runtime, kind: "warm", run: r, runs, phrases, seed: r, saveAudio: audioDevices.has(device) });
  return tasks.map((t, index) => ({ ...t, id: `${device}-${index}`, index, total: tasks.length }));
}
const record = (device, row) => appendFileSync(join(out, `${device}.jsonl`), JSON.stringify({ at: new Date().toISOString(), ...row }) + "\n");

function next(device, hello) {
  let d = devices.get(device);
  if (!d) {
    d = { tasks: plan(device), at: 0, tries: 0, failed: new Set() };
    devices.set(device, d);
    record(device, { hello });
  }
  const skip = () => {
    while (d.at < d.tasks.length && d.failed.has(d.tasks[d.at].runtime)) {
      record(device, { task: d.tasks[d.at], skipped: true });
      d.at++;
    }
  };
  if (hello.done) {
    const task = d.tasks[d.at];
    if (task && hello.done.id === task.id) {
      record(device, { task: { ...task, phrases: task.phrases.map((p) => p.id) }, result: hello.done });
      // A runtime that fails twice in a row on a device is skipped there.
      d.failures = hello.done.error ? (d.failures ?? 0) + 1 : 0;
      if (d.failures >= 2) d.failed.add(task.runtime);
      d.at++;
      d.tries = 0;
    }
  } else if (d.tries >= 3) {
    // The page came back without a result three times: it crashed running this task.
    const task = d.tasks[d.at];
    record(device, { task: { ...task, phrases: task.phrases.map((p) => p.id) }, result: { error: "The page reloaded three times without a result." } });
    d.failed.add(task.runtime);
    d.at++;
    d.tries = 0;
  }
  skip();
  if (d.at >= d.tasks.length) return { kind: "done" };
  if (!hello.done) d.tries++;
  const { prime, ...task } = d.tasks[d.at];
  console.log(device, task.index + 1, "/", task.total, task.runtime, task.kind, task.run, prime ? "(priming)" : "");
  return task;
}

const port = Number(arg("--port", "7893"));
await serve({
  root,
  port,
  onPost(path, body) {
    if (path === "/next") return next(body.device, body);
    if (path === "/audio") {
      const dir = join(out, "audio", body.device, body.runtime);
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, `${body.id}~${body.run}.wav`), Buffer.from(body.wav, "base64"));
      return {};
    }
  },
});
console.log(`serving on http://127.0.0.1:${port}/bench/?device=NAME`);
