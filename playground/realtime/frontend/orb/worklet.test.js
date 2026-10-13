const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SAMPLE_RATE = 24000;
const QUANTUM = 128;
// What app.js configures for a 1 s unit: JITTER_MS (300 ms) as target and prebuffer.
const JITTER_SAMPLES = 7200;
const UNIT_SAMPLES = 24000;
const LEVEL = 0.5;
// 5 ms: a faded stop or restart moves at most LEVEL / FADE_FRAMES per sample.
const FADE_FRAMES = 120;
const SILENCE = 1e-3;

function makePlayer() {
  const reports = [];
  const port = { onmessage: null, postMessage: (message) => reports.push(message) };
  const processors = new Map();
  const scope = vm.createContext({
    sampleRate: SAMPLE_RATE,
    currentFrame: 0,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = port;
      }
    },
    registerProcessor: (name, processor) => processors.set(name, processor),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "worklet.js"), "utf8"), scope);
  const processor = new (processors.get("playback-processor"))({ processorOptions: { initialSamples: JITTER_SAMPLES } });
  port.onmessage({ data: { type: "config", prebufferSamples: JITTER_SAMPLES, unitSamples: UNIT_SAMPLES } });
  return { scope, port, processor, reports };
}

function makeUnit() {
  return { type: "push", samples: new Float32Array(UNIT_SAMPLES).fill(LEVEL) };
}

// Each [seconds, message] is delivered before the first 128-frame quantum at or after it.
function render(player, schedule, seconds) {
  const output = new Float32Array(Math.ceil((seconds * SAMPLE_RATE) / QUANTUM) * QUANTUM);
  let next = 0;
  for (let frame = 0; frame < output.length; frame += QUANTUM) {
    player.scope.currentFrame = frame;
    while (next < schedule.length && schedule[next][0] * SAMPLE_RATE <= frame) player.port.onmessage({ data: schedule[next++][1] });
    player.processor.process([[]], [[output.subarray(frame, frame + QUANTUM)]]);
  }
  return output;
}

function findAudibleSpan(output) {
  let first = -1;
  let last = -1;
  output.forEach((value, index) => {
    if (Math.abs(value) >= SILENCE) {
      if (first < 0) first = index;
      last = index;
    }
  });
  return [first, last];
}

function countSilentFrames(output, first, last) {
  return output.subarray(first, last + 1).filter((value) => Math.abs(value) < SILENCE).length;
}

function findLargestStep(output) {
  return output.reduce((largest, value, index) => (index ? Math.max(largest, Math.abs(value - output[index - 1])) : largest), 0);
}

test("whole units up to the prebuffer late play back to back at normal speed", () => {
  const player = makePlayer();
  const arrivals = [0, 1.29, 2.05, 3.2, 4.29];
  const schedule = [[0, { type: "open" }], ...arrivals.map((seconds) => [seconds, makeUnit()]), [4.29, { type: "flush" }]];

  const output = render(player, schedule, 6);

  const [first, last] = findAudibleSpan(output);
  assert.equal(countSilentFrames(output, first, last), 0, "no gap between units");
  assert.ok(Math.abs(last + 1 - first - arrivals.length * UNIT_SAMPLES) <= FADE_FRAMES, `one queued unit is not backlog: ${last + 1 - first} frames heard`);
  assert.ok(Math.abs(first - JITTER_SAMPLES) <= QUANTUM, `starts a prebuffer after the first unit, not on arrival (frame ${first})`);
});

test("a unit later than the prebuffer stalls until it arrives without a click and is reported on flush", () => {
  const player = makePlayer();
  const schedule = [[0, { type: "open" }], [0, makeUnit()], [1.6, makeUnit()], [2.5, makeUnit()], [2.5, { type: "flush" }]];

  const output = render(player, schedule, 4.5);

  const silentFrames = countSilentFrames(output, ...findAudibleSpan(output));
  assert.ok(Math.abs(silentFrames / SAMPLE_RATE - 0.3) <= 0.01, `resumes when the late unit arrives, 1.6 s (silent ${silentFrames} frames)`);
  assert.ok(findLargestStep(output) <= (LEVEL / FADE_FRAMES) * 1.01, "stops and restarts are faded");
  const reports = player.reports.filter((message) => message.type === "response");
  assert.deepEqual(reports.map((report) => report.underruns), [1], "the response reports one underrun");
  assert.ok(Math.abs(reports[0].underrunMs - 300) <= 6, `underrun from 1.3 s to 1.6 s (${reports[0].underrunMs} ms)`);
});

test("a trim that drops nothing leaves playback untouched", () => {
  const player = makePlayer();
  const schedule = [[0, { type: "open" }], [0, makeUnit()], [0.6, { type: "trim", keepSamples: UNIT_SAMPLES }], [0.6, { type: "flush" }]];

  const output = render(player, schedule, 2);

  const [first, last] = findAudibleSpan(output);
  assert.equal(countSilentFrames(output, first, last), 0, "no gap");
  assert.ok(findLargestStep(output) <= (LEVEL / FADE_FRAMES) * 1.01, "no click");
});
