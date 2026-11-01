// The demo: speak the text, and light each word while it is heard.
import { Phonon } from "../src/index";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $("status");
const t0 = performance.now();
const tts = await Phonon.load({ progress: (p) => (status.textContent = `Downloading the model: ${p}%`) });
status.textContent = `Ready in ${Math.round(performance.now() - t0)} ms.`;
const voice = $<HTMLSelectElement>("voice");
for (const v of tts.voices) voice.add(new Option(`${v.name} (${v.locale}, ${v.gender})`, v.name));
voice.value = "Delia";
const button = $<HTMLButtonElement>("speak");
button.disabled = false;
let ctx: AudioContext | null = null;

button.onclick = async () => {
  ctx ??= new AudioContext({ sampleRate: 24000 });
  button.disabled = true;
  const start = performance.now();
  let first = 0;
  const speech = await tts.speak($<HTMLTextAreaElement>("text").value, {
    voice: voice.value,
    speed: Number($<HTMLInputElement>("speed").value),
    onAudio: () => (first ||= performance.now() - start),
  });
  const secs = speech.audio.length / speech.sampleRate;
  const took = performance.now() - start;
  status.textContent = `First audio in ${first.toFixed(1)} ms. ${secs.toFixed(1)} s of speech in ${took.toFixed(0)} ms (${(secs / (took / 1000)).toFixed(0)}× real time).`;
  const words = $("words");
  words.replaceChildren(
    ...speech.words.flatMap((w) => {
      const s = document.createElement("span");
      s.textContent = w.text;
      s.title = `${w.start.toFixed(2)}–${w.end.toFixed(2)} s`;
      return [s, document.createTextNode(" ")];
    }),
  );
  const buffer = ctx.createBuffer(1, speech.audio.length, speech.sampleRate);
  buffer.copyToChannel(speech.audio as Float32Array<ArrayBuffer>, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  const at = ctx.currentTime + 0.05;
  source.start(at);
  const spans = [...words.querySelectorAll("span")];
  const tick = () => {
    const t = ctx!.currentTime - at;
    spans.forEach((s, i) => s.classList.toggle("now", t >= speech.words[i].start && t < speech.words[i].end));
    if (t < secs) requestAnimationFrame(tick);
    else button.disabled = false;
  };
  requestAnimationFrame(tick);
};
// For tests: speak without playing.
Object.assign(globalThis, { phonon: tts });
