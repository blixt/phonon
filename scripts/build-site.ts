// The web page (docs/index.html): site/index.html with the README's text laid out in it.
// The README's tagline heads the page, and each of its "##" sections follows the first screen. The site's own charts replace the README's figures.
import { readFileSync, writeFileSync } from "node:fs";
import { marked } from "marked";

const REPO = "https://github.com/blixt/phonon";
/** README figures the page draws itself (site/site.ts), or leaves out: the first screen's chart
 *  shows speed, first audio and cold start. */
const CHARTS: Record<string, string | null> = {
  speed: null,
  "first-audio": null,
  "cold-start": null,
  "speed-by-length": "length",
  "word-timing": "timing",
  quality: "quality",
};

const slug = (s: string) =>
  s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const text = (html: string) => html.replace(/<[^>]+>/g, "");

function render(md: string): string {
  let html = marked.parse(md, { async: false });
  // Figures.
  html = html.replace(/<p><img src="docs\/figures\/([\w-]+)\.svg" alt="([^"]*)"\s*\/?><\/p>/g, (_, name: string, alt: string) => {
    const chart = CHARTS[name];
    if (chart === undefined) throw new Error(`No chart for the figure ${name}`);
    return chart ? `<figure class="chart-box" data-chart="${chart}" aria-label="${alt}"></figure>` : "";
  });
  // Links into the repository, and to this page.
  html = html.replace(/href="([^"#:]+)"/g, (_, path: string) => {
    const dir = !/\.\w+$/.test(path) || path === "docs/audio";
    return `href="${REPO}/${dir ? "tree" : "blob"}/main/${path}"`;
  });
  html = html.replace(/href="https:\/\/blixt\.xyz\/phonon\/?"/g, 'href="#demo"');
  html = html.replace(/href="https:\/\/blixt\.xyz\/phonon\/listen\.html"/g, 'href="listen.html"');
  html = html.replace(/<table>/g, '<div class="table"><table>').replace(/<\/table>/g, "</table></div>");
  html = html.replace(/<h3>(.*?)<\/h3>/g, (_, t: string) => `<h3 id="${slug(t)}">${t}</h3>`);
  return html;
}

const readme = readFileSync("README.md", "utf8");
const [head, ...parts] = readme.split(/^## /m);
const paragraphs = head.split(/\n{2,}/).map((p) => p.trim()).filter((p) => p && !p.startsWith("#"));
const [tagline] = paragraphs;
const sections = parts.map((part) => {
  const title = part.slice(0, part.indexOf("\n")).trim();
  const body = render(part.slice(part.indexOf("\n")));
  const subs = [...body.matchAll(/<h3 id="([^"]+)">(.*?)<\/h3>/g)];
  const toc = subs.length
    ? `<ul class="toc">${subs.map(([, id, t]) => `<li><a href="#${id}">${t}</a></li>`).join("")}</ul>`
    : "";
  const listen = title === "Listen" ? '<div class="listen" id="listen-players"></div>' : "";
  return `<section class="doc" id="${slug(title)}"><div><h2>${title}</h2>${toc}</div><div class="body">${body}${listen}</div></section>`;
});

const taglineHtml = marked.parseInline(tagline, { async: false });
const page = readFileSync("site/index.html", "utf8")
  .replace("{{tagline-text}}", text(taglineHtml))
  .replace("{{tagline}}", taglineHtml)
  .replace("{{sections}}", sections.join("\n"));
writeFileSync("docs/index.html", page);
console.log(`docs/index.html: ${sections.length} sections`);
