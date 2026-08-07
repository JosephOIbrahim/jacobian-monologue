import { readFileSync } from "fs";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!DOCTYPE html><body></body>", { pretendToBeVisual: true });
global.window = dom.window;
global.document = dom.window.document;
global.HTMLElement = dom.window.HTMLElement;
global.SVGElement = dom.window.SVGElement;

const md = readFileSync("README.md", "utf8");
const re = /```mermaid\n([\s\S]*?)```/g;
const blocks = [...md.matchAll(re)].map((m) => m[1].trim());

const mermaid = (await import("mermaid")).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });

let failed = 0;
for (let i = 0; i < blocks.length; i++) {
  try {
    const res = await mermaid.parse(blocks[i]);
    console.log(`block ${i + 1}: PARSE OK  (diagramType=${res?.diagramType ?? "flowchart"})`);
  } catch (e) {
    failed++;
    console.log(`block ${i + 1}: PARSE FAILED`);
    console.log("  " + String(e.message || e).split("\n").slice(0, 2).join(" | "));
  }
}
console.log(failed === 0 ? "\nALL BLOCKS PARSE - GitHub will render them" : `\n${failed} FAILED`);
process.exit(failed === 0 ? 0 : 1);
