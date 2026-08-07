const fs = require("fs");
const md = fs.readFileSync("README.md", "utf8");
const re = /```mermaid\n([\s\S]*?)```/g;
const blocks = [...md.matchAll(re)].map((m) => m[1]);
console.log("mermaid blocks found:", blocks.length);
let bad = 0;
blocks.forEach((b, i) => {
  const lines = b.trim().split("\n");
  const head = lines[0].trim();
  const arrows = (b.match(/-->/g) || []).length;
  const styles = (b.match(/style /g) || []).length;
  const ob = (b.match(/\[/g) || []).length,
    cb = (b.match(/\]/g) || []).length;
  const oc = (b.match(/\(/g) || []).length,
    cc = (b.match(/\)/g) || []).length;
  const oq = (b.match(/\{/g) || []).length,
    cq = (b.match(/\}/g) || []).length;
  const okB = ob === cb,
    okC = oc === cc,
    okQ = oq === cq;
  if (!okB || !okC || !okQ) bad++;
  console.log(`\nblock ${i + 1}: ${head}`);
  console.log(`  arrows=${arrows} styles=${styles}`);
  console.log(`  brackets [ ]=${ob}/${cb} ${okB ? "ok" : "MISMATCH"}`);
  console.log(`  parens  ( )=${oc}/${cc} ${okC ? "ok" : "MISMATCH"}`);
  console.log(`  braces  { }=${oq}/${cq} ${okQ ? "ok" : "MISMATCH"}`);
});
console.log(bad === 0 ? "\nALL BALANCED" : `\n${bad} BLOCK(S) WITH MISMATCH`);
