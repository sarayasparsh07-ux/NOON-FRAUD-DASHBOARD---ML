// Builds reports/Phase2_Report.docx from reports/phase2_report.md + the Phase 2 figures in outputs/phase2/.
// Usage: node reports/build_word_report_phase2.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, HeadingLevel, Footer, PageNumber, LevelFormat,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const NAVY = "1B2A4A", INK = "2B3445", MUTED = "6B7385", AMBER = "B45309", GRID = "D5DAE3";
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1000;
const CONTENT_W = PAGE_W - 2 * MARGIN;
const IMG_W = 640;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Consolas", size: (base.size || 19) - 1 }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function figure(file, caption, maxH = 820, widthPx = IMG_W) {
  const b = fs.readFileSync(path.join(ROOT, file));
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  let width = widthPx, height = Math.round(widthPx * h / w);
  if (height > maxH) { height = maxH; width = Math.round(maxH * w / h); }   // keep tall charts on one page
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 20 },
      children: [new ImageRun({ type: "png", data: b, transformation: { width, height } })] }),
    new Paragraph({ spacing: { after: 160 }, children: runs(caption, { size: 16, color: MUTED, italics: true }) }),
  ];
}

// Figures placed at the end of the named section
const FIGS = {
  "Data checked before any comparison": [
    ["outputs/phase2/data_quality/B3_noon_duplicate_multiplicity.png",
     "Figure 1 · Noon's file: how many times each distinct row appears. The observed counts match sampling with replacement (dashed line)."]],
  "2.1 Distributions": [
    ["outputs/phase2/distributions/2.1b_brief_features.png",
     "Figure 2 · The three features named in the brief, with shared bins (top) and cumulative distributions (bottom). Dashed line: Noon's file as supplied, with duplicates; it is almost identical to the unique rows."],
    ["outputs/phase2/distributions/2.1d_other_features.png",
     "Figure 3 · The remaining features, with 95% Wilson intervals for shares."],
    ["outputs/phase2/distributions/2.1e_effect_sizes.png",
     "Figure 4 · Effect size per feature for three pairs. Diamonds: the same CTGAN-vs-Noon comparison with Noon's duplicates kept.", 470],
    ["outputs/phase2/relationships/2.1f_relationship_profiles.png",
     "Figure 5 · Conditional relationship profiles (unique rows; groups with fewer than 20 rows are not plotted)."]],
  "2.3 Lie detector": [
    ["outputs/phase2/lie_detector/2.3_lie_detector_protocols.png",
     "Figure 6 · Lie-detector protocols. Solid bars: accuracy; hatched bars: ROC-AUC; amber line: accuracy of always guessing the larger source."],
    ["outputs/phase2/lie_detector/2.3_permutation_test_and_importance.png",
     "Figure 7 · Primary protocol: permutation null distribution (left) and held-out permutation importance (right)."]],
};

const md = fs.readFileSync(path.join(__dirname, "phase2_report.md"), "utf8").split("\n");
const children = [];
let tableLines = [];
let section = null;

function flushTable() {
  if (!tableLines.length) return;
  const rows = tableLines.filter(l => !/^\|\s*-/.test(l)).map(l => l.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const nCol = rows[0].length;
  const firstW = nCol === 2 ? 2900 : 3300;
  const otherW = Math.floor((CONTENT_W - firstW) / (nCol - 1));
  const widths = [firstW, ...Array(nCol - 1).fill(otherW)];
  widths[nCol - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: GRID };
  children.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: NAVY }
               : (/Primary/.test(r[0]) ? { type: ShadingType.CLEAR, color: "auto", fill: "FBF1E6" } : undefined),
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({ alignment: (j === 0 || c.length > 30) ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(c, { size: 16.5, color: i === 0 ? "FFFFFF" : INK, bold: i === 0 }) })],
      })),
    })),
  }), new Paragraph({ spacing: { after: 80 }, children: [] }));
  tableLines = [];
}

function closeSection() {
  flushTable();
  if (section && FIGS[section]) for (const [f, cap, mh] of FIGS[section]) children.push(...figure(f, cap, mh || 820));
}

for (const line of md) {
  if (line.startsWith("|")) { tableLines.push(line); continue; }
  flushTable();
  if (!line.trim()) continue;
  if (line.startsWith("# ")) {
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { after: 40 },
      children: [new TextRun({ text: line.slice(2), bold: true, size: 32, color: NAVY, font: FONT })] }));
  } else if (line.startsWith("## ")) {
    closeSection();
    section = line.slice(3).trim();
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: section, bold: true, size: 24, color: NAVY, font: FONT })] }));
  } else if (line.startsWith("**Sparsh")) {
    children.push(new Paragraph({ spacing: { after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: AMBER, space: 4 } },
      children: runs(line.replace(/\*\*/g, ""), { size: 18, color: MUTED }) }));
  } else if (/^\s*- /.test(line)) {
    const level = line.match(/^\s*/)[0].length >= 2 ? 1 : 0;
    children.push(new Paragraph({ numbering: { reference: "bul", level }, spacing: { after: 50, line: 252 },
      children: runs(line.replace(/^\s*- /, ""), { size: 19, color: INK }) }));
  } else {
    children.push(new Paragraph({ spacing: { after: 90, line: 252 },
      children: runs(line, { size: 19, color: INK }) }));
  }
}
closeSection();

const doc = new Document({
  creator: "Sparsh Saraya", title: "Phase 2 Report — My GAN vs Noon's GAN",
  styles: { default: { document: { run: { font: FONT, size: 19, color: INK } } } },
  numbering: { config: [{ reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } },
  ] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 900, bottom: 900, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Noon fraud-detection challenge · Phase 2 · page ", size: 15, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "Phase2_Report.docx"), buf);
  console.log("wrote", path.join(__dirname, "Phase2_Report.docx"));
});
