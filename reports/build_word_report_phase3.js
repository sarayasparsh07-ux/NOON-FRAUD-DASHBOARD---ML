// Builds reports/Phase3_Report.docx from reports/phase3_report.md + the Phase 3 figures in outputs/phase3/.
// Usage: node reports/build_word_report_phase3.js
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
  "Steps 3.1 and 3.2 — Results on the real referee set": [
    ["outputs/phase3/charts/3.1_3.2_confusion_matrices.png",
     "Figure 1 · Confusion matrices on the same 20,000 referee transactions (65 real frauds)."]],
  "Step 3.3 — The final comparison": [
    ["outputs/phase3/charts/3.3_final_bar_chart_recall_and_false_positives.png",
     "Figure 2 · The final bar chart: real frauds caught (left, with 95% Wilson intervals) and legitimate orders wrongly flagged (right)."],
    ["outputs/phase3/charts/3.4_fraud_detection_overlap.png",
     "Figure 3 · The 65 real frauds, by which model caught them."],
    ["outputs/phase3/charts/3.3_precision_recall_and_roc.png",
     "Figure 4 · Precision–recall and ROC curves (descriptive). Dots mark each model's frozen operating point; no threshold was chosen from these curves."]],
  "Step 3.4 — Explaining the gap": [
    ["outputs/phase3/charts/3.4_real_fraud_vs_generators_profiles.png",
     "Figure 5 · What real fraud looks like (n = 65) compared with my seed fraud, my CTGAN fraud and Noon's synthetic fraud."]],
};

const md = fs.readFileSync(path.join(__dirname, "phase3_report.md"), "utf8").split("\n");
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
               : (/My selected/.test(r[0]) ? { type: ShadingType.CLEAR, color: "auto", fill: "FBF1E6" } : undefined),
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
  if (line.trim().startsWith("|")) { tableLines.push(line.trim()); continue; }
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
      children: runs(line.trim(), { size: 19, color: INK }) }));
  }
}
closeSection();

const doc = new Document({
  creator: "Sparsh Saraya", title: "Phase 3 Report — The Referee",
  styles: { default: { document: { run: { font: FONT, size: 19, color: INK } } } },
  numbering: { config: [{ reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } },
  ] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 900, bottom: 900, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Noon fraud-detection challenge · Phase 3 · page ", size: 15, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "Phase3_Report.docx"), buf);
  console.log("wrote", path.join(__dirname, "Phase3_Report.docx"));
});
