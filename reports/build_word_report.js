// Builds reports/Phase1_Report.docx from reports/phase1_report.md (archived Phase 1 report; formerly final_report.md) + key figures in outputs/.
// Usage: node reports/build_word_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, HeadingLevel, PageBreak, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const NAVY = "1B2A4A", INK = "2B3445", MUTED = "6B7385", TEAL = "0E9F9A", VIOLET = "7C3AED", GRID = "D5DAE3";
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1000;                 // A4, ~0.69" margins
const CONTENT_W = PAGE_W - 2 * MARGIN;              // DXA

// --- tiny inline parser: **bold**, *italic*, `code` ---------------------------------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: (base.size || 19) - 1, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function figure(file, widthPx, caption) {
  const { w, h, data } = pngSize(path.join(ROOT, file));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 },
      children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] }),
    new Paragraph({ spacing: { after: 120 }, children: runs(caption, { size: 16, color: MUTED, italics: true }) }),
  ];
}

// --- parse the markdown report --------------------------------------------------------------
const md = fs.readFileSync(path.join(__dirname, "phase1_report.md"), "utf8").split("\n");
const children = [];
let tableLines = [];

function flushTable() {
  if (!tableLines.length) return;
  const rows = tableLines.filter(l => !/^\|\s*-/.test(l)).map(l => l.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const nCol = rows[0].length;
  const firstW = 2700;
  const otherW = Math.floor((CONTENT_W - firstW) / (nCol - 1));
  const widths = [firstW, ...Array(nCol - 1).fill(otherW)];
  widths[nCol - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: GRID };
  const table = new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: NAVY }
               : (r[0].includes("CTGAN") ? { type: ShadingType.CLEAR, color: "auto", fill: "F1ECFE" }
               : (r[0].includes("selected") ? { type: ShadingType.CLEAR, color: "auto", fill: "E6F5F4" } : undefined)),
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({ alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(c, { size: 16.5, color: i === 0 ? "FFFFFF" : INK, bold: i === 0 }) })],
      })),
    })),
  });
  children.push(table, new Paragraph({ spacing: { after: 60 }, children: [] }));
  tableLines = [];
}

for (const line of md) {
  if (line.startsWith("|")) { tableLines.push(line); continue; }
  flushTable();
  if (!line.trim() || line.startsWith(">")) continue;          // archive note is for the Markdown copy only
  if (line.startsWith("# ")) {
    children.push(new Paragraph({ heading: HeadingLevel.TITLE, spacing: { after: 40 },
      children: [new TextRun({ text: line.slice(2), bold: true, size: 32, color: NAVY, font: FONT })] }));
  } else if (line.startsWith("**Sparsh")) {
    children.push(new Paragraph({ spacing: { after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 4 } },
      children: runs(line.replace(/\*\*/g, ""), { size: 18, color: MUTED }) }));
  } else if (line.startsWith("*Pending")) {
    children.push(new Paragraph({ spacing: { before: 60 }, children: runs(line.replace(/^\*|\*$/g, ""), { size: 17, color: MUTED, italics: true }) }));
  } else {
    children.push(new Paragraph({ spacing: { after: 90, line: 252 }, alignment: AlignmentType.JUSTIFIED,
      children: runs(line, { size: 19, color: INK }) }));
  }
}
flushTable();

// --- page 2: key figures ---------------------------------------------------------------------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 80 },
  children: [new TextRun({ text: "Key evidence (all figures from the executed notebook)", bold: true, size: 26, color: NAVY, font: FONT })] }));
const W = 640;
children.push(...figure("outputs/metrics/03_before_after_test_metrics.png", W,
  "Figure 1 · Test-set recall, precision, F1 and average precision for all six training arms (thresholds frozen on validation). CTGAN (violet) trades recall for precision; no arm clearly beats the Baseline (teal)."));
children.push(...figure("outputs/gan_diagnostics/01_ctgan_loss_curves.png", W,
  "Figure 2 · CTGAN training on the 1,058 training-partition fraud rows: generator loss falls and flattens, discriminator loss stays near zero — a balanced adversarial game."));
children.push(...figure("outputs/gan_diagnostics/03_fraud_typology_signatures.png", W,
  "Figure 3 · Joint fraud patterns: CTGAN reproduces account-takeover and stolen-card signatures that a column-by-column sampler (green) destroys, but under-produces promo abuse and COD fakes."));

const doc = new Document({
  creator: "Sparsh Saraya", title: "Phase 1 Report — GAN-Augmented Fraud Detection",
  styles: { default: { document: { run: { font: FONT, size: 19, color: INK } } } },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 900, bottom: 900, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "Noon fraud-detection challenge · Phase 1 · page ", size: 15, color: MUTED }),
      new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "Phase1_Report.docx"), buf);
  console.log("wrote", path.join(__dirname, "Phase1_Report.docx"));
});
