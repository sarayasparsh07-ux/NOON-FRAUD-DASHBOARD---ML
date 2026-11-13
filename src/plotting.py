"""Chart design system and all Phase 1 figures.

Rules applied everywhere: descriptive title, named axes with units, legend whenever >1 series,
direct value labels on bars (palette contrast relief), one y-axis per panel, consistent colours:
legitimate/baseline = teal, fraud = coral, GAN = violet, caution = amber.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .config import (AMBER, ARM_COLORS, ARMS, ARM_ORDER, FEATURES, FRAUD, GAN, GRID, INK, LEGIT, MUTED,
                     NAVY, PAYMENT_METHODS, BINARY_COLS)

SEQ_CORAL = LinearSegmentedColormap.from_list("seq_coral", ["#FFF4F2", "#F9B4AC", "#E5534B", "#9E2A24"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1F5FBF", "#9DBBE8", "#F2F2F2", "#F2A39E", "#B83A33"])
SHORT = {"A_baseline": "A Baseline", "B_ctgan": "B CTGAN", "C_class_weighted": "C Class-weighted",
         "D_smote_nc": "D SMOTE-NC", "E_random_oversampling": "E Random oversample",
         "F_independent_marginals": "F Indep. marginals"}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK, "axes.edgecolor": "#B9BFCB", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "figure.titlesize": 14, "figure.titleweight": "bold", "figure.facecolor": "white",
        "axes.facecolor": "white", "lines.linewidth": 2.0,
    })


def save(fig, path, show=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if show:   # display the saved PNG so the notebook shows exactly the file that is delivered
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import Image, display
                display(Image(filename=str(path)))
        except ImportError:
            pass
    return path


def _suptitle(fig, title, subtitle=None):
    """Figure title top-left; optional explanatory note as a caption BELOW the panels (never collides)."""
    fig.suptitle(title, x=0.01, ha="left", color=NAVY)
    if subtitle:
        fig.text(0.01, -0.01, subtitle, ha="left", va="top", color=MUTED, fontsize=9.5)


def tint(hex_color, amount=0.55):
    """Lighter version of a colour (mixed with white) — used for 'raw/uncalibrated' variants."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


def _bar_labels(ax, bars, fmt="{:.2f}", inside=False, fontsize=8, color=INK):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=color)


def _log_axis(ax, ticks):
    """Readable log axis: a few plain-number major ticks, no cluttered minor labels."""
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    t = [v for v in ticks if lo <= v <= hi]
    ax.set_xticks(t, [f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in t])
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())


def _log_edges(vmin, vmax, n, integer=False):
    edges = np.logspace(np.log10(max(vmin, 1)), np.log10(vmax), n)
    if integer:   # integer-aligned bins so 0, 1, 2 days never fall into empty slivers
        centres = np.unique(np.round(edges))
        edges = np.r_[centres - 0.5, centres[-1] + 0.5]
        edges[0] = max(edges[0], 0.5)
    return edges


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def class_distribution(df, path, title="Class distribution of the seed dataset"):
    counts = df["is_fraud"].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    bars = ax.bar(["Legitimate (is_fraud = 0)", "Fraud (is_fraud = 1)"], counts.values, color=[LEGIT, FRAUD], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,} transactions\n({v / counts.sum():.2%})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Number of transactions (count)")
    ax.set_xlabel("Class label")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax.set_title(title)
    return save(fig, path)


def numeric_by_class(df, col, path, xlabel, log=False, bins=40, title=None, clip=None):
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    x = df[col].astype(float)
    if clip is not None:
        x = x.clip(upper=clip)
    is_int = pd.api.types.is_integer_dtype(df[col])
    if log:
        edges = _log_edges(max(x.min(), 1), x.max() + 1, bins, integer=is_int)
    else:
        edges = np.linspace(x.min(), x.max(), bins)
    for cls, color, name in ((0, LEGIT, "Legitimate"), (1, FRAUD, "Fraud")):
        v = (x[df["is_fraud"] == cls] + (1 if log and x.min() <= 0 else 0))
        w = np.full(len(v), 100 / len(v))
        ax.hist(v, bins=edges, weights=w, histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(v, bins=edges, weights=w, histtype="step", linewidth=1.8, color=color,
                label=f"{name} (n = {len(v):,}; median {np.median(x[df['is_fraud'] == cls]):,.1f})")
    if log:
        _log_axis(ax, [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of the class's transactions in bin (%)")
    ax.legend(loc="upper right")
    ax.set_title(title or f"{col} by class")
    return save(fig, path)


def discrete_by_class(df, col, path, xlabel, max_val=15, title=None):
    v = df[col].clip(upper=max_val)
    tab = pd.crosstab(v, df["is_fraud"], normalize="columns") * 100
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    xs = np.arange(len(tab))
    w = 0.4
    ax.bar(xs - w / 2, tab[0], w, color=LEGIT, label=f"Legitimate (n = {(df.is_fraud == 0).sum():,})")
    ax.bar(xs + w / 2, tab[1], w, color=FRAUD, label=f"Fraud (n = {(df.is_fraud == 1).sum():,})")
    ax.set_xticks(xs, [str(i) if i < max_val else f"{max_val}+" for i in tab.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of the class's transactions (%)")
    ax.legend()
    ax.set_title(title or f"{col} by class")
    return save(fig, path)


def fraud_rate_by_category(df, col, path, xlabel, order=None, title=None):
    g = df.groupby(col)["is_fraud"].agg(["mean", "size", "sum"])
    if order is not None:
        g = g.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    bars = ax.bar([str(i) for i in g.index], g["mean"] * 100, color=FRAUD, width=0.55)
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.annotate(f"{r['mean']:.2%}\n({int(r['sum'])} of {int(r['size']):,})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    overall = df["is_fraud"].mean() * 100
    ax.axhline(overall, color=AMBER, ls="--", lw=1.5, label=f"Overall fraud rate ({overall:.2f}%)")
    ax.set_ylim(0, g["mean"].max() * 100 * 1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraud rate within group (%)")
    ax.legend(loc="upper left")
    ax.set_title(title or f"Fraud rate by {col}")
    return save(fig, path)


def rate_heatmap(ax, df, row, col, row_order=None, col_order=None, vmax=None, min_n=20, title=None):
    g = df.groupby([row, col], observed=False)["is_fraud"].agg(["mean", "size"])
    rate = g["mean"].unstack()
    n = g["size"].unstack()
    if row_order is not None:
        rate, n = rate.reindex(row_order), n.reindex(row_order)
    if col_order is not None:
        rate, n = rate[col_order], n[col_order]
    shown = rate.where(n >= min_n)
    im = ax.imshow(shown.values * 100, cmap=SEQ_CORAL, vmin=0, vmax=vmax, aspect="auto")
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            r, k = rate.values[i, j], n.values[i, j]
            if np.isnan(k) or k < min_n:
                txt, c = f"n={0 if np.isnan(k) else int(k)}\n(too few)", MUTED
            else:
                txt = f"{r:.1%}\nn={int(k):,}"
                c = "white" if r * 100 > (vmax or 100) * 0.55 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=c)
    ax.set_xticks(range(shown.shape[1]), [str(c) for c in shown.columns], rotation=0)
    ax.set_yticks(range(shown.shape[0]), [str(r) for r in shown.index])
    ax.grid(False)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel(row.replace("_", " "))
    if title:
        ax.set_title(title, fontsize=10.5)
    return im


def association_heatmap(df, path, title="Spearman rank correlation between features and the fraud label"):
    d = df.copy()
    for m in PAYMENT_METHODS:
        d[f"pay_{m}"] = (d["payment_method"] == m).astype(int)
    cols = [c for c in d.columns if c != "payment_method"]
    cols = [c for c in cols if c != "is_fraud"] + ["is_fraud"]
    corr = d[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    im = ax.imshow(corr.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1))
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman correlation coefficient (−1 to 1)")
    ax.set_title(title)
    return save(fig, path), corr


def single_feature_auc(aucs: pd.Series, path):
    s = aucs.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    bars = ax.barh(s.index, s.values, color=[FRAUD if v >= 0.85 else LEGIT for v in s.values], height=0.6)
    for b, v in zip(bars, s.values):
        ax.annotate(f"{v:.3f}", (v, b.get_y() + b.get_height() / 2), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":", label="0.5 = no information")
    ax.axvline(0.85, color=AMBER, lw=1.5, ls="--", label="0.85 = shortcut warning level (assumed)")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("ROC-AUC of the single feature alone (direction-free, max(AUC, 1−AUC))")
    ax.set_ylabel("Feature")
    ax.legend(loc="lower right")
    ax.set_title("Shortcut check: how well does any one feature separate fraud on its own?")
    return save(fig, path)


# ---------------------------------------------------------------------------
# GAN diagnostics
# ---------------------------------------------------------------------------
def gan_losses(loss: pd.DataFrame, path, window=25, title="CTGAN adversarial training (fitted on training-partition fraud only)"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, col, color in ((axes[0], "Generator Loss", GAN), (axes[1], "Discriminator Loss", NAVY)):
        ax.plot(loss["Epoch"], loss[col], color=color, alpha=0.25, lw=1, label="Last mini-batch of each epoch")
        ax.plot(loss["Epoch"], loss[col].rolling(window, min_periods=1).mean(), color=color, lw=2.2,
                label=f"{window}-epoch rolling mean")
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel(f"{col} (WGAN-GP units)")
        ax.set_title(col)
        ax.legend(loc="best")
    _suptitle(fig, title)
    return save(fig, path)


def marginal_comparison(frames: dict, colors: dict, path, title):
    """Real training fraud vs generated fraud, one panel per feature (shares/densities in %)."""
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.6), constrained_layout=True)
    specs = [("order_amount", "Order amount (AED, log scale)", "log"),
             ("account_age_days", "Account age (days + 1, log scale)", "log1"),
             ("delivery_distance_km", "Delivery distance (km)", "lin"),
             ("num_orders_last_7d", "Orders in previous 7 days (count)", "disc"),
             ("prev_refunds", "Previous refund claims (count)", "disc"),
             ("hour_of_day", "Hour of day (0–23)", "hour"),
             ("payment_method", "Payment method", "cat"),
             ("discount_applied", "Discount applied (0/1)", "cat"),
             ("device_change", "Device change (0/1)", "cat"),
             ("location_mismatch", "Location mismatch (0/1)", "cat")]
    names = list(frames)
    for ax, (col, xl, kind) in zip(axes.ravel(), specs):
        if kind in ("log", "log1", "lin"):
            allv = pd.concat([f[col] for f in frames.values()]).astype(float) + (1 if kind == "log1" else 0)
            edges = (_log_edges(max(allv.min(), 1), allv.max(), 30, integer=(kind == "log1")) if kind != "lin"
                     else np.linspace(0, allv.max(), 30))
            for nm in names:
                v = frames[nm][col].astype(float) + (1 if kind == "log1" else 0)
                ax.hist(v, bins=edges, weights=np.full(len(v), 100 / len(v)), histtype="step", lw=1.8,
                        color=colors[nm], label=nm)
            if kind != "lin":
                _log_axis(ax, [1, 10, 100, 1000] if kind == "log1" else [20, 50, 100, 200, 500, 1000])
            ax.set_ylabel("Share of rows in bin (%)")
        else:
            if kind == "disc":
                cats = list(range(0, 11))
                vals = {nm: frames[nm][col].clip(upper=10).value_counts(normalize=True).reindex(cats, fill_value=0) * 100 for nm in names}
                labels = [str(c) if c < 10 else "10+" for c in cats]
            elif kind == "hour":
                bands = pd.IntervalIndex.from_breaks([-1, 5, 10, 14, 17, 22, 23])
                lab = ["00–05", "06–10", "11–14", "15–17", "18–22", "23"]
                vals = {nm: pd.cut(frames[nm][col], bands).value_counts(normalize=True).reindex(bands, fill_value=0).set_axis(lab) * 100 for nm in names}
                labels = lab
            else:
                cats = PAYMENT_METHODS if col == "payment_method" else [0, 1]
                vals = {nm: frames[nm][col].value_counts(normalize=True).reindex(cats, fill_value=0) * 100 for nm in names}
                labels = [str(c) for c in cats]
            xs = np.arange(len(labels))
            w = 0.8 / len(names)
            for k, nm in enumerate(names):
                ax.bar(xs + (k - (len(names) - 1) / 2) * w, vals[nm].values, w, color=colors[nm], label=nm)
            ax.set_xticks(xs, labels, rotation=0 if len(labels) <= 6 else 0, fontsize=8)
            ax.set_ylabel("Share of rows (%)")
        ax.set_xlabel(xl)
        ax.set_title(col, fontsize=10.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, -0.04), fontsize=10)
    _suptitle(fig, title)
    return save(fig, path)


def correlation_triptych(real, syn_frames: dict, path, title):
    num = [c for c in FEATURES if c != "payment_method"]
    def corr(d):
        x = d[num].copy()
        for m in PAYMENT_METHODS:
            x[f"pay_{m}"] = (d["payment_method"] == m).astype(int)
        return x.corr(method="spearman").fillna(0)
    cr = corr(real)
    panels = [("Training fraud (reference)", cr)]
    for nm, d in syn_frames.items():
        panels.append((f"{nm}: difference from reference", corr(d) - cr))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.2), constrained_layout=True)
    labels = list(cr.columns)
    for ax, (nm, m) in zip(axes, panels):
        im = ax.imshow(m.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1))
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels), fontsize=7.5)
        ax.grid(False)
        sub = "" if nm.startswith("Training") else f"\nmean |Δρ| = {np.abs(m.values[np.triu_indices(len(labels), 1)]).mean():.3f}"
        ax.set_title(nm + sub, fontsize=10)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label("Spearman ρ (reference) / Δρ (generators)")
    _suptitle(fig, title)
    return save(fig, path)


def grouped_bars(df: pd.DataFrame, colors: dict, path, title, ylabel, xlabel, fmt="{:.1%}", ylim=None,
                 refline=None, figsize=(12, 5), legend_loc="upper right", rotate=0, pct_axis=True):
    """df: index = x groups, columns = series."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    xs = np.arange(len(df.index))
    w = 0.82 / len(df.columns)
    for k, c in enumerate(df.columns):
        bars = ax.bar(xs + (k - (len(df.columns) - 1) / 2) * w, df[c].values, w * 0.95, color=colors[c], label=c)
        _bar_labels(ax, bars, fmt=fmt, fontsize=7.5)
    ax.set_xticks(xs, df.index, rotation=rotate, ha="right" if rotate else "center")
    if refline is not None:
        ax.axhline(refline[0], color=AMBER, ls="--", lw=1.4, label=refline[1])
    if ylim:
        ax.set_ylim(*ylim)
    if pct_axis:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(l), 6), fontsize=9)
    ax.set_title(title)
    return save(fig, path)


def c2st_chart(res: pd.DataFrame, colors: dict, path):
    """res: index = generator name, columns = AUC — Logistic regression / AUC — Gradient boosting."""
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    xs = np.arange(len(res))
    for k, (col, hatch) in enumerate(((res.columns[0], None), (res.columns[1], "//"))):
        bars = ax.bar(xs + (k - 0.5) * 0.38, res[col].values, 0.36, color=[colors[i] for i in res.index],
                      hatch=hatch, edgecolor="white", label=col.replace("AUC — ", "Detector: "))
        _bar_labels(ax, bars, fmt="{:.3f}", fontsize=8)
    ax.axhline(0.5, color=AMBER, ls="--", lw=1.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor="#9AA3B2", edgecolor="white", label="Detector: logistic regression (solid)"),
               Patch(facecolor="#9AA3B2", edgecolor="white", hatch="//", label="Detector: gradient boosting (hatched)"),
               Line2D([0], [0], color=AMBER, ls="--", lw=1.5, label="0.5 = indistinguishable from training fraud")]
    ax.set_xticks(xs, res.index)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Cross-validated ROC-AUC of 'real vs generated' detector")
    ax.set_xlabel("Rows compared against the training-fraud reference (bar colour = source)")
    ax.legend(handles=handles, loc="upper right", ncol=1)
    ax.set_title("Classifier two-sample test: can a model tell generated fraud from the real training fraud?")
    return save(fig, path)


def dcr_chart(dcr: dict, colors: dict, path):
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    allv = np.concatenate(list(dcr.values()))
    edges = np.linspace(0, np.percentile(allv, 99), 40)
    for nm, v in dcr.items():
        ax.hist(v, bins=edges, weights=np.full(len(v), 100 / len(v)), histtype="step", lw=1.9, color=colors[nm],
                label=f"{nm} (median {np.median(v):.2f}; exact copies {np.mean(v < 1e-9):.1%})")
    ax.set_xlabel("Distance to the closest training-fraud row (standardised feature space)")
    ax.set_ylabel("Share of rows in bin (%)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=9)
    ax.set_title("Memorisation check: distance from each row to its nearest training-fraud record")
    return save(fig, path)


# ---------------------------------------------------------------------------
# Classifier results
# ---------------------------------------------------------------------------
def confusion_grid(results: dict, path, title):
    arms = [a for a in ARM_ORDER if a in results]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, a in zip(axes.ravel(), arms):
        m = results[a]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        rowpct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap=LinearSegmentedColormap.from_list("x", ["#FFFFFF", ARM_COLORS[a]]), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({rowpct[i, j]:.1%} of row)", ha="center", va="center", fontsize=10,
                        color="white" if rowpct[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Predicted LEGIT", "Predicted FRAUD"])
        ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
        ax.grid(False)
        ax.set_title(f"{ARMS[a]}\nthreshold {m['threshold']:.3f} · recall {m['recall']:.1%} · precision {m['precision']:.1%}", fontsize=10)
    _suptitle(fig, title)
    return save(fig, path)


def pr_curves(curves: dict, points: dict, prevalence: float, path, title):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
    for a in [a for a in ARM_ORDER if a in curves]:
        y, p = curves[a]
        pr, rc, _ = precision_recall_curve(y, p)
        ax.step(rc, pr, where="post", color=ARM_COLORS[a], lw=1.8,
                label=f"{ARMS[a]} (AP {average_precision_score(y, p):.3f})")
        if a in points:
            ax.plot(points[a][0], points[a][1], "o", ms=8, color=ARM_COLORS[a], mec="white", mew=1.5)
    ax.axhline(prevalence, color=AMBER, ls="--", lw=1.4, label=f"No-skill baseline = fraud prevalence ({prevalence:.2%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of actual fraud caught)")
    ax.set_ylabel("Precision (share of flagged orders that are fraud)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(title)
    fig.text(0.01, -0.02, "Dots mark each arm's frozen operating point (threshold chosen on validation).", color=MUTED, fontsize=9)
    return save(fig, path)


def threshold_sensitivity(sweep: pd.DataFrame, chosen: float, path, title, cost_points=None):
    sw = sweep.copy()
    sw.loc[sw["flagged"] == 0, "precision"] = np.nan        # precision is undefined when nothing is flagged
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(sw.threshold, sw.precision, color=NAVY, label="Precision")
    ax.plot(sw.threshold, sw.recall, color=FRAUD, label="Recall")
    ax.plot(sw.threshold, sw.f1, color=GAN, label="F1")
    ax.plot(sw.threshold, sw.fpr, color=LEGIT, ls="-.", label="False-positive rate")
    ax.axvline(chosen, color=AMBER, ls="--", lw=1.8, label=f"Frozen threshold {chosen:.3f} (max F1)")
    ax.set_xlabel("Decision threshold on calibrated fraud probability")
    ax.set_ylabel("Metric value (0–1)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.set_title("Precision, recall, F1 and FPR vs threshold")
    ax = axes[1]
    ax.plot(sw.threshold, sw.FP, color=LEGIT, label="False positives (legit orders flagged)")
    ax.plot(sw.threshold, sw.FN, color=FRAUD, label="False negatives (fraud missed)")
    ax.axvline(chosen, color=AMBER, ls="--", lw=1.8, label=f"Frozen threshold {chosen:.3f} (max F1)")
    styles = [":", (0, (5, 2)), (0, (1, 1, 4, 1))]
    for (lab, t), ls in zip((cost_points or {}).items(), styles):
        ax.axvline(t, color=MUTED, ls=ls, lw=1.4, label=f"{lab.replace(chr(10), ' ')}: {t:.2f}")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, max(sw.FP.max(), sw.FN.max()) * 1.6)
    ax.set_xlabel("Decision threshold on calibrated fraud probability")
    ax.set_ylabel("Number of transactions (count, symlog scale)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
    ax.set_title("Error counts vs threshold (grey lines = HYPOTHETICAL cost ratios)")
    _suptitle(fig, title)
    return save(fig, path)


def reliability(tables: dict, colors: dict, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    mx = max(max(t["mean_predicted"].max(), t["observed_rate"].max()) for t in tables.values())
    lim = min(1.0, mx * 1.08)
    ax.plot([0, lim], [0, lim], color=MUTED, ls=":", lw=1.2, label="Perfect calibration")
    for nm, t in tables.items():
        ax.plot(t["mean_predicted"], t["observed_rate"], "o-", color=colors[nm], ms=6, label=nm)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted fraud probability in bin")
    ax.set_ylabel("Observed fraud rate in bin")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(title)
    fig.text(0.01, -0.02, "Bins are deciles of predicted probability (equal counts), test partition.", color=MUTED, fontsize=9)
    return save(fig, path)


def forest_plot(diffs: pd.DataFrame, path, title):
    """diffs columns: arm, metric, diff_B_minus_A, ci_low, ci_high."""
    metrics = [("pr_auc_ap", "Δ Average precision"), ("recall", "Δ Recall"), ("precision", "Δ Precision"), ("f1", "Δ F1")]
    arms = [a for a in ARM_ORDER if a in diffs["arm"].unique()]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True, sharey=True)
    for ax, (m, lab) in zip(axes, metrics):
        d = diffs[diffs.metric == m].set_index("arm").reindex(arms)
        ys = np.arange(len(arms))[::-1]
        for y, a in zip(ys, arms):
            r = d.loc[a]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=ARM_COLORS[a], lw=2.5)
            ax.plot(r.diff_B_minus_A, y, "o", color=ARM_COLORS[a], ms=8, mec="white", mew=1.5)
            ax.annotate(f"{r.diff_B_minus_A:+.3f}", (r.diff_B_minus_A, y), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
        ax.axvline(0, color=AMBER, ls="--", lw=1.3)
        ax.set_yticks(ys, [SHORT[a] for a in arms])
        ax.set_xlabel(f"{lab} vs Baseline (test)")
        ax.set_title(lab, fontsize=10.5)
    axes[0].set_ylabel("Training arm")
    _suptitle(fig, title)
    fig.text(0.01, -0.03, "Points = test-set difference from Baseline; bars = paired, class-stratified bootstrap 95% interval (2,000 resamples).",
             color=MUTED, fontsize=9)
    return save(fig, path)


def seed_dots(df: pd.DataFrame, path, title):
    """df columns: seed, arm, pr_auc_ap, recall, precision, f1, FP."""
    metrics = [("pr_auc_ap", "Average precision"), ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]
    arms = [a for a in ARM_ORDER if a in df["arm"].unique()]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), constrained_layout=True, sharey=True)
    for ax, (m, lab) in zip(axes, metrics):
        for k, a in enumerate(arms[::-1]):
            v = df[df.arm == a][m].values
            ax.plot(v, np.full(len(v), k) + np.linspace(-0.15, 0.15, len(v)), "o", color=ARM_COLORS[a], ms=6, alpha=0.8)
            ax.plot([v.mean()], [k], "|", color=NAVY, ms=18, mew=2.5)
            ax.annotate(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}", (v.mean(), k), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(arms)), [SHORT[a] for a in arms[::-1]])
        ax.set_xlabel(f"{lab} on that seed's test partition (0–1)")
        ax.set_title(lab, fontsize=10.5)
    axes[0].set_ylabel("Training arm")
    _suptitle(fig, title)
    fig.text(0.01, -0.03, "Each dot = one full re-run (new seed data, split, CTGAN, classifiers). Navy tick = mean; label = mean ± SD.",
             color=MUTED, fontsize=9)
    return save(fig, path)


def heat_table(df: pd.DataFrame, path, title, xlabel, ylabel, fmt="{:.0%}", cmap=SEQ_CORAL, vmin=0, vmax=1, cbar_label="", figsize=(11, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(df.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.6 else INK)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=0, ha="center", fontsize=8.5)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)
    ax.set_title(title)
    return save(fig, path)


def single_confusion(m: dict, path, title, color=LEGIT):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    rowpct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5), constrained_layout=True)
    ax.imshow(rowpct, cmap=LinearSegmentedColormap.from_list("x", ["#FFFFFF", color]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({rowpct[i, j]:.1%} of row)", ha="center", va="center", fontsize=11,
                    color="white" if rowpct[i, j] > 0.6 else INK)
    ax.set_xticks([0, 1], ["Predicted LEGIT", "Predicted FRAUD"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_xlabel("Model decision")
    ax.set_ylabel("True label")
    ax.grid(False)
    ax.set_title(title, fontsize=11)
    return save(fig, path)


def ratio_selection(grid: pd.DataFrame, refs: dict, chosen: dict, path, title):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for arm, g in grid.groupby("arm"):
        g = g.sort_values("ratio")
        ax.plot(g["ratio"], g["val_ap"], "o-", color=ARM_COLORS[arm], label=ARMS[arm], ms=7)
        c = g[g["ratio"] == chosen[arm]]
        ax.plot(c["ratio"], c["val_ap"], "o", ms=14, mfc="none", mec=ARM_COLORS[arm], mew=2)
    for arm, v in refs.items():
        ax.axhline(v, color=ARM_COLORS[arm], ls="--", lw=1.4, label=f"{ARMS[arm]} (no synthetic rows)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(grid["ratio"].unique()), [f"{r:g}×" for r in sorted(grid["ratio"].unique())])
    ax.set_xlabel("Synthetic fraud rows added, as a multiple of the training fraud count (log2 scale)")
    ax.set_ylabel("Validation average precision (PR-AUC)")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.set_title(title)
    fig.text(0.01, -0.03, "Open circles mark the ratio selected for each arm (best validation AP; ties within 0.002 → smaller ratio).",
             color=MUTED, fontsize=9)
    return save(fig, path)


def line_by_group(df: pd.DataFrame, x: str, y: str, group: str, colors: dict, labels: dict, path, title, xlabel, ylabel,
                  xticks=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for g, d in df.groupby(group):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], "o-", color=colors[g], label=labels.get(g, g), ms=7)
        for xi, yi in zip(d[x], d[y]):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    if xticks is not None:
        ax.set_xticks(xticks[0], xticks[1])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8.5)
    ax.set_title(title)
    return save(fig, path)


def epoch_sensitivity(tab: pd.DataFrame, main_epochs: int, path):
    """Two panels: SDMetrics scores (1 = best) and C2ST AUC (0.5 = best) — different directions, separate axes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    ep = tab["epochs"].values
    ax = axes[0]
    for col, color, mk in (("Column Shapes", LEGIT, "o"), ("Column Pair Trends", NAVY, "s"), ("LogisticDetection", GAN, "^")):
        ax.plot(ep, tab[col], marker=mk, ms=7, color=color, label=col)
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("SDMetrics score vs training fraud (1 = identical)")
    ax.set_title("Fidelity scores (higher is better)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(ep, tab["C2ST AUC (gradient boosting; 0.5 ideal)"], "o-", ms=7, color=FRAUD, label="Two-sample test AUC (gradient boosting)")
    for x_, y_ in zip(ep, tab["C2ST AUC (gradient boosting; 0.5 ideal)"]):
        ax.annotate(f"{y_:.3f}", (x_, y_), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=AMBER, ls="--", lw=1.4, label="0.5 = indistinguishable")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Detector ROC-AUC (0.5 = best)")
    ax.set_title("Detectability of generated rows (lower is better)")
    ax.legend(loc="upper right")
    for ax in axes:
        ax.axvline(main_epochs, color=MUTED, ls=":", lw=1.2)
        ax.set_xticks(ep, [str(e) + (" (used)" if e == main_epochs else "") for e in ep])
        ax.set_xlabel("CTGAN training epochs")
    _suptitle(fig, "Synthetic-fraud fidelity by CTGAN training length (same data and seed)")
    return save(fig, path)


# ---------------------------------------------------------------------------
# Phase 2 — my GAN vs Noon's GAN
# ---------------------------------------------------------------------------
AGE_BAND_EDGES = [-0.5, 0.5, 7.5, 30.5, 90.5, 180.5, 365.5, 730.5, np.inf]
AGE_BAND_LABELS = ["0", "1–7", "8–30", "31–90", "91–180", "181–365", "366–730", "> 730"]


def _shares(s, levels):
    return pd.Series(s).value_counts(normalize=True).reindex(levels, fill_value=0) * 100


def _grouped(ax, cats, series: dict, colors: dict, ylabel, xlabel, fmt="{:.0f}", label_size=6.5, rot=0):
    xs = np.arange(len(cats))
    w = 0.82 / len(series)
    for k, (nm, vals) in enumerate(series.items()):
        bars = ax.bar(xs + (k - (len(series) - 1) / 2) * w, vals, w * 0.94, color=colors[nm], label=nm)
        if label_size:
            for b, v in zip(bars, vals):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 1.5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=label_size, color=INK)
    ax.set_xticks(xs, cats, rotation=rot)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def p2_brief_features(mine, noon_u, noon_all, colors, path, title):
    """The three features the brief names, with SHARED bins: order_amount (continuous), account_age_days and
    num_orders_last_7d (discrete ordered integers). Top row: shares per shared bin; bottom row: ECDFs."""
    names = list(colors)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.6), constrained_layout=True)
    # order amount: shared log-spaced bins
    allv = np.concatenate([mine.order_amount, noon_u.order_amount])
    edges = np.logspace(np.log10(allv.min() * 0.98), np.log10(allv.max() * 1.02), 26)
    for nm, d in ((names[0], mine), (names[1], noon_u)):
        ax = axes[0, 0]
        ax.hist(d.order_amount, bins=edges, weights=np.full(len(d), 100 / len(d)), histtype="stepfilled", alpha=0.22, color=colors[nm])
        ax.hist(d.order_amount, bins=edges, weights=np.full(len(d), 100 / len(d)), histtype="step", lw=1.9, color=colors[nm], label=nm)
    _log_axis(axes[0, 0], [10, 20, 50, 100, 200, 500, 1000])
    axes[0, 0].set_xlabel("Order amount (AED, log scale; same 25 bins for both)")
    axes[0, 0].set_ylabel("Share of the dataset's rows in bin (%)")
    axes[0, 0].set_title("order_amount", fontsize=10.5)
    # account age: discrete ordered → shares by integer bands
    sers = {nm: pd.cut(d.account_age_days, AGE_BAND_EDGES, labels=AGE_BAND_LABELS).value_counts(normalize=True).reindex(AGE_BAND_LABELS) * 100
            for nm, d in ((names[0], mine), (names[1], noon_u))}
    _grouped(axes[0, 1], AGE_BAND_LABELS, sers, colors, "Share of the dataset's rows (%)", "Account age band (whole days)", rot=0)
    axes[0, 1].set_title("account_age_days (integer days)", fontsize=10.5)
    # orders: one bar per integer, 10 = 10 or more
    lv = list(range(11))
    sers = {nm: _shares(d.num_orders_last_7d.clip(upper=10), lv) for nm, d in ((names[0], mine), (names[1], noon_u))}
    _grouped(axes[0, 2], [str(i) if i < 10 else "10+" for i in lv], sers, colors, "Share of the dataset's rows (%)",
             "Orders in previous 7 days (count)")
    axes[0, 2].set_title("num_orders_last_7d (integer count)", fontsize=10.5)
    # ECDFs
    for ax, col, xl, kind in ((axes[1, 0], "order_amount", "Order amount (AED, log scale)", "log"),
                              (axes[1, 1], "account_age_days", "Account age (days + 1, log scale)", "log1"),
                              (axes[1, 2], "num_orders_last_7d", "Orders in previous 7 days (count)", "lin")):
        for nm, d, ls in ((names[0], mine, "-"), (names[1], noon_u, "-"), ("Noon fraud — as supplied (with duplicates)", noon_all, "--")):
            v = np.sort(d[col].to_numpy(float) + (1 if kind == "log1" else 0))
            c = colors[names[1]] if "as supplied" in nm else colors[nm]
            ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=c, lw=1.6 if ls == "--" else 2.1, ls=ls, label=nm)
        if kind == "log":
            _log_axis(ax, [10, 20, 50, 100, 200, 500, 1000])
        elif kind == "log1":
            _log_axis(ax, [1, 10, 100, 1000])
        ax.set_ylim(0, 101)
        ax.set_xlabel(xl)
        ax.set_ylabel("Rows at or below value (%)")
        ax.set_title(f"{col} — cumulative distribution", fontsize=10.5)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=10)
    _suptitle(fig, title)
    return save(fig, path)


def p2_other_features(mine, noon_u, colors, path, title):
    names = list(colors)
    D = ((names[0], mine), (names[1], noon_u))
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    _grouped(ax, [str(h) for h in range(24)], {nm: _shares(d.hour_of_day, list(range(24))) for nm, d in D}, colors,
             "Share of the dataset's rows (%)", "Hour of day the order was placed (0–23)", label_size=0)
    ax.set_title("hour_of_day (integer, cyclic)", fontsize=10.5)
    ax = fig.add_subplot(gs[0, 2])
    edges = np.arange(0, max(mine.delivery_distance_km.max(), noon_u.delivery_distance_km.max()) + 1.01, 1.0)
    for nm, d in D:
        ax.hist(d.delivery_distance_km, bins=edges, weights=np.full(len(d), 100 / len(d)), histtype="step", lw=1.9, color=colors[nm], label=nm)
    ax.set_xlabel("Delivery distance (km; same 1-km bins for both)")
    ax.set_ylabel("Share of the dataset's rows in bin (%)")
    ax.set_title("delivery_distance_km", fontsize=10.5)
    ax = fig.add_subplot(gs[1, 0])
    lv = list(range(6))
    _grouped(ax, [str(i) if i < 5 else "5+" for i in lv], {nm: _shares(d.prev_refunds.clip(upper=5), lv) for nm, d in D}, colors,
             "Share of the dataset's rows (%)", "Previous refund claims (count)")
    ax.set_title("prev_refunds (integer count)", fontsize=10.5)
    for gi, (feat_list, ttl, xl) in enumerate(((["payment_method"], "payment_method", "Payment method"),
                                               (BINARY_COLS, "Binary flags (share = 1)", "Flag"))):
        ax = fig.add_subplot(gs[1, 1 + gi])
        cats, vals, lo, hi = [], {nm: [] for nm, _ in D}, {nm: [] for nm, _ in D}, {nm: [] for nm, _ in D}
        levels = [(f, lvl) for f in feat_list for lvl in (PAYMENT_METHODS if f == "payment_method" else [1])]
        from .compare import wilson
        for f, lvl in levels:
            cats.append(lvl if f == "payment_method" else f.replace("_", "\n"))
            for nm, d in D:
                k, n = int((d[f] == lvl).sum()), len(d)
                vals[nm].append(100 * k / n)
                a, b = wilson(k, n)
                lo[nm].append(100 * (k / n - a)); hi[nm].append(100 * (b - k / n))
        xs = np.arange(len(cats)); w = 0.4
        for k_, (nm, _) in enumerate(D):
            bars = ax.bar(xs + (k_ - 0.5) * w, vals[nm], w * 0.94, color=colors[nm], yerr=[lo[nm], hi[nm]], capsize=3,
                          error_kw={"ecolor": NAVY, "lw": 0.9}, label=nm)
            for bb, v, e in zip(bars, vals[nm], hi[nm]):
                ax.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v + e), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(xs, cats)
        ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.2)
        ax.set_ylabel("Share of the dataset's rows (%, 95% Wilson CI)")
        ax.set_xlabel(xl)
        ax.set_title(ttl, fontsize=10.5)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10)
    _suptitle(fig, title)
    return save(fig, path)


def p2_effect_chart(eff: pd.DataFrame, pair_colors: dict, supplied: pd.DataFrame, path, title):
    feats = list(dict.fromkeys(eff["feature"]))
    pairs = list(dict.fromkeys(eff["pair"]))
    fig, ax = plt.subplots(figsize=(12.5, 1.8 + len(feats) * (0.24 * len(pairs) + 0.25)), constrained_layout=True)
    h = 0.8 / len(pairs)
    ys = np.arange(len(feats))[::-1]
    for k, pr in enumerate(pairs):
        d = eff[eff["pair"] == pr].set_index("feature").reindex(feats)
        pos = ys + (len(pairs) - 1) / 2 * h - k * h
        bars = ax.barh(pos, d["effect size"], h * 0.92, color=pair_colors[pr], label=pr)
        sup = None
        if k == 0 and supplied is not None:
            sup = supplied.set_index("feature").reindex(feats)["effect size"].to_numpy()
            ax.plot(sup, pos, "D", ms=6, mfc="white", mec=NAVY, mew=1.4, zorder=4,
                    label="Same pair (first bar of each group), Noon as supplied with duplicates kept")
        for j, (b, v) in enumerate(zip(bars, d["effect size"])):
            xlab = v if sup is None else max(v, sup[j])
            txt = f"{v:.2f}" if sup is None else f"{v:.2f} (as supplied {sup[j]:.2f})"
            ax.annotate(txt, (xlab, b.get_y() + b.get_height() / 2), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    kinds = eff.drop_duplicates("feature").set_index("feature")["headline effect"]
    ax.set_yticks(ys, [f"{f}  [{kinds[f]}]" for f in feats])
    xmax = max(0.45, eff["effect size"].max() * 1.25)
    ax.set_xlim(0, xmax)
    for cut in (0.05, 0.15, 0.30):
        ax.axvline(cut, color=MUTED, ls=":", lw=1)
    for lo, hi, lab in ((0, 0.05, "negl."), (0.05, 0.15, "small"), (0.15, 0.30, "moderate"), (0.30, xmax, "large")):
        ax.annotate(lab, ((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.set_xlabel("Effect size (KS D for continuous and integer features; total-variation distance for hour and categorical features; 0 = identical)")
    ax.set_ylabel("Feature [effect measure]")
    ax.set_title(title, pad=20)
    hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=9)
    return save(fig, path)


def relationship_panels(rel: pd.DataFrame, colors: dict, path, title):
    rels = list(dict.fromkeys(rel["relationship"]))
    datasets = list(dict.fromkeys(rel["dataset"]))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.4), constrained_layout=True)
    for ax, r in zip(axes.ravel(), rels):
        d = rel[rel["relationship"] == r]
        groups = list(dict.fromkeys(d["group"]))
        is_rate = bool(d["is_rate"].iloc[0])
        sers = {}
        for nm in datasets:
            dd = d[d["dataset"] == nm].set_index("group").reindex(groups)
            sers[nm] = (dd["value"] * (100 if is_rate else 1)).to_numpy()
        xs = np.arange(len(groups)); w = 0.8 / len(datasets)
        for k, nm in enumerate(datasets):
            vals = sers[nm]
            bars = ax.bar(xs + (k - (len(datasets) - 1) / 2) * w, np.nan_to_num(vals), w * 0.92, color=colors[nm], label=nm)
            for b, v in zip(bars, vals):
                txt = "n<20" if np.isnan(v) else (f"{v:.0f}" if (is_rate or v >= 10) else f"{v:.2f}")
                ax.annotate(txt, (b.get_x() + b.get_width() / 2, 0 if np.isnan(v) else v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5, color=MUTED if np.isnan(v) else INK, rotation=90 if np.isnan(v) else 0)
        ax.set_xticks(xs, groups, fontsize=8.5)
        ax.set_ylabel(d["y label"].iloc[0])
        ax.set_xlabel(d["x label"].iloc[0])
        allv = np.concatenate([v for v in sers.values()])
        ax.set_ylim(0, np.nanmax(allv) * 1.18 if np.isfinite(np.nanmax(allv)) else 1)
        ax.set_title(r, fontsize=10.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(l), fontsize=10)
    _suptitle(fig, title)
    fig.text(0.01, -0.055, "Groups with fewer than 20 rows are not plotted (labelled n<20). Rates are shares of the group's rows. Unique rows.",
             ha="left", va="top", color=MUTED, fontsize=9.5)
    return save(fig, path)


def p2_multiplicity(obs: pd.Series, exp: pd.Series, fit: dict, color, path, title):
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.bar(obs.index.astype(str), obs.values, color=color, width=0.62, label="Observed distinct rows")
    for b, v, e in zip(bars, obs.values, exp.values):
        if v >= 30:   # label inside the bar, clear of the expected-value marker above/at the bar top
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v / 2), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        else:         # short bars: label above both the bar and the marker
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, max(v, e)), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.plot(obs.index.astype(str), exp.values, "o--", color=NAVY, lw=1.6, ms=6,
            label=f"Expected if 3,000 rows were drawn with replacement from ≈{fit['implied base rows N = rows/λ']:,.0f} base rows "
                  f"(zero-truncated Poisson, λ = {fit['lambda']:.2f}; χ² p = {fit['p (χ² goodness of fit)']:.2f})")
    ax.set_xlabel("Number of times an identical row appears in Noon's file")
    ax.set_ylabel("Number of distinct rows (count)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=9)
    ax.set_title(title)
    return save(fig, path)


def p2_lie_detector(tab: pd.DataFrame, path, title):
    """tab columns: protocol, ROC-AUC, ROC-AUC sd, accuracy, majority baseline, color."""
    fig, ax = plt.subplots(figsize=(15, 6.2), constrained_layout=True)
    xs = np.arange(len(tab)); w = 0.36
    b1 = ax.bar(xs - w / 2, tab["accuracy"], w * 0.94, color=tab["color"])
    b2 = ax.bar(xs + w / 2, tab["ROC-AUC"], w * 0.94, color=tab["color"], hatch="//", edgecolor="white")
    from matplotlib.patches import Patch
    ax.add_artist(ax.legend(handles=[Patch(facecolor="#8A93A3", label="Accuracy (solid bar)"),
                                     Patch(facecolor="#8A93A3", hatch="//", edgecolor="white", label="ROC-AUC (hatched bar)")],
                            loc="upper right", fontsize=8.5, frameon=True))
    for bars, col in ((b1, "accuracy"), (b2, "ROC-AUC")):
        for b, v in zip(bars, tab[col]):
            if np.isnan(v):
                ax.annotate("not\nreported", (b.get_x() + b.get_width() / 2, 0.41), ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    for x, bl in zip(xs, tab["majority baseline"]):
        ax.plot([x - w, x], [bl, bl], color=AMBER, lw=2.4)
    ax.plot([], [], color=AMBER, lw=2.4, label="Accuracy baseline: always predict the larger source")
    ax.axhline(0.5, color=MUTED, ls=":", lw=1, label="ROC-AUC chance level (0.5)")
    import textwrap
    ax.set_xticks(xs, ["\n".join(textwrap.wrap(str(p).replace("\n", " "), 22)) for p in tab["protocol"]], fontsize=8.5)
    ax.set_ylim(0.4, 1.12)
    ax.set_ylabel("Cross-validated score (0–1)")
    ax.set_xlabel("Lie-detector protocol (Random Forest, 100 trees, helper's preprocessing; is_fraud excluded)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    ax.set_title(title)
    return save(fig, path)


def p2_perm_and_importance(perm: dict, imp: pd.DataFrame, color, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.hist(perm["null"], bins=30, color="#9AA3B2", label=f"Null: source labels permuted ({perm['n_permutations']:,} times)")
    ax.axvline(perm["observed mean CV ROC-AUC"], color=color, lw=2.6, label=f"Observed ROC-AUC {perm['observed mean CV ROC-AUC']:.3f} (p = {perm['permutation p-value']:.4f})")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Mean 5-fold cross-validated ROC-AUC")
    ax.set_ylabel("Number of permutations (count)")
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_title("Permutation test (unique rows, balanced, shuffled stratified CV)", fontsize=10.5)
    ax = axes[1]
    s = imp["mean drop in ROC-AUC"].sort_values()
    bars = ax.barh(s.index, s.values, color=color, height=0.62, xerr=imp.loc[s.index, "sd over folds"], error_kw={"ecolor": NAVY, "lw": 0.9})
    sds = imp.loc[s.index, "sd over folds"].to_numpy()
    for b, v, sd in zip(bars, s.values, sds):
        ax.annotate(f"{v:.3f}", (max(v, 0) + sd, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlim(min(0, (s - sds).min() * 1.1), (s.values + sds).max() * 1.15)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Drop in held-out ROC-AUC when the feature is permuted (mean over folds ± SD)")
    ax.set_ylabel("Feature")
    ax.set_title("Which features give the source away? (held-out permutation importance)", fontsize=10.5)
    _suptitle(fig, title)
    return save(fig, path)


# ---------------------------------------------------------------------------
# Phase 3 — the referee: real data judges every model
# ---------------------------------------------------------------------------
def p3_headline(tab: pd.DataFrame, colors: dict, path, title):
    """Step 3.3 final bar chart. tab: index = model name; columns TP, FN, FP, n fraud, recall, recall CI low/high."""
    names = list(tab.index)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    xs = np.arange(len(names))
    ax = axes[0]
    rec = tab["recall"].to_numpy() * 100
    lo = rec - tab["recall 95% CI low"].to_numpy() * 100
    hi = tab["recall 95% CI high"].to_numpy() * 100 - rec
    bars = ax.bar(xs, rec, 0.58, color=[colors[n] for n in names], yerr=[lo, hi], capsize=5,
                  error_kw={"ecolor": NAVY, "lw": 1.1})
    for b, r, h, (_, row) in zip(bars, rec, hi, tab.iterrows()):
        ax.annotate(f"{r:.1f}%\n{row['TP']} of {row['n fraud']} caught", (b.get_x() + b.get_width() / 2, r + h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Fraud recall: real frauds caught (%, 95% Wilson CI)")
    ax.set_title("Who caught more real fraud? (higher is better)", fontsize=11)
    ax = axes[1]
    fp = tab["FP"].to_numpy()
    bars = ax.bar(xs, fp, 0.58, color=[colors[n] for n in names])
    n_legit = int(tab["FP"].iloc[0] + tab["TN"].iloc[0])
    for b, v in zip(bars, fp):
        ax.annotate(f"{v:,}\n({v / n_legit:.2%} of {n_legit:,} legit)", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(0, fp.max() * 1.25 if fp.max() > 0 else 1)
    ax.set_ylabel("Legitimate orders wrongly flagged (false positives, count)")
    ax.set_title("Who flagged fewer innocent orders? (lower is better)", fontsize=11)
    for ax in axes:
        ax.set_xticks(xs, [n.replace(" (", "\n(") for n in names], fontsize=9.5)
        ax.set_xlabel("Model (each with its own frozen prediction rule)")
    _suptitle(fig, title)
    return save(fig, path)


def p3_confusions(rows: pd.DataFrame, colors: dict, path, title):
    names = list(rows.index)
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.9), constrained_layout=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        r = rows.loc[n]
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        rowpct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap=LinearSegmentedColormap.from_list("x", ["#FFFFFF", colors[n]]), vmin=0, vmax=1)
        lab = [["Legit correctly passed", "Legit wrongly flagged"], ["Fraud missed", "Fraud caught"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{lab[i][j]}\n{cm[i, j]:,}\n({rowpct[i, j]:.1%} of row)", ha="center", va="center", fontsize=9.5,
                        color="white" if rowpct[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Predicted LEGIT", "Predicted FRAUD"])
        ax.set_yticks([0, 1], ["Actual legit\n(n = {:,})".format(int(cm[0].sum())), "Actual fraud\n(n = {:,})".format(int(cm[1].sum()))])
        ax.grid(False)
        ax.set_title(n, fontsize=10.5)
    _suptitle(fig, title)
    return save(fig, path)


def p3_pr_roc(y, scores: dict, points: dict, colors: dict, path, title):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    prev = float(np.mean(y))
    for n, s in scores.items():
        pr, rc, _ = precision_recall_curve(y, s)
        axes[0].step(rc, pr, where="post", color=colors[n], lw=1.9, label=f"{n} (AP {average_precision_score(y, s):.3f})")
        axes[0].plot(points[n][0], points[n][1], "o", ms=9, color=colors[n], mec="white", mew=1.5, zorder=5)
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, color=colors[n], lw=1.9, label=f"{n} (ROC-AUC {roc_auc_score(y, s):.3f})")
        axes[1].plot(points[n][2], points[n][0], "o", ms=9, color=colors[n], mec="white", mew=1.5, zorder=5)
    axes[0].axhline(prev, color=AMBER, ls="--", lw=1.4, label=f"No-skill precision = fraud prevalence ({prev:.3%})")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Recall (share of the 65 real frauds caught)")
    axes[0].set_ylabel("Precision (share of flagged orders that are fraud)")
    axes[0].set_title("Precision–recall curves on the referee set", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[1].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="Chance (ROC-AUC 0.5)")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("False-positive rate (share of legitimate orders flagged)")
    axes[1].set_ylabel("Recall / true-positive rate")
    axes[1].set_title("ROC curves on the referee set", fontsize=11)
    axes[1].legend(loc="lower right", fontsize=8.5)
    _suptitle(fig, title, "Dots mark each model's frozen operating point (its own stored threshold or prediction rule). "
                           "Curves are descriptive only; no threshold was chosen from them.")
    return save(fig, path)


def p3_detection_overlap(counts: pd.DataFrame, colors: list, path, title):
    """counts: one row, columns = categories of the 65 frauds (caught by both / only A / only B / neither)."""
    fig, ax = plt.subplots(figsize=(13, 2.9), constrained_layout=True)
    left = 0
    total = counts.values.sum()
    for (cat, v), c in zip(counts.iloc[0].items(), colors):
        ax.barh([0], [v], left=left, color=c, height=0.55, label=f"{cat}: {v}")
        if v:
            ax.text(left + v / 2, 0, f"{v}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        left += v
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"Real fraud cases in the referee set (n = {total})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=len(counts.columns), fontsize=9)
    ax.set_title(title)
    ax.grid(False)
    return save(fig, path)


def p3_fraud_profiles(tab: pd.DataFrame, colors: dict, path, title):
    """tab: index = indicator, columns = datasets (shares in %, already scaled)."""
    fig, ax = plt.subplots(figsize=(16, 5.8), constrained_layout=True)
    _grouped(ax, list(tab.index), {c: tab[c].to_numpy() for c in tab.columns}, colors,
             "Share of the dataset's fraud rows (%)", "Fraud indicator", label_size=7.5)
    ax.set_ylim(0, tab.to_numpy().max() * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(tab.columns), fontsize=9.5)
    ax.set_title(title)
    return save(fig, path)
