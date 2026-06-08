#!/usr/bin/env python3
"""Render recall/QPS frontier and memory plots from ann_frontier.py JSON output.

    python benchmarks/plot_frontier.py benchmarks/frontier_dbpedia100k.json \
        benchmarks/frontier_dbpedia1m.json --outdir docs

For each input JSON it writes a recall/QPS Pareto SVG (frontier_<stem>.svg) and
a memory bar SVG (memory_<stem>.svg). Run in the bench venv (matplotlib).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantal-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e6e8eb"
BG = "#ffffff"

# Order also sets legend order; quantal first so it reads as the subject.
STYLE = {
    "quantal":     {"color": "#1f77b4", "label": "quantal",     "lw": 2.6, "z": 6, "ms": 5},
    "hnswlib":     {"color": "#d62728", "label": "hnswlib",     "lw": 1.7, "z": 4, "ms": 4},
    "faiss-hnsw":  {"color": "#ff7f0e", "label": "FAISS HNSW",  "lw": 1.7, "z": 4, "ms": 4},
    "faiss-ivfpq": {"color": "#9467bd", "label": "FAISS IVFPQ", "lw": 1.5, "z": 3, "ms": 4},
    "turbovec":    {"color": "#2ca02c", "label": "turbovec",    "lw": 1.7, "z": 4, "ms": 4},
}


def _style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def _corpus(data):
    n = data.get("n_train", 0)
    return f"{n/1e6:.0f}M" if n >= 1_000_000 else f"{n/1e3:.0f}k"


def _name(data):
    ds = data.get("dataset", "").lower()
    if "dbpedia" in ds:
        return "DBpedia-1536"
    if "glove" in ds:
        return "GloVe-100"
    return data.get("dataset", "dataset")


def frontier_plot(data, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    _style_ax(ax)
    xmin = 1.0
    for name, res in data["indexes"].items():
        st = STYLE.get(name, {"color": "#888", "label": name, "lw": 1.5, "z": 3, "ms": 4})
        pts = sorted(res["points"], key=lambda p: p["recall"])
        xs = [p["recall"] for p in pts]
        ys = [p["qps"] for p in pts]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", ms=st["ms"], color=st["color"], lw=st["lw"],
                zorder=st["z"], label=st["label"])
        # Track a sensible left bound from the real contenders (skip IVFPQ's
        # low-recall plateau so it doesn't squash the useful region).
        if name != "faiss-ivfpq":
            xmin = min(xmin, min(xs))

    ax.set_yscale("log")
    ax.set_xlim(max(0.80, xmin - 0.01), 1.002)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(f"recall@{data['k']}  (higher is better)", color=INK, fontsize=10)
    ax.set_ylabel("queries / sec, single thread (log, higher is better)", color=INK, fontsize=10)
    ax.set_title(f"{_name(data)}, {_corpus(data)} vectors: recall vs QPS",
                 color=INK, loc="left", fontweight="bold", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def memory_plot(data, out):
    rows = []
    for name, res in data["indexes"].items():
        if not res.get("bytes"):
            continue
        best = max((p["recall"] for p in res["points"]), default=0.0)
        rows.append((name, res["bytes"] / 1e6, best))
    rows.sort(key=lambda r: r[1])  # smallest memory at top

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    _style_ax(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.grid(False, axis="y")
    names = [STYLE.get(n, {}).get("label", n) for n, _, _ in rows]
    sizes = [mb for _, mb, _ in rows]
    colors = [STYLE.get(n, {}).get("color", "#888") for n, _, _ in rows]
    bars = ax.barh(names, sizes, color=colors, zorder=3)
    for (n, mb, rec), b in zip(rows, bars):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f"{mb:,.0f} MB  (max recall {rec:.3f})", va="center", color=MUTED, fontsize=8.5)
    ax.set_xlim(0, max(sizes) * 1.35)
    ax.set_xlabel("index size on disk, MB  (lower is better)", color=INK, fontsize=10)
    ax.set_title(f"{_name(data)}, {_corpus(data)} vectors: index memory",
                 color=INK, loc="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for jpath in args.jsons:
        data = json.load(open(jpath))
        stem = Path(jpath).stem.replace("frontier_", "")
        frontier_plot(data, outdir / f"frontier_{stem}.svg")
        memory_plot(data, outdir / f"memory_{stem}.svg")


if __name__ == "__main__":
    main()
