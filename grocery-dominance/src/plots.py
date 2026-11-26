"""The individual figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from viz import GREY, INK, MUTED, SUB

SOURCE = "Data: Overture Maps Foundation places & buildings ({rel}); Census cb_2023_us_state_20m. H3 res-4 hexes, winner over each hex + its 6 neighbours."


def winner_map(w: pd.DataFrame, pal: dict, states, headline: str, subtitle: str, fname: str, rel: str,
               legend_title: str = "Hexes won", alpha_by_share: bool = False):
    keys = viz.bucket(w["winner"], pal)
    fig, ax, lax = viz.new_map(headline, subtitle)
    alphas = None
    if alpha_by_share:
        alphas = share_alpha(w["share"].values)
    viz.draw_hexes(ax, w["hex"], keys.map(pal), alphas)
    viz.draw_states(ax, states)
    viz.set_extent(ax)
    counts = keys.value_counts()
    order = [k for k in pal if k not in ("Independent", "Other chains") and counts.get(k, 0) > 0]
    order = sorted(order, key=lambda k: -counts[k]) + [k for k in ("Other chains", "Independent") if counts.get(k, 0) > 0]
    rows = [(k, pal[k], f"{counts[k]:,}") for k in order]
    y = viz.legend_rows(lax, rows, legend_title)
    if alpha_by_share:
        _share_ramp(lax, y - 0.02)
    viz.footer(fig, SOURCE.format(rel=rel))
    return viz.save(fig, fname)


SHARE_LO, SHARE_HI, ALPHA_MIN = 0.20, 0.60, 0.28


def share_alpha(share):
    return ALPHA_MIN + (1 - ALPHA_MIN) * np.clip((np.asarray(share) - SHARE_LO) / (SHARE_HI - SHARE_LO), 0, 1)


def _share_ramp(lax, y):
    lax.text(0.0, y, "Winner's share of local sales", fontsize=11, color=INK, va="top", fontweight="bold")
    lax.text(0.0, y - 0.035, "(Walmart blue shown; same scale for all)", fontsize=9, color=SUB, va="top")
    y -= 0.03
    y -= 0.055
    shares = np.linspace(SHARE_LO, SHARE_HI, 6)
    for i, s in enumerate(shares):
        a = share_alpha(s)
        c = np.array(plt.matplotlib.colors.to_rgb("#0071CE")) * a + (1 - a)
        lax.add_patch(plt.Rectangle((i * 0.15, y - 0.02), 0.14, 0.035, facecolor=c, edgecolor="none"))
    lax.text(0.0, y - 0.04, "≤20%", fontsize=9.5, color=SUB, va="top")
    lax.text(0.89, y - 0.04, "≥60%", fontsize=9.5, color=SUB, va="top", ha="right")


def flips_map(wc: pd.DataFrame, wr: pd.DataFrame, pal: dict, states, rel: str, fname: str):
    m = wc[["hex", "winner"]].merge(wr[["hex", "winner"]], on="hex", suffixes=("_c", "_r"))
    flip = m["winner_c"] != m["winner_r"]
    pct = flip.mean() * 100
    keys = viz.bucket(m["winner_r"], pal)
    colors = np.where(flip, keys.map(pal), GREY)
    fig, ax, lax = viz.new_map(
        f"{pct:.0f}% of the map changes hands when you follow the money",
        f"Hexes whose top grocer changes from most stores to most estimated revenue, coloured by the new winner "
        f"({flip.sum():,} of {len(m):,} hexes flipped).")
    viz.draw_hexes(ax, m["hex"], colors)
    viz.draw_states(ax, states)
    viz.set_extent(ax)
    gained = keys[flip].value_counts()
    lost = viz.bucket(m.loc[flip, "winner_c"], pal).value_counts()
    order = [k for k in gained.index if k != "Other chains"] + (["Other chains"] if "Other chains" in gained else [])
    rows = [(k, pal[k], f"+{gained[k]:,}") for k in order[:16]]
    y = viz.legend_rows(lax, rows, "Hexes gained")
    lax.add_patch(plt.Rectangle((0.0, y - 0.017), 0.075, 0.034, facecolor=GREY, edgecolor="none"))
    lax.text(0.11, y, "No change", fontsize=11.5, color=INK, va="center")
    # biggest losers
    y -= 0.08
    lax.text(0.0, y, "Biggest losers", fontsize=12.5, fontweight="bold", color=INK, va="top")
    y -= 0.06
    for k, v in lost.head(4).items():
        lax.text(0.0, y, k, fontsize=11.5, color=INK, va="center")
        lax.text(1.0, y, f"−{v:,}", fontsize=11.5, color=SUB, va="center", ha="right")
        y -= 0.045
    viz.footer(fig, SOURCE.format(rel=rel))
    return viz.save(fig, fname), pct, gained, lost


def share_bars(results: dict[str, pd.DataFrame], pal: dict, fname: str, rel: str, top: int = 15):
    shares = pd.DataFrame({m: viz.bucket(w["winner"], pal).value_counts(normalize=True) * 100 for m, w in results.items()}).fillna(0)
    order = shares.max(axis=1).sort_values(ascending=False).index.tolist()
    order = [k for k in order if k not in ("Other chains",)][:top] + ["Other chains"]
    shares = shares.loc[[k for k in order if k in shares.index]]
    metrics = list(results)
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.08)
    n = len(shares)
    yy = np.arange(n)[::-1]
    h = 0.26
    shades = [0.45, 0.72, 1.0]
    for j, m in enumerate(metrics):
        cols = []
        for k in shares.index:
            base = np.array(plt.matplotlib.colors.to_rgb(pal[k]))
            cols.append(base * shades[j] + (1 - shades[j]))
        ax.barh(yy + (1 - j) * h, shares[m].values, height=h * 0.92, color=cols, edgecolor="none")
        for y0, v in zip(yy + (1 - j) * h, shares[m].values):
            if v >= 0.5:
                ax.text(v + 0.3, y0, f"{v:.0f}%", va="center", fontsize=8.5, color=SUB)
    ax.set_yticks(yy)
    ax.set_yticklabels(shares.index, fontsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of drawn hexes won (%)", fontsize=11, color=SUB)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.text(0.02, 0.96, "Who wins depends on what you count", fontsize=23, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, "Share of hexes won by each parent company. Bars per chain, light → dark: "
             + " → ".join(metrics) + ".", fontsize=12.5, color=SUB, va="top")
    # tiny key
    from matplotlib.patches import Patch
    hd = [Patch(facecolor=np.array([0.3, 0.3, 0.3]) * s + (1 - s), label=m) for m, s in zip(metrics, shades)]
    ax.legend(handles=hd, loc="lower right", frameon=False, fontsize=11)
    fig.text(0.02, 0.012, SOURCE.format(rel=rel), fontsize=8.5, color=MUTED)
    return viz.save(fig, fname), shares


def empire_extent(cells, pad=0.12, aspect=(0.75 * 16) / (0.80 * 9.6)):
    """Window around the chain's core hexes, matching the map axes' aspect ratio."""
    xy = np.vstack(viz.hex_polys(cells))
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    span_x = max((x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad) * aspect, 1_900_000)
    span_y = span_x / aspect
    return (cx - span_x / 2, cx + span_x / 2, cy - span_y / 2, cy + span_y / 2)


def empire(parent: str, color: str, wr: pd.DataFrame, presence: set, all_hexes: pd.Series, states, rel: str, fname: str,
           stats: dict):
    wins = set(wr.loc[wr["winner"] == parent, "hex"])
    present = presence - wins
    cells = all_hexes.values
    col = np.where(np.isin(cells, list(wins)), color, np.where(np.isin(cells, list(present)), "__tint", GREY))
    tint = np.array(plt.matplotlib.colors.to_rgb(color)) * 0.33 + 0.67
    col = [tuple(tint) if c == "__tint" else c for c in col]
    fig, ax, lax = viz.new_map(f"The {parent} Empire",
                               f"{stats['stores']:,} stores · wins {len(wins):,} hexes by estimated revenue · present in {len(presence):,}",
                               legend=True, figsize=(16, 9.6))
    viz.draw_hexes(ax, cells, col, lw=0.3)
    viz.draw_states(ax, states, lw=0.6)
    ext = empire_extent(stats.get("core") or list(presence))
    viz.set_extent(ax, ext)
    rows = [(f"{parent} wins", color, f"{len(wins):,}"), ("Present, not top", tuple(tint), f"{len(present):,}"),
            ("No stores nearby", GREY, "")]
    y = viz.legend_rows(lax, rows, "Hexes")
    if stats.get("banners"):
        y -= 0.04
        lax.text(0.0, y, "Biggest banners (stores)", fontsize=12.5, fontweight="bold", color=INK, va="top")
        y -= 0.06
        for b, n in stats["banners"]:
            lax.text(0.0, y, b, fontsize=11.5, color=INK, va="center")
            lax.text(1.0, y, f"{n:,}", fontsize=11.5, color=SUB, va="center", ha="right")
            y -= 0.045
    viz.footer(fig, SOURCE.format(rel=rel))
    return viz.save(fig, fname)


def bloc_triptych(results: dict[str, pd.DataFrame], pal: dict, states, rel: str, fname: str):
    """Three small maps: winners when every independent store counts as one company."""
    fig = plt.figure(figsize=(24, 9))
    fig.text(0.012, 0.97, "What if every independent grocer were one company?", fontsize=27,
             fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, "Same maps, but every unbranded store is pooled into a single \"Independent\" bloc "
             "that competes against the chains", fontsize=14, color=SUB, va="top")
    counts = {}
    for i, (m, w) in enumerate(results.items()):
        ax = fig.add_axes([0.005 + i * 0.332, 0.15, 0.325, 0.66])
        ax.set_axis_off()
        ax.set_aspect("equal")
        keys = viz.bucket(w["winner"], pal)
        viz.draw_hexes(ax, w["hex"], keys.map(pal), lw=0.2)
        viz.draw_states(ax, states, lw=0.4)
        viz.set_extent(ax)
        vc = keys.value_counts()
        counts[m] = vc
        ind = vc.get("Independent", 0) / len(w) * 100
        raw = w["winner"].value_counts().drop("Independent", errors="ignore")
        pct = raw.iloc[0] / len(w) * 100 if len(raw) else 0
        lead = f"{raw.index[0]} {pct:.0f}%" if pct >= 1 else f"{raw.index[0]} <1%"
        ax.set_title(f"{m}\nIndependents win {ind:.0f}% of hexes · top chain: {lead}",
                     fontsize=14, color=INK, loc="left", pad=6)
    # shared horizontal legend: every colour that appears in any panel
    present = set().union(*[set(v.index) for v in counts.values()])
    order = [k for k in pal if k in present and k not in ("Other chains", "Independent")]
    order = ["Independent"] + order + (["Other chains"] if "Other chains" in present else [])
    lax = fig.add_axes([0.012, 0.045, 0.976, 0.085])
    lax.set_axis_off()
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    per_row = 8
    for j, k in enumerate(order):
        x, y = (j % per_row) / per_row, 0.75 - (j // per_row) * 0.5
        lax.add_patch(plt.Rectangle((x, y - 0.13), 0.012, 0.26, facecolor=pal[k], edgecolor="none"))
        lax.text(x + 0.017, y, k, fontsize=12.5, color=INK, va="center")
    fig.text(0.012, 0.012, SOURCE.format(rel=rel), fontsize=9.5, color=MUTED)
    return viz.save(fig, fname), counts
