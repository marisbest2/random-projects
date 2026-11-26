"""Map and chart rendering (matplotlib, EPSG:5070)."""
from __future__ import annotations

from functools import lru_cache

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from pyproj import Transformer

from config import OUT

_T = Transformer.from_crs(4326, 5070, always_xy=True)
INK, SUB, MUTED = "#1a1a1a", "#555555", "#8a8a8a"
INDEPENDENT_C = "#D9C8A3"   # neutral tan
OTHER_C = "#A9B1BA"         # slate grey
GREY = "#E4E4E4"
FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "savefig.facecolor": "white", "figure.facecolor": "white"})

# Brand-ish colours. Hues kept apart where chains are neighbours.
BRAND = {
    "Walmart": "#0071CE",          # Walmart blue
    "Kroger": "#14306E",           # Kroger navy
    "Albertsons": "#00A3A1",       # Albertsons/Safeway teal
    "Ahold Delhaize": "#7B3F98",   # Giant/Food Lion purple
    "Publix": "#3E8E2F",           # Publix green
    "H-E-B": "#E2231A",            # H-E-B red
    "Aldi": "#F39C12",             # Aldi orange
    "Southeastern Grocers": "#B5122A",  # Winn-Dixie red
    "Hy-Vee": "#F06292",           # Hy-Vee red/pink
    "Meijer": "#AD1457",
    "Save A Lot": "#F4D03F",       # Save A Lot yellow
    "IGA": "#8B1E1E",              # IGA maroon
    "Piggly Wiggly": "#FF6F61",    # pig pink-red
    "Amazon": "#00674B",           # Whole Foods green
    "Wakefern": "#C0392B",         # ShopRite red
    "Giant Eagle": "#E67E22",
    "Brookshire Grocery": "#8E5A2B",
    "K-VA-T (Food City)": "#5D6D7E",
    "Fareway": "#7A8B2B",
    "Ingles": "#1565C0",
    "Trader Joe's": "#A93226",
    "Lidl": "#0050AA",
    "Northeast Grocery": "#6D4C41",
    "Chedraui USA": "#D35400",
    "Harps": "#4A148C",
    "C&S Wholesale Grocers": "#6D4C41",
    "Weis Markets": "#B71C1C",
    "WinCo": "#7CB342",
    "Save Mart": "#E53935",
    "Stater Bros": "#C0392B",
    "Sprouts": "#76B947",
    "Brookshire Brothers": "#9C640C",
    "Independent": INDEPENDENT_C,
    "Other chains": OTHER_C,
}
# Fallback hues for keys without a brand colour (replication map store names).
FALLBACK = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F",
            "#E15759", "#86BCB6", "#8CD17D", "#B6992D", "#499894", "#D37295"]


@lru_cache(maxsize=None)
def _hex_xy(cell: str) -> np.ndarray:
    lat, lng = np.array(h3.cell_to_boundary(cell)).T
    x, y = _T.transform(lng, lat)
    return np.column_stack([x, y])


def hex_polys(cells) -> list[np.ndarray]:
    return [_hex_xy(c) for c in cells]


def build_palette(ranked_keys: list[str], n: int = 14, extra: dict | None = None) -> dict:
    """Top-n keys keep their own colour; everything else -> Other chains."""
    pal, used = {}, set()
    fb = iter(FALLBACK)
    lookup = {**BRAND, **(extra or {})}
    for k in ranked_keys:
        if k == "Independent":
            continue
        if len(pal) >= n:
            break
        c = lookup.get(k)
        if c is None or c in used:
            c = next(c2 for c2 in fb if c2 not in used)
        pal[k] = c
        used.add(c)
    pal["Independent"] = INDEPENDENT_C
    pal["Other chains"] = OTHER_C
    return pal


def bucket(keys: pd.Series, pal: dict) -> pd.Series:
    return keys.where(keys.isin(pal.keys()), "Other chains")


# ------------------------------------------------------------------- figure scaffolding
L48_EXTENT = (-2_380_000, 2_290_000, 230_000, 3_200_000)


def new_map(headline: str, subtitle: str, legend: bool = True, figsize=(16, 9.6)):
    fig = plt.figure(figsize=figsize)
    map_w = 0.75 if legend else 0.98
    ax = fig.add_axes([0.01, 0.04, map_w, 0.80])
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.text(0.02, 0.955, headline, fontsize=25, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.02, 0.895, subtitle, fontsize=13, color=SUB, va="top", ha="left")
    lax = None
    if legend:
        lax = fig.add_axes([0.795, 0.06, 0.195, 0.78])
        lax.set_axis_off()
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
    return fig, ax, lax


def draw_states(ax, states, lw=0.55, color="#3a3a3a"):
    states.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=5)


def draw_hexes(ax, cells, colors, alphas=None, edge="white", lw=0.35, zorder=2):
    rgba = np.array([to_rgba(c) for c in colors])
    if alphas is not None:
        # blend toward white rather than true transparency -> crisp white borders stay white
        a = np.asarray(alphas)[:, None]
        rgba[:, :3] = rgba[:, :3] * a + (1 - a)
    pc = PolyCollection(hex_polys(cells), facecolors=rgba, edgecolors=edge, linewidths=lw, zorder=zorder)
    ax.add_collection(pc)


def set_extent(ax, ext=L48_EXTENT):
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])


def legend_rows(lax, rows: list[tuple[str, str, str]], title: str, top: float = 0.98, row_h: float = 0.052):
    """rows = [(label, color, right-aligned value)]"""
    lax.text(0.0, top, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    y = top - 0.065
    for label, color, val in rows:
        lax.add_patch(Rectangle((0.0, y - 0.017), 0.075, 0.034, facecolor=color, edgecolor="none"))
        lax.text(0.11, y, label, fontsize=11.5, color=INK, va="center")
        lax.text(1.0, y, val, fontsize=11.5, color=SUB, va="center", ha="right")
        y -= row_h
    return y


def footer(fig, text: str):
    fig.text(0.02, 0.012, text, fontsize=9, color=MUTED, ha="left", va="bottom")


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
