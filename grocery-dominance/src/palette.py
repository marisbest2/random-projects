"""Many-colour palettes where neighbouring regions never get look-alike colours.

The replication map has dozens of winners. Brand colours are used where they exist, but
a winner only keeps its brand colour if it is clearly different (in CIELAB) from every
winner it borders that already has a colour; otherwise it gets the candidate colour
that is farthest from its neighbours.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

# ~48 distinct, mid-saturation colours (no greys, nothing too pale to see on white).
CANDIDATES = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1",
    "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#fdae6b", "#a1d99b", "#de9ed6", "#ce6dbd",
    "#e7ba52", "#ad494a", "#5254a3", "#8ca252", "#bd9e39", "#d6616b", "#a55194", "#6b6ecf", "#b5cf6b",
    "#e7969c", "#9c9ede", "#cedb9c", "#e7cb94", "#00a3a1", "#f4d03f", "#ff6f61", "#00674b", "#c2185b",
    "#4a148c", "#7a8b2b", "#0050aa",
]


def _extended_candidates(min_sep: float = 9.0) -> list[str]:
    """Base list plus an HLS grid, dropping colours too close to one already kept."""
    import colorsys
    from matplotlib.colors import to_hex

    grid = [to_hex(colorsys.hls_to_rgb(h / 360, l, sat))
            for l, sat in ((0.45, 0.75), (0.32, 0.65), (0.62, 0.70), (0.25, 0.55))
            for h in range(0, 360, 12)]
    out: list[str] = []
    for c in CANDIDATES + grid:
        if all(_dist(c, o) >= min_sep for o in out):
            out.append(c)
    return out


def _lab(hex_color: str) -> np.ndarray:
    rgb = np.array(to_rgb(hex_color))
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ rgb
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _dist(a: str, b: str) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def winner_adjacency(w: pd.DataFrame) -> dict[str, set]:
    """Which winners share a hex edge with which other winners."""
    owner = dict(zip(w["hex"], w["winner"]))
    adj: dict[str, set] = {k: set() for k in w["winner"].unique()}
    for h, k in owner.items():
        for n in h3.grid_ring(h, 1):
            o = owner.get(n)
            if o is not None and o != k:
                adj[k].add(o)
    return adj


def adjacency_palette(w: pd.DataFrame, n: int, brand: dict[str, str], reserved: dict[str, str],
                      min_dist: float = 28.0) -> dict[str, str]:
    """Colour the top-n winners (by hexes won). `reserved` (Independent/Other) is never reused."""
    ranked = [k for k in w["winner"].value_counts().index if k not in reserved][:n]
    adj = winner_adjacency(w)
    pal: dict[str, str] = {}
    used: set[str] = set(reserved.values())
    for k in ranked:
        neigh = [pal[o] for o in adj.get(k, ()) if o in pal]
        avoid = neigh + list(reserved.values())

        def ok(c):
            return c not in used and all(_dist(c, a) >= min_dist for a in avoid)

        c = brand.get(k)
        if c is None or not ok(c):
            pool = [c2 for c2 in EXTENDED if all(_dist(c2, u) >= 6 for u in used)] or EXTENDED
            # farthest from the neighbours (ties broken by distance to everything used)
            c = max(pool, key=lambda c2: (min([_dist(c2, a) for a in avoid] or [999]),
                                          min([_dist(c2, u) for u in used] or [999])))
        pal[k] = c
        used.add(c)
    pal.update(reserved)
    return pal


EXTENDED = _extended_candidates()
