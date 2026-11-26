"""Build the interactive hover map: docs/grocery-dominance/{index.html,data.json}.

    python interactive.py            # needs the caches from run.py

data.json holds projected hex outlines (EPSG:5070 scaled to the SVG viewBox), state
outlines, and for every view the winner, its share and the top 3 per hex. The page
(site_template.html) is plain HTML + vanilla JS with no external libraries.
"""
from __future__ import annotations

import json
import re

import h3
import numpy as np
import pandas as pd

import analysis
import data
import palette
import run
import viz
from config import CACHE, ROOT

SITE = ROOT.parent / "docs" / "grocery-dominance"
W = 1000  # SVG viewBox width
X0, X1, Y0, Y1 = viz.L48_EXTENT
SCALE = (X1 - X0) / W
H = round((Y1 - Y0) / SCALE)


def _xy(x, y):
    return (np.asarray(x) - X0) / SCALE, (Y1 - np.asarray(y)) / SCALE


def smoothed(df: pd.DataFrame, key: str, weight: str | None, drawn: set) -> pd.DataFrame:
    """Sum of weight per (hex, key) over each drawn hex + its 6 neighbours."""
    w = df[weight] if weight else pd.Series(1.0, index=df.index)
    cell = pd.DataFrame({"h3": df["h3"], "key": df[key], "w": w.astype(float)}).groupby(["h3", "key"], as_index=False)["w"].sum()
    rows = []
    for c, k, v in cell.itertuples(index=False):
        for n in h3.grid_disk(c, 1):
            if n in drawn:
                rows.append((n, k, v))
    return pd.DataFrame(rows, columns=["hex", "key", "w"]).groupby(["hex", "key"], as_index=False)["w"].sum()


def top_table(sm: pd.DataFrame, hexes: list[str], names: dict[str, int], k: int = 3, exclude=()) -> list[list[int]]:
    """Per hex: flat [nameIdx, pct, nameIdx, pct, ...] for the top-k keys (pct of the neighbourhood total)."""
    tot = sm.groupby("hex")["w"].sum()
    s = sm[~sm["key"].isin(exclude)].sort_values(["hex", "w"], ascending=[True, False])
    s = s.groupby("hex").head(k)
    out = {}
    for h, g in s.groupby("hex"):
        flat = []
        for key, wv in zip(g["key"], g["w"]):
            flat += [names.setdefault(key, len(names)), int(round(100 * wv / tot[h]))]
        out[h] = flat
    return [out.get(h, []) for h in hexes]


def build():
    rel = (CACHE / "release.txt").read_text().strip()
    pl = data.places(rel)
    states = data.states()
    st = pd.read_parquet(CACHE / "stores_final.parquet")

    # ---- store-name view (same method + palette as figure 01)
    rep_w, _, rep_pal, rep_df = run.replication(pl, states, rel)
    # ---- parent views
    st["owner"] = np.where(st["parent"].eq("Independent"), "Independent:" + st["id"], st["parent"])
    metrics = {"count": None, "sqft": "sqft", "revenue": "revenue"}
    res = {}
    for m, col in metrics.items():
        w = analysis.hex_winners(st, "owner", col)
        w["winner"] = w["winner"].str.replace(r"^Independent:.*$", "Independent", regex=True)
        res[m] = w
    bloc = analysis.hex_winners(st, "parent", "revenue")

    hexes = sorted(set(rep_w["hex"]) | set(res["count"]["hex"]))
    hix = {h: i for i, h in enumerate(hexes)}
    drawn = set(hexes)

    # parent palette: brand colours, adjacency-aware, ~30 named
    brand = {k: v for k, v in viz.BRAND.items() if k not in ("Independent", "Other chains")}
    par_pal = palette.adjacency_palette([res["count"], res["sqft"], res["revenue"], bloc], 30, brand,
                                        {"Independent": viz.INDEPENDENT_C, "Other chains": viz.OTHER_C})

    # ---- geometry
    geo = []
    for h in hexes:
        xy = viz._hex_xy(h)
        x, y = _xy(xy[:, 0], xy[:, 1])
        geo.append([int(round(v * 10)) for pair in zip(x, y) for v in pair])
    simp = states.copy()
    simp["geometry"] = simp.geometry.simplify(2500)
    paths = []
    for g in simp.geometry:
        polys = getattr(g, "geoms", [g])
        for p in polys:
            x, y = _xy(*p.exterior.xy)
            paths.append("M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + "Z")
    # state of each hex = most common state among its stores
    hex_state = pd.concat([st[["h3", "state"]], rep_df[["h3", "state"]]]).groupby("h3")["state"].agg(lambda s: s.mode().iat[0])
    state_codes = sorted(hex_state.unique())
    n_hex_par = st.groupby("h3").size()
    n_hex_rep = rep_df.groupby("h3").size()

    names: dict[str, int] = {}

    def idx(series_map, default=-1):
        return [names.setdefault(series_map[h], len(names)) if h in series_map else default for h in hexes]

    views = {}
    # names view
    rep_sm = smoothed(rep_df, "label", None, drawn)
    rw = rep_w.set_index("hex")
    views["names"] = dict(winner=idx(rw["winner"].to_dict()),
                          share=[int(round(100 * rw["share"].get(h, 0))) for h in hexes],
                          top=top_table(rep_sm, hexes, names),
                          n=[int(n_hex_rep.get(h, 0)) for h in hexes],
                          palette=rep_pal)
    # parent views: tooltip ranks chains, with all independents shown as one combined line
    st_pooled = st.assign(pkey=st["parent"])
    for m, col in metrics.items():
        w = res[m].set_index("hex")
        sm = smoothed(st_pooled, "pkey", col, drawn)
        tot = sm.groupby("hex")["w"].sum()
        ind = sm[sm["key"] == "Independent"].set_index("hex")["w"]
        views[m] = dict(winner=idx(w["winner"].to_dict()),
                        share=[int(round(100 * w["share"].get(h, 0))) for h in hexes],
                        top=top_table(sm, hexes, names, exclude=("Independent",)),
                        indep=[int(round(100 * ind.get(h, 0) / tot[h])) if h in tot.index else 0 for h in hexes],
                        n=[int(n_hex_par.get(h, 0)) for h in hexes],
                        palette=par_pal)
    b = bloc.set_index("hex")
    sm = smoothed(st_pooled, "pkey", "revenue", drawn)
    views["bloc"] = dict(winner=idx(b["winner"].to_dict()),
                         share=[int(round(100 * b["share"].get(h, 0))) for h in hexes],
                         top=top_table(sm, hexes, names),
                         n=[int(n_hex_par.get(h, 0)) for h in hexes],
                         palette=par_pal)

    name_list = [None] * len(names)
    for k, i in names.items():
        name_list[i] = k
    for v in views.values():
        pal = v.pop("palette")
        v["colors"] = {str(i): pal[k] for k, i in names.items() if k in pal}
    out = dict(
        release=rel, viewBox=[W, H], hexes=len(hexes), geo=geo, states=paths,
        stateCodes=state_codes, state=[state_codes.index(hex_state[h]) if h in hex_state.index else -1 for h in hexes],
        names=name_list, other=viz.OTHER_C, independent=viz.INDEPENDENT_C, views=views,
    )
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    body = (ROOT / "src" / "site_template.html").read_text()
    (SITE / "index.html").write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + body.split("<!--BODY-->")[0] + "</head>\n<body>\n" + body.split("<!--BODY-->")[1] + "</body>\n</html>\n")
    print(f"wrote {SITE/'index.html'} and data.json ({(SITE/'data.json').stat().st_size/1e3:.0f} KB, {len(hexes)} hexes, {len(names)} names)")


if __name__ == "__main__":
    build()
