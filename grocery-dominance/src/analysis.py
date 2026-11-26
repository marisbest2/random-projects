"""H3 hex winners with k=1 neighbour smoothing, and revenue allocation."""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from config import DATA, FLAT_RATE_MAX_SQFT, H3_RES, INDEPENDENT_SALES_PER_SQFT


def add_cells(df: pd.DataFrame, res: int = H3_RES) -> pd.DataFrame:
    df = df.copy()
    df["h3"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(df["lat"], df["lon"])]
    return df


def hex_winners(df: pd.DataFrame, key: str, weight: str | None = None) -> pd.DataFrame:
    """Winner per drawn hex = argmax of summed weight over the hex + its 6 neighbours.

    Only hexes containing >= 1 store are drawn. Returns hex, winner, share (winner's share
    of the smoothed local total), total, n_stores (in the hex itself).
    """
    w = df[weight] if weight else pd.Series(1.0, index=df.index)
    cell = pd.DataFrame({"h3": df["h3"], "key": df[key], "w": w.astype(float)})
    cell = cell.groupby(["h3", "key"], as_index=False)["w"].sum()
    # Neighbourhood is symmetric: a cell's weight counts toward every hex in its disk.
    disk = {c: h3.grid_disk(c, 1) for c in cell["h3"].unique()}
    drawn = set(df["h3"])
    rows_h, rows_k, rows_w = [], [], []
    for c, k, v in cell.itertuples(index=False):
        for n in disk[c]:
            if n in drawn:
                rows_h.append(n); rows_k.append(k); rows_w.append(v)
    s = pd.DataFrame({"hex": rows_h, "key": rows_k, "w": rows_w}).groupby(["hex", "key"], as_index=False)["w"].sum()
    # deterministic tie-break: larger national total wins
    national = cell.groupby("key")["w"].sum()
    s["nat"] = s["key"].map(national)
    s = s.sort_values(["hex", "w", "nat"], ascending=[True, False, False])
    tot = s.groupby("hex")["w"].sum()
    top = s.drop_duplicates("hex").set_index("hex")
    out = pd.DataFrame({"winner": top["key"], "w": top["w"], "total": tot})
    out["share"] = out["w"] / out["total"]
    out["n_stores"] = df.groupby("h3").size().reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


# ---------------------------------------------------------------------------- revenue
def load_revenue() -> pd.DataFrame:
    rev = pd.read_csv(DATA / "parent_revenue.csv")
    rev = rev[pd.to_numeric(rev["revenue_usd_billions"], errors="coerce").notna()].copy()
    rev["revenue"] = rev["revenue_usd_billions"].astype(float) * 1e9
    return rev


def allocate_revenue(stores: pd.DataFrame, rev: pd.DataFrame,
                     indep_per_sqft: float = INDEPENDENT_SALES_PER_SQFT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spread each listed parent's revenue over its stores by sqft; flat $/sqft otherwise.

    Returns (stores with `revenue`, per-parent sanity table with implied $/sqft).
    """
    df = stores.copy()
    # Where Overture is known to miss many of a chain's stores, only the covered share of its
    # revenue is spread (otherwise each mapped store would absorb the missing stores' sales).
    n_data = df.groupby("parent").size()
    cnt = pd.to_numeric(rev.set_index("parent").get("us_store_count"), errors="coerce")
    coverage = (n_data.reindex(cnt.index) / cnt).clip(upper=1).fillna(1.0)
    rmap = rev.set_index("parent")["revenue"] * coverage
    tot_sqft = df.groupby("parent")["sqft"].transform("sum")
    listed = df["parent"].isin(rmap.index)
    # Flat-rate stores: footprints above FLAT_RATE_MAX_SQFT are mostly mixed-use buildings or
    # mis-tagged warehouses, so the sales-generating floor area is capped there.
    flat_sqft = df["sqft"].clip(upper=FLAT_RATE_MAX_SQFT)
    df["revenue"] = np.where(listed, df["parent"].map(rmap) * df["sqft"] / tot_sqft, flat_sqft * indep_per_sqft)
    df["revenue_basis"] = np.where(listed, "parent revenue x sqft share", f"flat ${indep_per_sqft:.0f}/sqft")
    g = df[listed].groupby("parent").agg(stores=("id", "size"), sqft=("sqft", "sum"))
    g["revenue_bn"] = rmap.reindex(g.index) / 1e9
    g["coverage_adj"] = coverage.reindex(g.index).round(2)
    g["implied_per_sqft"] = g["revenue_bn"] * 1e9 / g["sqft"]
    g["avg_store_sqft"] = g["sqft"] / g["stores"]
    g["rev_per_store_m"] = g["revenue_bn"] * 1e3 / g["stores"]
    g["flag"] = np.where(g["implied_per_sqft"].between(300, 900), "", "CHECK")
    return df, g.sort_values("revenue_bn", ascending=False)
