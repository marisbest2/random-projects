"""Spot-check the barzonDESIGN "grocery nations" cartoon map against the Overture store data.

For each territory on the graphic we take a rough region (states or a lat/lon box), rank
banners inside it by store count, square footage and estimated revenue, and see where the
claimed banner lands. Independents are excluded from shares and rankings (each is a
separate owner).

    python spotcheck.py     # needs cache/stores_final.parquet from run.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from config import CACHE, OUT
from viz import INK, MUTED, SUB

# (territory label on the graphic, claimed banner or parent, region filter)
# Region filter: dict(states=..., lat=(lo, hi), lon=(lo, hi)); boxes are rough on purpose.
CLAIMS = [
    ("The Pacific Empire of Safeway", "Safeway", dict(states={"OR", "CA"}, lat=(38.0, 46.3))),
    ("The Free State of Fred Meyer", "Fred Meyer", dict(states={"WA"}, lon=(-124, -121.5))),
    ("WinCo Demilitarized Zone", "WinCo", dict(states={"ID", "OR"}, lat=(42.0, 45.0), lon=(-118, -114.5))),
    ("The Grand Dominion of Albertsons", "Albertsons", dict(states={"ID", "MT", "WY", "UT", "NV", "NM"})),
    ("The Trader Joe's Commune", "Trader Joe's", dict(states={"CA"}, lat=(32.5, 34.6), lon=(-120, -117.0))),
    ("The Dynasty of Fry's", "Fry's", dict(states={"AZ"}, lat=(33.0, 34.0), lon=(-112.7, -111.4))),
    ("República de El Super", "El Super", dict(states={"AZ", "NM", "TX"}, lat=(31.0, 32.6), lon=(-115, -106.0))),
    ("King Sooperstan", "King Soopers", dict(states={"CO"})),
    ("The Northern Cash Wise Territory", "Cash Wise", dict(states={"ND"})),
    ("The People's Republic of Hy-Vee", "Hy-Vee", dict(states={"IA", "NE", "SD", "MN"})),
    ("The Fareway Resistance", "Fareway", dict(states={"IA"}, lat=(41.4, 42.6), lon=(-94.6, -92.4))),
    ("The Sultanate of Dillons", "Dillons", dict(states={"KS"})),
    ("Homeland Reservation", "Homeland Stores", dict(states={"OK"})),
    ("The Greater H-E-B Republic", "H-E-B", dict(states={"TX"})),
    ("The Whole Foods Resistance (Austin)", "Whole Foods", dict(states={"TX"}, lat=(30.1, 30.55), lon=(-97.95, -97.55))),
    ("The Formerly Glorious State of Walmart", "Walmart", dict(states={"AR"})),
    ("The Southern Rouses Parish", "Rouses", dict(states={"LA"})),
    ("The Piggly Wiggly Confederacy", "Piggly Wiggly", dict(states={"MS", "AL"})),
    ("The Great Southern Publix Empire", "Publix", dict(states={"FL", "GA"})),
    ("The Harris Teeter Protectorate", "Harris Teeter", dict(states={"SC"})),
    ("Winn-Dixie Disputed Territory", "Winn-Dixie", dict(states={"GA", "FL"}, lat=(30.0, 32.2), lon=(-82.5, -80.8))),
    ("The Food Lion Autonomous Zone", "Food Lion", dict(states={"NC", "VA"})),
    ("Krogervania", "Kroger", dict(states={"OH", "KY", "TN", "IN", "WV"})),
    ("Meijer Free State", "Meijer", dict(states={"MI"}, lon=(-87, -84.8))),
    ("Jewel-Osco Kingdom", "Jewel-Osco", dict(states={"IL"}, lat=(41.4, 42.5), lon=(-88.6, -87.5))),
    ("The ALDI Cartel", "Aldi", dict(states={"IL"}, lat=(39.5, 41.4))),
    ("Canton of Schnucks", "Schnucks", dict(states={"MO", "IL"}, lat=(38.3, 39.0), lon=(-90.8, -89.8))),
    ("Pick 'n Save Nation", "Pick 'n Save", dict(states={"WI"})),
    ("The Giant Eagle Federation", "Giant Eagle", dict(states={"PA"}, lon=(-80.6, -79.0))),
    ("The Sovereign Wegmans Nation", "Wegmans", dict(states={"NY"}, lat=(42.0, 45.1), lon=(-79.8, -75.5))),
    ("The Order of Price Chopper", "Price Chopper", dict(states={"VT"})),
    ("The Grand Hannaford Duchy", "Hannaford", dict(states={"NH"})),
    ("Market Basket Kingdom", "Market Basket", dict(states={"ME"})),
    ("The Holy Order of Stop & Shop", "Stop & Shop", dict(states={"CT", "RI"})),
    ("The ShopRite Separatist State", "ShopRite", dict(states={"NJ"})),
    ("The Royal Council of Weis (central PA)", "Weis", dict(states={"PA"}, lat=(39.7, 41.6), lon=(-78.5, -75.6))),
]
# Not checkable here: Costco (warehouse clubs are excluded), Carrs (Alaska) and Foodland (Hawaii) are outside
# the lower 48, and the hatched "Sprouts" area and the "Smithsylvania" sliver have no well-defined region.


def _key(st: pd.DataFrame) -> pd.Series:
    """Banner, except chains whose banners we split internally (Walmart formats)."""
    return np.where(st["parent"].isin(["Walmart", "Aldi"]), st["parent"], st["banner"])


def _region(st: pd.DataFrame, f: dict) -> pd.DataFrame:
    m = st["state"].isin(f["states"])
    if "lat" in f:
        m &= st["lat"].between(*f["lat"])
    if "lon" in f:
        m &= st["lon"].between(*f["lon"])
    return st[m]


METRICS = {"stores": None, "sqft": "sqft", "revenue": "revenue"}
METRIC_LABEL = {"stores": "Store count", "sqft": "Square footage", "revenue": "Est. revenue"}


def check(st: pd.DataFrame) -> pd.DataFrame:
    """One row per territory: the claimed chain's share, rank and the leader under each metric.

    Shares are of the chain total in the region (independents excluded), so the three
    metrics are on the same footing as the rankings.
    """
    st = st.assign(key=_key(st))
    rows = []
    for label, claim, f in CLAIMS:
        r = _region(st, f)
        chains = r[r["parent"] != "Independent"]
        # a claim can name a banner (King Soopers) or a whole parent (Albertsons, Kroger)
        level = "parent" if claim in set(chains["parent"]) and claim not in set(chains["key"]) else "key"
        row = dict(territory=label, claimed=claim, stores_in_region=len(r),
                   claimed_stores=int((chains[level] == claim).sum()),
                   independent_store_share=(r["parent"] == "Independent").mean() if len(r) else 0)
        for m, col in METRICS.items():
            agg = (chains.groupby(level).size() if col is None else chains.groupby(level)[col].sum()).sort_values(ascending=False)
            tot = agg.sum()
            row[f"share_{m}"] = agg.get(claim, 0) / tot if tot else 0
            row[f"rank_{m}"] = agg.index.get_loc(claim) + 1 if claim in agg.index else None
            row[f"leader_{m}"] = agg.index[0] if len(agg) else ""
            row[f"leader_share_{m}"] = agg.iloc[0] / tot if tot else 0
        leads = sum(row[f"rank_{m}"] == 1 for m in METRICS)
        row["leads_on"] = leads
        if leads == 3:
            row["verdict"] = "Holds up"
        elif leads >= 1:
            row["verdict"] = "Partly"
        elif row["rank_revenue"] and row["rank_revenue"] <= 3:
            row["verdict"] = "Contender"
        else:
            row["verdict"] = "Doesn't hold"
        rows.append(row)
    return pd.DataFrame(rows)


VERDICT_C = {"Holds up": "#2E7D32", "Partly": "#B8860B", "Contender": "#D35400", "Doesn't hold": "#C0392B"}
RANK_C = {1: "#2E7D32", 2: "#F2C230", 3: "#C0392B"}   # 1st / 2nd / 3rd or worse


def rank_color(rank) -> str:
    return RANK_C[1] if rank == 1 else RANK_C[2] if rank == 2 else RANK_C[3]


def figure(df: pd.DataFrame, rel: str, fname: str = "08_spotcheck_barzon_map.png"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    d = df.iloc[::-1].reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(16, 22))
    fig.subplots_adjust(left=0.265, right=0.70, top=0.905, bottom=0.04)
    h = 0.26
    offs = {"stores": h, "sqft": 0.0, "revenue": -h}
    for m, off in offs.items():
        ax.barh(np.arange(n) + off, d[f"share_{m}"] * 100, height=h * 0.9,
                color=[rank_color(r) for r in d[f"rank_{m}"]], edgecolor="white", linewidth=0.5)
        # name the three bars once, on the top row
        ax.text(d[f"share_{m}"].iloc[-1] * 100 + 0.6, n - 1 + off, {"stores": "stores", "sqft": "sq ft", "revenue": "revenue"}[m],
                va="center", fontsize=8.5, color=SUB)
        # leader's share as a tick where someone else leads
        other = d[f"rank_{m}"] != 1
        ax.scatter(d.loc[other, f"leader_share_{m}"] * 100, (np.arange(n) + off)[other], marker="|", s=130,
                   color="#333333", linewidths=1.6, zorder=3)
    # right-hand column: ranks under each metric and who leads where the claim doesn't
    tx = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, n - 0.2, "Rank: stores / sq ft / revenue      Verdict", transform=tx, fontsize=10, fontweight="bold", color=INK, va="bottom")
    for yi, row in d.iterrows():
        ranks = " / ".join(f"#{int(row[f'rank_{m}'])}" if row[f"rank_{m}"] == row[f"rank_{m}"] else "–" for m in METRICS)
        others = list(dict.fromkeys(row[f"leader_{m}"] for m in METRICS if row[f"rank_{m}"] != 1))
        ax.text(1.02, yi, ranks, transform=tx, va="center", fontsize=10.5, color=INK, family="DejaVu Sans Mono")
        ax.text(1.29, yi, row["verdict"], transform=tx, va="center", fontsize=10.5, fontweight="bold",
                color=VERDICT_C[row["verdict"]])
        if others:
            ax.text(1.02, yi - 0.32, "leader: " + ", ".join(others), transform=tx, va="center", fontsize=9, color=SUB)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"{t}  ({c})" for t, c in zip(d["territory"], d["claimed"])], fontsize=10.5)
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.6, n - 0.4)
    for yi in range(n):
        ax.axhline(yi + 0.5, color="#f0f0f0", lw=0.8, zorder=0)
    ax.set_xlabel("Claimed chain's share of the territory's chain stores / floor space / estimated revenue (%)", color=SUB)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    counts = df["verdict"].value_counts()
    fig.text(0.02, 0.988, "Fact-checking the grocery-nations cartoon map", fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.966, f"{counts.get('Holds up', 0)} of {len(df)} territories hold up: the claimed chain leads its area on stores, "
             "floor space and estimated revenue.\nPartly = leads on one or two metrics · Contender = top 3 by revenue.\n"
             "Bars, top to bottom: stores, sq ft, revenue. Colour = the claimed chain's rank on that metric; | = the leader's share.", fontsize=12.5, color=SUB, va="top")
    handles = [Patch(color=RANK_C[1], label="Ranked 1st"), Patch(color=RANK_C[2], label="Ranked 2nd"),
               Patch(color=RANK_C[3], label="3rd or lower")]
    handles.append(Line2D([], [], marker="|", color="#333333", lw=0, markersize=12, markeredgewidth=1.6, label="Leader's share"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.928), ncol=8, frameon=False, fontsize=10.5,
               handlelength=1.4, columnspacing=1.3)
    fig.text(0.02, 0.01, f"Data: Overture Maps ({rel}). Regions are rough state / lat-lon boxes. "
             "Independents excluded from shares and rankings.", fontsize=9, color=MUTED)
    return viz.save(fig, fname)


def main():
    st = pd.read_parquet(CACHE / "stores_final.parquet")
    rel = (CACHE / "release.txt").read_text().strip()
    df = check(st)
    df.to_csv(OUT / "spotcheck_barzon_map.csv", index=False)
    figure(df, rel)
    pd.set_option("display.width", 250)
    print(df[["territory", "claimed", "rank_stores", "rank_sqft", "rank_revenue", "leader_stores", "leader_sqft",
              "leader_revenue", "share_revenue", "verdict"]].round(2).to_string(index=False))
    print(df["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
