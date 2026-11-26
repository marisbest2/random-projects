"""Spot-check the barzonDESIGN "grocery nations" cartoon map against the Overture store data.

For each territory on the graphic we take a rough region (states or a lat/lon box), rank
banners inside it by store count and by estimated revenue, and see where the claimed
banner lands. Independents are excluded from the ranking (each is a separate owner) but
their share of local revenue is reported.

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


def check(st: pd.DataFrame) -> pd.DataFrame:
    st = st.assign(key=_key(st))
    rows = []
    for label, claim, f in CLAIMS:
        r = _region(st, f)
        chains = r[r["parent"] != "Independent"]
        # a claim can name a banner (King Soopers) or a whole parent (Albertsons, Kroger)
        level = "parent" if claim in set(chains["parent"]) and claim not in set(chains["key"]) else "key"
        by_n = chains.groupby(level).size().sort_values(ascending=False)
        by_rev = chains.groupby(level)["revenue"].sum().sort_values(ascending=False)
        tot_rev = r["revenue"].sum()
        rank_n = by_n.index.get_loc(claim) + 1 if claim in by_n.index else None
        rank_rev = by_rev.index.get_loc(claim) + 1 if claim in by_rev.index else None
        if rank_rev == 1 and rank_n == 1:
            verdict = "Holds up"
        elif rank_rev == 1 or rank_n == 1:
            verdict = "Partly"
        elif rank_rev and rank_rev <= 3:
            verdict = "Contender"
        else:
            verdict = "Doesn't hold"
        rows.append(dict(
            territory=label, claimed=claim, stores_in_region=len(r),
            claimed_stores=int(by_n.get(claim, 0)), rank_by_count=rank_n,
            leader_by_count=by_n.index[0] if len(by_n) else "", leader_count=int(by_n.iloc[0]) if len(by_n) else 0,
            claimed_rev_share=by_rev.get(claim, 0) / tot_rev if tot_rev else 0, rank_by_revenue=rank_rev,
            leader_by_revenue=by_rev.index[0] if len(by_rev) else "",
            leader_rev_share=by_rev.iloc[0] / tot_rev if len(by_rev) else 0,
            independent_rev_share=r.loc[r["parent"] == "Independent", "revenue"].sum() / tot_rev if tot_rev else 0,
            verdict=verdict,
        ))
    return pd.DataFrame(rows)


VERDICT_C = {"Holds up": "#2E7D32", "Partly": "#9CCC65", "Contender": "#F5A623", "Doesn't hold": "#C0392B"}


def figure(df: pd.DataFrame, rel: str, fname: str = "08_spotcheck_barzon_map.png"):
    d = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 15))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.9, bottom=0.05)
    y = np.arange(len(d))
    ax.barh(y, d["leader_rev_share"] * 100, color="#E4E4E4", height=0.7, label="Revenue leader's share")
    cols = [VERDICT_C[v] for v in d["verdict"]]
    ax.barh(y, d["claimed_rev_share"] * 100, color=cols, height=0.7)
    for yi, row in d.iterrows():
        if row["rank_by_revenue"] == 1 and row["rank_by_count"] == 1:
            txt = "leads on revenue and stores"
        elif row["rank_by_revenue"] == 1:
            txt = f"leads on revenue · #{row['rank_by_count']} by stores ({row['leader_by_count']} leads)"
        elif row["rank_by_count"] == 1:
            txt = f"leads on stores · #{row['rank_by_revenue']} by revenue ({row['leader_by_revenue']} leads)"
        else:
            txt = f"#{row['rank_by_revenue'] or '–'} by revenue · {row['leader_by_revenue']} leads"
        ax.text(max(row["leader_rev_share"], row["claimed_rev_share"]) * 100 + 0.8, yi, txt,
                va="center", fontsize=9.5, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  ({c})" for t, c in zip(d["territory"], d["claimed"])], fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of estimated grocery revenue in the territory (%)", color=SUB)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    n_ok = (df["verdict"] == "Holds up").sum()
    fig.text(0.02, 0.965, "Fact-checking the grocery-nations cartoon map", fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.935, f"{n_ok} of {len(df)} territories hold up: the claimed chain leads its area by both store count and "
             "estimated revenue.\nPartly = leads on only one of the two. Contender = top 3 by revenue. "
             "Bar = claimed chain's revenue share; grey = the leader's.",
             fontsize=12, color=SUB, va="top")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in VERDICT_C.items()] + [Patch(color="#E4E4E4", label="Leader's share")],
              loc="lower right", frameon=False, fontsize=10.5)
    fig.text(0.02, 0.012, f"Data: Overture Maps ({rel}). Regions are rough state / lat-lon boxes. "
             "Independents excluded from rankings.", fontsize=9, color=MUTED)
    return viz.save(fig, fname)


def main():
    st = pd.read_parquet(CACHE / "stores_final.parquet")
    rel = (CACHE / "release.txt").read_text().strip()
    df = check(st)
    df.to_csv(OUT / "spotcheck_barzon_map.csv", index=False)
    figure(df, rel)
    pd.set_option("display.width", 250)
    print(df[["territory", "claimed", "claimed_stores", "rank_by_count", "leader_by_count", "rank_by_revenue",
              "leader_by_revenue", "claimed_rev_share", "leader_rev_share", "verdict"]].round(2).to_string(index=False))
    print(df["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
