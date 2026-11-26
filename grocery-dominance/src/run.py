"""Who owns each part of the US grocery map?  End-to-end pipeline.

    python run.py                    # full run (uses cached building join if present)
    python run.py --skip-buildings   # fast pass with typical store sizes instead of footprints
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

import analysis
import buildings
import data
import parents
import plots
import viz
from config import CACHE, DATA, INDEPENDENT_SALES_PER_SQFT, OUT

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)


# Display fixes for brand strings that Overture gets wrong or shouts.
NAME_ALIASES = {"Shoprite South Africa": "ShopRite", "TOPs Friendly Markets": "Tops Friendly Markets", "ALDI": "Aldi",
                "Hannaford Supermarkets": "Hannaford", "Fareway Stores": "Fareway", "Harps Food Stores": "Harps"}
REPLICATION_COLOURS = 80


def name_key(raw: pd.Series) -> pd.Series:
    """Light normalisation so spelling variants of one name count together.

    'Save-A-Lot' = 'Save A Lot', 'Piggly Wiggly - Midlands' = 'Piggly Wiggly',
    'Super 1 Foods Fuel Center' = 'Super 1 Foods', 'Ridley's Market-Spring Creek' = 'Ridley's Market'.
    """
    s = raw.fillna("").str.replace("\u2019", "'", regex=False)
    s = s.str.replace(r"\s+-\s+.*$", "", regex=True)                       # "X - Location"
    s = s.str.replace(r"^(\S+\s+\S+.*?)-(?=[A-Z]).*$", r"\1", regex=True)  # "Two Words-Location" (keeps Winn-Dixie)
    s = s.str.replace(r"\s+\S+ county, \w\w$", "", regex=True, case=False)  # "Food Giant Muhlenberg County, Ky"
    s = s.str.replace(r"\s+#?\d+$", "", regex=True)
    s = s.str.lower().str.replace(r"\s+(fuel center|liquor|pharmacy|bakery|deli|gas)$", "", regex=True)
    s = s.str.replace(r"[-.']", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def replication(pl: pd.DataFrame, states, rel: str):
    """Step 2: faithful recreation - strict grocery categories, store name/brand, by count."""
    import palette

    df = pl[pl["strict"]].copy()
    raw = df["brand"].fillna(df["name"]).fillna("").str.replace("\u2019", "'", regex=False)
    key = name_key(raw)
    df["key"] = key
    # label = most common cleaned spelling within the key
    clean = raw.str.replace(r"\s+-\s+.*$", "", regex=True).str.replace(r"^(\S+\s+\S+.*?)-(?=[A-Z]).*$", r"\1", regex=True)
    clean = clean.str.replace(r"(?i)\s+\S+ county, \w\w$", "", regex=True).str.replace(r"\s+#?\d+$", "", regex=True)
    clean = clean.str.replace(r"(?i)\s+(fuel center|liquor|pharmacy|bakery|deli|gas)$", "", regex=True).str.strip()
    label = clean.groupby(key).agg(lambda s: s.value_counts().index[0]).replace(NAME_ALIASES)
    df = analysis.add_cells(df)
    w = analysis.hex_winners(df, "key")
    w["winner"] = w["winner"].map(label)
    # brand colours where a name clearly belongs to a known chain
    brand = {}
    for k in w["winner"].unique():
        for p in viz.BRAND:
            if re.fullmatch(re.escape(p.lower()) + r"( .*)?", k.lower()):
                brand[k] = viz.BRAND[p]
    brand.update({"Walmart Supercenter": "#5FA8E8", "Walmart Neighborhood Market": "#9CC9F0", "Food Lion": "#7B3F98",
                  "Safeway": "#00A3A1", "Stop & Shop": "#9B59B6", "Winn-Dixie": "#B5122A", "Kroger": "#14306E",
                  "Brookshire's": "#8E5A2B", "Harps": "#4A148C", "United Supermarkets": "#66BB6A", "Aldi": "#F39C12",
                  "Hannaford": "#7B3F98", "Fareway": "#7A8B2B"})
    pal = palette.adjacency_palette(w, REPLICATION_COLOURS, brand, {"Other chains": viz.OTHER_C})
    w.to_parquet(CACHE / "winners_replication.parquet")
    named = viz.bucket(w["winner"], pal).ne("Other chains").mean()
    p = plots.winner_map(
        w, pal, states,
        "America's grocery map, by store name",
        f"Most common grocery store name/brand in each hex and its 6 neighbours · {len(df):,} stores · {len(w):,} hexes "
        f"· top {REPLICATION_COLOURS} names cover {named:.0%} of the map",
        "01_replication_store_name_count.png", rel, legend_cols=3, other_label="Other names")
    return w, p


def build_store_table(pl: pd.DataFrame) -> pd.DataFrame:
    a = parents.assign_parents(pl)
    kept = a[a["parent"] != parents.EXCLUDE]
    st = parents.dedupe(kept)
    # ---- audit -----------------------------------------------------------------------
    audit = (a.assign(kind=np.where(a.parent.eq(parents.EXCLUDE), "excluded",
                                    np.where(a.matched, "rule", np.where(a.parent.eq("Independent"), "independent", "branded-unmatched"))))
             .groupby(["parent", "banner", "kind"]).size().rename("records").reset_index())
    after = st.groupby(["parent", "banner"]).size().rename("stores_after_dedupe").reset_index()
    audit = audit.merge(after, on=["parent", "banner"], how="left").fillna({"stores_after_dedupe": 0})
    audit = audit.sort_values("records", ascending=False)
    audit.to_csv(OUT / "audit_parent_banner_counts.csv", index=False)
    top_unmatched = audit[audit.kind == "branded-unmatched"].head(30)
    print("\n[audit] biggest branded-but-unmatched (each is its own parent):")
    print(top_unmatched[["parent", "records"]].to_string(index=False))
    print(f"[audit] {len(a):,} records -> {len(kept):,} after exclusions -> {len(st):,} after de-dupe")
    print(f"[audit] rule-matched {a.matched.mean():.1%} of records; independents {st.parent.eq('Independent').mean():.1%} of stores")
    return analysis.add_cells(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-buildings", action="store_true", help="use typical sizes, skip the footprint join")
    ap.add_argument("--indep-per-sqft", type=float, default=INDEPENDENT_SALES_PER_SQFT)
    args = ap.parse_args()

    rel = data.latest_release()
    print(f"[release] {rel}")
    pl = data.places(rel)
    states = data.states()
    print(f"[places] {len(pl):,} candidate places ({pl.strict.sum():,} in grocery categories)")

    rep_w, _ = replication(pl, states, rel)

    st = build_store_table(pl)

    # ---- square footage ----------------------------------------------------------------
    matches = None if args.skip_buildings else buildings.match_stores(rel, pl)
    st = buildings.add_sqft(st, matches)
    if matches is not None:
        mr = st["sqft_source"].value_counts(normalize=True)
        print(f"\n[sqft] footprint match rate: {st['bldg_sqft'].notna().mean():.1%} any building, "
              f"{mr.get('footprint', 0):.1%} reliable ({buildings.MIN_SQFT:,}-{buildings.MAX_SQFT:,} sqft)")
        by = st.groupby("parent").agg(stores=("id", "size"), reliable=("sqft_source", lambda s: (s == "footprint").mean()),
                                      median_sqft=("sqft", "median"))
        by.sort_values("stores", ascending=False).head(25).to_csv(OUT / "sqft_by_parent.csv")
        print(by.sort_values("stores", ascending=False).head(20).round(2).to_string())

    # ---- revenue -------------------------------------------------------------------------
    rev = analysis.load_revenue()
    st, sanity = analysis.allocate_revenue(st, rev, args.indep_per_sqft)
    sanity.to_csv(OUT / "revenue_sanity_per_sqft.csv")
    print("\n[revenue] implied sales per sqft (flag outside $300-900):")
    print(sanity.round(1).to_string())
    st.to_parquet(CACHE / "stores_final.parquet")

    # ---- winners ---------------------------------------------------------------------------
    # Independents are thousands of separate owners, not one company: each competes on its
    # own, so "Independent" (tan) wins only where no chain out-weighs a single local store.
    st["owner"] = np.where(st["parent"].eq("Independent"), "Independent:" + st["id"], st["parent"])
    res = {}
    for m, wcol in (("By store count", None), ("By square footage", "sqft"), ("By est. revenue", "revenue")):
        w = analysis.hex_winners(st, "owner", wcol)
        w["winner"] = w["winner"].str.replace(r"^Independent:.*$", "Independent", regex=True)
        res[m] = w
    # Counterfactual: every independent pooled into one "Independent" company.
    bloc = {m: analysis.hex_winners(st, "parent", wcol)
            for m, wcol in (("By store count", None), ("By square footage", "sqft"), ("By est. revenue", "revenue"))}
    ranked = (pd.concat([w["winner"].value_counts() for w in res.values()], axis=1).max(axis=1)
              .sort_values(ascending=False).index.tolist())
    pal = viz.build_palette(ranked, n=14)
    for m, w in res.items():
        w.to_parquet(CACHE / f"winners_parent_{m.split()[-1].replace('.', '')}.parquet")

    sub = {
        "By store count": ("Who has the most stores?", "Parent company with the most grocery stores in each hex + its 6 neighbours"),
        "By square footage": ("Who has the most floor space?", "Parent company with the most grocery square footage (Overture building footprints) in each hex + neighbours"),
        "By est. revenue": ("Who takes the most money?", f"Parent company with the most estimated grocery sales in each hex + neighbours (chain revenue spread by sqft; independents at ${args.indep_per_sqft:.0f}/sqft)"),
    }
    names = {"By store count": "02a_parent_by_count.png", "By square footage": "02b_parent_by_sqft.png",
             "By est. revenue": "02c_parent_by_revenue.png"}
    for m, w in res.items():
        plots.winner_map(w, pal, states, sub[m][0], sub[m][1], names[m], rel)

    plots.winner_map(res["By est. revenue"], pal, states, "How dominant is the local leader?",
                     "Parent with the most estimated grocery sales; fainter hexes = the winner takes a smaller share of local sales",
                     "03_dominance_revenue_share.png", rel, alpha_by_share=True)
    _, pct, gained, lost = plots.flips_map(res["By store count"], res["By est. revenue"], pal, states, rel, "04_flips_count_to_revenue.png")
    _, shares = plots.share_bars(res, pal, "05_share_of_hexes_by_metric.png", rel)
    shares.round(2).to_csv(OUT / "share_of_hexes_by_metric.csv")

    _, bloc_counts = plots.bloc_triptych(bloc, pal, states, rel, "07_independents_as_one_bloc.png")
    bloc_tbl = pd.DataFrame(bloc_counts).fillna(0).astype(int).sort_values("By est. revenue", ascending=False)
    bloc_tbl.to_csv(OUT / "hexes_won_independents_as_bloc.csv")
    print("\n[independents as one bloc] hexes won\n" + bloc_tbl.head(12).to_string())

    # ---- empires ---------------------------------------------------------------------------
    wr = res["By est. revenue"]
    all_hexes = wr["hex"]
    nat = st.groupby("parent")["revenue"].sum().drop("Independent", errors="ignore").sort_values(ascending=False)
    top8 = nat.index[:8].tolist()
    # presence = hexes whose 7-hex neighbourhood includes at least one of the chain's stores
    import h3
    drawn = set(all_hexes)
    for i, p in enumerate(top8, 1):
        cells = set(st.loc[st.parent == p, "h3"])
        pres = {n for c in cells for n in h3.grid_disk(c, 1)} & drawn
        sp = st[st.parent == p]
        # zoom to the chain's core: 2nd-98th percentile of its store locations
        lo = sp[["lon", "lat"]].quantile(0.02)
        hi = sp[["lon", "lat"]].quantile(0.98)
        core = sp[sp.lon.between(lo.lon, hi.lon) & sp.lat.between(lo.lat, hi.lat)]["h3"].unique().tolist()
        stats = {"stores": len(sp), "banners": list(sp["banner"].value_counts().head(6).items()), "core": core}
        color = pal.get(p, viz.BRAND.get(p, "#333333"))
        fn = f"06_{i}_empire_{re.sub(r'[^a-z0-9]+', '_', p.lower()).strip('_')}.png"
        plots.empire(p, color, wr, pres, all_hexes, states, rel, fn, stats)

    # ---- summary tables -----------------------------------------------------------------------
    summary = pd.DataFrame({m: viz.bucket(w["winner"], pal).value_counts() for m, w in res.items()}).fillna(0).astype(int)
    summary["count→revenue"] = summary["By est. revenue"] - summary["By store count"]
    summary = summary.sort_values("By est. revenue", ascending=False)
    summary.to_csv(OUT / "hexes_won_by_metric.csv")
    print("\n[hexes won]\n" + summary.to_string())
    print(f"\n[flips] {pct:.1f}% of hexes change winner from count to revenue")
    print("gained:", gained.head(8).to_dict())
    print("lost:", lost.head(8).to_dict())
    print("\n[replication] top store names:", rep_w["winner"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
