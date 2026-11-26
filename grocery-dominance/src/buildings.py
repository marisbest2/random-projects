"""Store square footage from Overture building footprints.

Pipeline (all cached):
 1. Read the parquet footers of theme=buildings and keep row groups whose bbox
    statistics overlap the lower 48 *and* contain at least one store point.
 2. For each such row group read only `id` + `bbox`, drop small footprints by bbox
    size (houses), keep buildings whose bbox contains a store, and only then fetch
    `geometry` for that row group.
 3. DuckDB range-joins stores to the candidate building bboxes and confirms with
    ST_Contains. Footprint area is computed in EPSG:5070 with pyproj (always_xy).
"""
from __future__ import annotations

import concurrent.futures as cf
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

from config import BLDG_MIN_BBOX_M2, CACHE, L48, MAX_SQFT, MIN_SQFT, S3_BUCKET, S3_REGION, SQM_TO_SQFT

_S3 = pafs.S3FileSystem(anonymous=True, region=S3_REGION)
_TO_5070 = Transformer.from_crs(4326, 5070, always_xy=True)


def _rowgroup_index(release: str) -> pd.DataFrame:
    path = CACHE / f"building_rowgroups_{release}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sel = pafs.FileSelector(f"{S3_BUCKET}/release/{release}/theme=buildings/type=building")
    files = [i.path for i in _S3.get_file_info(sel) if i.type == pafs.FileType.File]

    def stats(f):
        md = pq.read_metadata(f, filesystem=_S3)
        names = [md.row_group(0).column(j).path_in_schema for j in range(md.row_group(0).num_columns)]
        ix = {k: names.index(f"bbox.{k}") for k in ("xmin", "xmax", "ymin", "ymax")}
        rows = []
        for r in range(md.num_row_groups):
            rg = md.row_group(r)
            rows.append(dict(
                file=f, rg=r, n=rg.num_rows,
                xmin=rg.column(ix["xmin"]).statistics.min, xmax=rg.column(ix["xmax"]).statistics.max,
                ymin=rg.column(ix["ymin"]).statistics.min, ymax=rg.column(ix["ymax"]).statistics.max,
            ))
        return rows

    with cf.ThreadPoolExecutor(32) as ex:
        df = pd.DataFrame([r for rows in ex.map(stats, files) for r in rows])
    df.to_parquet(path)
    return df


def _candidate_rowgroups(rgs: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    rgs = rgs[(rgs.xmax > L48["xmin"]) & (rgs.xmin < L48["xmax"]) & (rgs.ymax > L48["ymin"]) & (rgs.ymin < L48["ymax"])]
    order = np.argsort(pts["lon"].values)
    lon, lat = pts["lon"].values[order], pts["lat"].values[order]
    keep = []
    for xmin, xmax, ymin, ymax in rgs[["xmin", "xmax", "ymin", "ymax"]].itertuples(index=False):
        a, b = np.searchsorted(lon, [xmin, xmax])
        keep.append(bool(((lat[a:b] >= ymin) & (lat[a:b] <= ymax)).any()))
    return rgs[np.array(keep)]


def _scan_rowgroup(task, pts_lon, pts_lat, meta_cache):
    """Return candidate buildings (id, bbox, wkb) in one row group that may contain a store."""
    f, rg = task
    pf = pq.ParquetFile(f, filesystem=_S3, metadata=meta_cache.get(f))
    t = pf.read_row_group(rg, columns=["id", "bbox"])
    bb = t.column("bbox").combine_chunks()
    xmin, xmax = bb.field("xmin").to_numpy(), bb.field("xmax").to_numpy()
    ymin, ymax = bb.field("ymin").to_numpy(), bb.field("ymax").to_numpy()
    # bbox size prefilter (metres, equirectangular approx) -> drops houses / sheds
    w = (xmax - xmin) * 111_320 * np.cos(np.radians((ymin + ymax) / 2))
    h = (ymax - ymin) * 110_574
    big = np.flatnonzero(w * h >= BLDG_MIN_BBOX_M2)
    if big.size == 0:
        return None
    # does any store fall inside the bbox? (stores pre-sorted by lon)
    a = np.searchsorted(pts_lon, xmin[big], "left")
    b = np.searchsorted(pts_lon, xmax[big], "right")
    hit = np.zeros(big.size, bool)
    for k in np.flatnonzero(b > a):
        la = pts_lat[a[k]:b[k]]
        hit[k] = ((la >= ymin[big[k]]) & (la <= ymax[big[k]])).any()
    sel = big[hit]
    if sel.size == 0:
        return None
    g = pf.read_row_group(rg, columns=["geometry"]).column("geometry")
    idx = pa.array(sel)
    return pa.table({
        "bid": pc.take(t.column("id"), idx),
        "xmin": xmin[sel], "xmax": xmax[sel], "ymin": ymin[sel], "ymax": ymax[sel],
        "wkb": pc.take(g, idx),
    })


def candidate_buildings(release: str, pts: pd.DataFrame, workers: int = 48) -> pd.DataFrame:
    path = CACHE / f"building_candidates_{release}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rgs = _candidate_rowgroups(_rowgroup_index(release), pts)
    print(f"[buildings] {len(rgs):,} row groups ({rgs.n.sum()/1e6:.0f}M buildings) touch a store; scanning ...")
    meta = {f: pq.read_metadata(f, filesystem=_S3) for f in rgs.file.unique()}
    order = np.argsort(pts["lon"].values)
    plon, plat = pts["lon"].values[order], pts["lat"].values[order]
    tasks = list(rgs[["file", "rg"]].itertuples(index=False, name=None))
    out, t0 = [], time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_scan_rowgroup, t, plon, plat, meta) for t in tasks]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            if r is not None:
                out.append(r)
            if i % 500 == 0:
                print(f"  {i:,}/{len(tasks):,} row groups  {time.time()-t0:,.0f}s")
    df = pa.concat_tables(out).to_pandas()
    df.to_parquet(path)
    return df


def match_stores(release: str, stores: pd.DataFrame) -> pd.DataFrame:
    """One row per store id with matched building id and footprint sqft (NaN if none)."""
    path = CACHE / f"store_buildings_{release}.parquet"
    if path.exists():
        m = pd.read_parquet(path)
        if set(stores["id"]) <= set(m["id"]):
            return m
    pts = stores[["id", "lon", "lat"]].drop_duplicates("id")
    cand = candidate_buildings(release, pts)
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    con.register("p", pts)
    con.register("b", cand)
    j = con.sql("""
        SELECT p.id, b.bid, b.wkb::BLOB AS wkb
        FROM p JOIN b
          ON p.lon BETWEEN b.xmin AND b.xmax AND p.lat BETWEEN b.ymin AND b.ymax
        WHERE ST_Contains(ST_GeomFromWKB(b.wkb), ST_Point(p.lon, p.lat))
    """).df()
    geoms = shapely.from_wkb([bytes(b) for b in j["wkb"]])
    geoms = shapely.transform(geoms, lambda xy: np.column_stack(_TO_5070.transform(xy[:, 0], xy[:, 1])))
    j["bldg_sqft"] = shapely.area(geoms) * SQM_TO_SQFT
    # If a point sits in several footprints keep the smallest (most specific) one.
    j = j.sort_values("bldg_sqft").drop_duplicates("id")[["id", "bid", "bldg_sqft"]]
    m = pts[["id"]].merge(j, on="id", how="left")
    m.to_parquet(path)
    return m


# Typical gross store sizes (sqft) used only by --skip-buildings fast passes.
TYPICAL_SQFT = {
    "Walmart": 160_000, "Kroger": 65_000, "Albertsons": 50_000, "Ahold Delhaize": 40_000,
    "Publix": 48_000, "H-E-B": 90_000, "Aldi": 20_000, "Lidl": 30_000, "Trader Joe's": 12_000,
    "Amazon": 40_000, "Meijer": 190_000, "Hy-Vee": 80_000, "Wegmans": 100_000, "Giant Eagle": 60_000,
    "Save A Lot": 15_000, "Sprouts": 25_000, "Southeastern Grocers": 45_000, "Wakefern": 55_000,
    "Independent": 12_000,
}


def add_sqft(stores: pd.DataFrame, matches: pd.DataFrame | None) -> pd.DataFrame:
    """Attach `sqft` (+ `sqft_source`). Unreliable/unmatched -> chain median of reliable matches.

    Chains with <5 reliable matches fall back to the median over all reliable stores of
    their format (chains vs independents).
    """
    df = stores.copy()
    if matches is None:
        df["bldg_sqft"] = np.nan
    else:
        df = df.merge(matches[["id", "bldg_sqft"]], on="id", how="left")
    ok = df["bldg_sqft"].between(MIN_SQFT, MAX_SQFT)
    df["sqft_source"] = np.where(ok, "footprint", np.where(df["bldg_sqft"].notna(), "unreliable->median", "unmatched->median"))
    rel = df[ok]
    med = rel.groupby("parent")["bldg_sqft"].agg(["median", "size"])
    med = med.loc[med["size"] >= 5, "median"]
    indep = df["parent"].eq("Independent")
    fallback = np.where(indep, rel.loc[rel.parent.eq("Independent"), "bldg_sqft"].median() if ok.any() else 15_000,
                        rel.loc[~rel.parent.eq("Independent"), "bldg_sqft"].median() if ok.any() else 40_000)
    chain_med = df["parent"].map(med)
    df["sqft"] = np.where(ok, df["bldg_sqft"], chain_med.fillna(pd.Series(fallback, index=df.index)))
    if matches is None:  # --skip-buildings: typical format sizes from industry norms
        df["sqft"] = df["parent"].map(TYPICAL_SQFT).fillna(35_000)
        df["sqft_source"] = "assumed"
    return df
