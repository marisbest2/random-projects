"""Data acquisition: Overture places + Census state outlines, cached to disk."""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import duckdb
import geopandas as gpd
import pandas as pd

from config import CACHE, L48, MIN_CONFIDENCE, NON_L48, RELEASE_LIST_URL, S3_BUCKET, S3_REGION, STATES_URL

GROCERY_SQL = "(taxonomy.primary ILIKE '%grocer%' OR taxonomy.primary = 'supermarket')"
# Aldi and Walmart Supercenters are often filed under discount/department store in the
# new taxonomy. They are pulled too, flagged `strict=False`, and only used by the extensions.
RESCUE_SQL = """(taxonomy.primary IN ('discount_store','department_store','superstore')
  AND (coalesce(brand.names.primary,'') ILIKE 'aldi%' OR names.primary ILIKE 'aldi%'
       OR coalesce(brand.names.primary,'') ILIKE 'walmart%' OR names.primary ILIKE 'walmart%'))"""


def latest_release() -> str:
    path = CACHE / "release.txt"
    xml = urllib.request.urlopen(RELEASE_LIST_URL, timeout=30).read().decode()
    rels = sorted(re.findall(r"<Prefix>release/([^<]+)/</Prefix>", xml))
    rel = rels[-1]
    path.write_text(rel)
    return rel


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.sql(f"SET s3_region='{S3_REGION}'")
    return con


def places(release: str) -> pd.DataFrame:
    """Grocery-ish places in the lower 48 (one parquet per release)."""
    path = CACHE / f"places_{release}.parquet"
    if not path.exists():
        print(f"[places] scanning Overture {release} ... (cached afterwards)")
        con = connect()
        con.sql(f"""
        COPY (
          SELECT id,
                 names.primary                AS name,
                 brand.names.primary          AS brand,
                 brand.wikidata               AS brand_wikidata,
                 taxonomy.primary             AS category,
                 {GROCERY_SQL}                AS strict,
                 confidence,
                 operating_status,
                 addresses[1].region          AS region,
                 (bbox.xmin + bbox.xmax) / 2  AS lon,
                 (bbox.ymin + bbox.ymax) / 2  AS lat
          FROM read_parquet('s3://{S3_BUCKET}/release/{release}/theme=places/type=place/*')
          WHERE bbox.xmin BETWEEN {L48['xmin']} AND {L48['xmax']}
            AND bbox.ymin BETWEEN {L48['ymin']} AND {L48['ymax']}
            AND confidence >= {MIN_CONFIDENCE}
            AND addresses[1].country = 'US'
            AND ({GROCERY_SQL} OR {RESCUE_SQL})
        ) TO '{path}' (FORMAT parquet)
        """)
    df = pd.read_parquet(path)
    df = df[df["operating_status"].fillna("open") != "permanently_closed"]
    df = add_state(df)
    return df[df["state"].notna()].reset_index(drop=True)


def add_state(df: pd.DataFrame) -> pd.DataFrame:
    """State by point-in-polygon (Overture's region field mixes 'US-CA'/'CA'/null).

    Points just off the generalised 1:20m coastline snap to the nearest state within 5 km;
    anything else (Canada/Mexico/offshore) is dropped.
    """
    st = states()
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(5070)
    j = gpd.sjoin_nearest(pts, st[["STUSPS", "geometry"]], how="left", max_distance=5_000)
    j = j[~j.index.duplicated()]
    df = df.copy()
    df["state"] = j["STUSPS"].values
    return df


def states() -> gpd.GeoDataFrame:
    path = CACHE / "states_l48_5070.parquet"
    if not path.exists():
        raw = urllib.request.urlopen(STATES_URL, timeout=60).read()
        d = CACHE / "cb_2023_us_state_20m"
        zipfile.ZipFile(io.BytesIO(raw)).extractall(d)
        g = gpd.read_file(d / "cb_2023_us_state_20m.shp")
        g = g[~g["STUSPS"].isin(NON_L48)].to_crs(5070)
        g[["STUSPS", "NAME", "geometry"]].to_parquet(path)
    return gpd.read_parquet(path)
