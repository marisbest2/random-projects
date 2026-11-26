"""Shared paths and tunable parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = ROOT / "out"
DATA = ROOT / "data"
for _d in (CACHE, OUT, DATA):
    _d.mkdir(exist_ok=True)

S3_BUCKET = "overturemaps-us-west-2"
S3_REGION = "us-west-2"
RELEASE_LIST_URL = (
    f"https://{S3_BUCKET}.s3.amazonaws.com/?list-type=2&prefix=release/&delimiter=/"
)

# Lower-48 bounding box (lon/lat)
L48 = dict(xmin=-125.0, xmax=-66.5, ymin=24.3, ymax=49.5)

H3_RES = 4
MIN_CONFIDENCE = 0.5

# Square footage
MIN_SQFT = 4_000          # matches below this are treated as unreliable
MAX_SQFT = 320_000        # matches above this are treated as unreliable (malls, DCs)
BLDG_MIN_BBOX_M2 = 300    # bbox prefilter: drops houses before the spatial join
SQM_TO_SQFT = 10.7639

# Revenue
INDEPENDENT_SALES_PER_SQFT = float(os.environ.get("INDEP_SALES_PER_SQFT", 400))
FLAT_RATE_MAX_SQFT = 30_000   # cap on floor area credited to flat-rate (independent) stores

STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
NON_L48 = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}

# Overture's bucket is public: drop any AWS credentials in the environment so reads stay anonymous.
for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ.pop(_k, None)
