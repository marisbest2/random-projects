"""Roll store names/brands up to parent companies with ordered regex rules.

Rules are checked top to bottom; the first match wins. Each rule is
(parent, banner, regex, mode, regions):
  mode="any"   -> regex may match the brand field OR the store name
  mode="brand" -> regex must match the brand field (used for ambiguous names
                  such as City Market, Lucky, Tops, Market Street, Price Chopper)
  regions      -> optional set of state codes the rule is restricted to
Parent "EXCLUDE" removes the record (dollar stores, pharmacies, c-stores, clubs,
store departments that duplicate the main store, etc.).
"""
from __future__ import annotations

import re

import h3
import numpy as np
import pandas as pd

EXCLUDE = "EXCLUDE"
INDEPENDENT = "Independent"

NE = {"NY", "VT", "NH", "MA", "CT", "PA", "ME"}
MID_ATL = {"MD", "VA", "DC", "DE", "WV"}

R = [
    # ---- exclusions: non-grocery formats and department/duplicate records ------------
    (EXCLUDE, "dollar store", r"dollar general|family dollar|dollar tree|99 cents only|five below|\bpopshelf", "any", None),
    (EXCLUDE, "pharmacy", r"pharmacy|\bcvs\b|walgreens|rite aid|duane reade|\brx\b", "any", None),
    (EXCLUDE, "convenience", r"7-eleven|7 eleven|circle k|\bwawa\b|sheetz|casey'?s|quiktrip|speedway|\bgetgo\b|kwik|cumberland farms|racetrac|murphy usa|love'?s travel|stewart'?s shops|royal farms|\bmapco\b|holiday station|kum & go|maverik|buc-ee|food mart|mini ?mart|\bgas\b|fuel|\bexxon|\bshell\b|chevron|valero|sunoco|marathon|\bbp\b|\bcitgo|tedeschi|convenience", "any", None),
    (EXCLUDE, "warehouse club / foodservice", r"costco|sam'?s club|\bsams club|bj'?s wholesale|gordon food service|restaurant depot|chef'?store|us foods|smart foodservice|\bcash ?& ?carry|jetro|distribut|wholesale club", "any", None),
    (EXCLUDE, "department / duplicate record", r"bakery|\bdeli\b|coffee bar|starbucks|sushi|pick ?up|delivery|liquor|wine & spirits|beer & wine|optical|vision center|floral|business center|money center|photo|auto care|\btires?\b|bishops'? storehouse|home storage center|food bank|pantry|catering|\bcafe\b|kitchen", "any", None),
    (EXCLUDE, "non-grocery", r"^target$|^target grocery$|\bmeat market\b.*official page|\bbutterball\b|tyson foods|mission foods|frito-lay|\bcbd\b|nutrition|super suppers|sandwich|hackney|^quevos$|^elvis mtz$|^market plate$|^market cocina$|cliff house classic|wic accepted|drinkfreeaf|^corner store$", "any", None),
    # Banners that no longer exist but linger in the places data.
    (EXCLUDE, "defunct banner", r"fresh ?& ?easy|bottom dollar food|sweetbay|rainbow foods|^bi-?lo\b|^farm fresh( supermarket)?$|^shoppers food", "any", None),

    # ---- Kroger ---------------------------------------------------------------------
    ("Kroger", "Kroger", r"\bkroger", "any", None),
    ("Kroger", "Ralphs", r"\bralphs\b", "any", None),
    ("Kroger", "Fred Meyer", r"fred meyer", "any", None),
    ("Kroger", "King Soopers", r"king soopers", "any", None),
    ("Kroger", "Smith's", r"smith'?s (food|marketplace)|^smith'?s$", "any", None),
    ("Kroger", "Fry's", r"fry'?s (food|marketplace|mercado)|^fry'?s$", "any", None),
    ("Kroger", "Harris Teeter", r"harris teeter", "any", None),
    ("Kroger", "Mariano's", r"mariano'?s", "any", None),
    ("Kroger", "Pick 'n Save", r"pick ?'?n'? ?save", "any", None),
    ("Kroger", "Metro Market", r"^metro market", "any", {"WI"}),
    ("Kroger", "Dillons", r"\bdillons\b", "any", None),
    ("Kroger", "QFC", r"\bqfc\b|quality food center", "any", None),
    ("Kroger", "Food 4 Less", r"food ?4 ?less|foods co\b", "any", None),
    ("Kroger", "City Market", r"^city market$", "brand", {"CO", "UT", "WY", "NM"}),
    ("Kroger", "Baker's", r"^baker'?s( supermarket)?$", "brand", {"NE"}),
    ("Kroger", "Gerbes", r"\bgerbes\b", "any", None),
    ("Kroger", "Jay C", r"\bjay ?c\b", "any", None),
    ("Kroger", "Pay Less", r"pay ?less super ?markets?", "any", {"IN"}),
    ("Kroger", "Ruler Foods", r"ruler foods", "any", None),
    ("Kroger", "Copps", r"^copps", "any", {"WI"}),
    ("Kroger", "Owen's", r"^owen'?s", "brand", None),

    # ---- Albertsons (separate from Kroger: merger blocked Dec 2024) -------------------
    ("Albertsons", "Albertsons", r"\balbertsons", "any", None),
    ("Albertsons", "Safeway", r"\bsafeway\b", "any", None),
    ("Albertsons", "Vons", r"\bvons\b", "any", None),
    ("Albertsons", "Pavilions", r"^pavilions", "any", {"CA"}),
    ("Albertsons", "Jewel-Osco", r"jewel[- ]?osco|^jewel$", "any", None),
    ("Albertsons", "Shaw's", r"shaw'?s( supermarket)?", "any", NE),
    ("Albertsons", "Star Market", r"^star market", "any", {"MA"}),
    ("Albertsons", "Acme", r"\bacme (markets?|supermarket)|^acme$", "any", NE | {"NJ", "DE", "MD"}),
    ("Albertsons", "Tom Thumb", r"tom thumb", "any", {"TX"}),
    ("Albertsons", "Randalls", r"\brandalls\b", "any", {"TX"}),
    ("Albertsons", "United Supermarkets", r"united supermarkets|united express|^united$", "any", {"TX"}),
    ("Albertsons", "Market Street", r"^market street$", "brand", {"TX", "NM"}),
    ("Albertsons", "Amigos", r"^amigos", "brand", {"TX"}),
    ("Albertsons", "Carrs", r"^carrs", "any", None),
    ("Albertsons", "Haggen", r"\bhaggen\b", "any", None),
    ("Albertsons", "Andronico's", r"andronico", "any", None),

    # ---- Ahold Delhaize (Giant Eagle must be checked before Giant) --------------------
    ("Giant Eagle", "Giant Eagle", r"giant eagle", "any", None),
    ("Giant Eagle", "Market District", r"^market district", "any", None),
    ("Ahold Delhaize", "Food Lion", r"food lion", "any", None),
    ("Ahold Delhaize", "Stop & Shop", r"stop ?(&|and|n) ?shop", "any", None),
    ("Ahold Delhaize", "The Giant Company", r"the giant company|giant food stores", "any", None),
    ("Ahold Delhaize", "Giant Food", r"^giant( food)?( stores?)?$|^giant food #", "any", MID_ATL),
    ("Ahold Delhaize", "The Giant Company", r"^giant( food)?( stores?)?$", "any", {"PA"}),
    ("Ahold Delhaize", "Martin's", r"^martin'?s( food markets?)?$", "brand", {"PA", "VA", "WV", "MD"}),
    ("Ahold Delhaize", "Hannaford", r"hannaford", "any", None),

    # ---- other majors ----------------------------------------------------------------
    ("Walmart", "Neighborhood Market", r"walmart neighborhood", "any", None),
    ("Walmart", "Supercenter / Walmart", r"wal-?mart", "any", None),
    ("Publix", "Publix", r"\bpublix", "any", None),
    ("H-E-B", "H-E-B", r"^h-?e-?b\b|\bh-e-b\b", "any", None),
    ("H-E-B", "Central Market", r"^central market", "any", {"TX"}),
    ("H-E-B", "Joe V's", r"joe v'?s", "any", None),
    ("H-E-B", "Mi Tienda", r"^mi tienda", "any", {"TX"}),
    ("Amazon", "Whole Foods", r"whole foods|^wfm\b", "any", None),
    ("Amazon", "Amazon Fresh", r"amazon fresh", "any", None),
    ("Aldi", "Aldi", r"^aldi\b", "any", None),
    # Aldi bought Southeastern Grocers in 2024 but sold Winn-Dixie/Harveys to a
    # consortium (incl. C&S) in Feb 2025, converting ~170 stores to Aldi.
    ("Southeastern Grocers", "Winn-Dixie", r"winn[- ]?dixie", "any", None),
    ("Southeastern Grocers", "Harveys", r"harvey'?s supermarket", "any", None),
    ("Wakefern", "ShopRite", r"shop ?rite", "any", None),
    ("Wakefern", "Price Rite", r"price ?rite", "any", None),
    ("Wakefern", "The Fresh Grocer", r"the fresh grocer", "any", None),
    ("Wakefern", "Fairway", r"^fairway market", "any", None),
    ("Trader Joe's", "Trader Joe's", r"trader joe", "any", None),
    ("Meijer", "Meijer", r"\bmeijer\b", "any", None),
    ("Hy-Vee", "Hy-Vee", r"hy-?vee", "any", None),
    ("Wegmans", "Wegmans", r"wegmans", "any", None),
    ("Lidl", "Lidl", r"^lidl\b", "any", None),
    ("Sprouts", "Sprouts", r"sprouts farmers", "any", None),
    ("WinCo", "WinCo", r"winco", "any", None),
    ("Save Mart", "Save Mart", r"save ?mart", "any", {"CA", "NV"}),
    ("Save Mart", "Lucky", r"^lucky( supermarkets| california)?$", "brand", {"CA", "NV"}),
    ("Save Mart", "FoodMaxx", r"food ?maxx", "any", None),
    ("Schnucks", "Schnucks", r"schnuck", "any", None),
    ("Weis Markets", "Weis", r"^weis\b", "any", None),
    ("Ingles", "Ingles", r"ingles market|^ingles$", "any", None),
    ("Raley's", "Raley's", r"raley'?s|nob hill foods|bel air market", "any", None),
    ("Stater Bros", "Stater Bros", r"stater bro", "any", None),
    ("Brookshire Brothers", "Brookshire Brothers", r"brookshire brothers", "any", None),
    ("Brookshire Grocery", "Brookshire's", r"brookshire'?s", "any", None),
    ("Brookshire Grocery", "Super 1 Foods", r"super ?1 foods", "any", {"TX", "LA", "AR"}),
    ("Miner's Inc.", "Super One Foods", r"super (one|1) foods", "any", {"MN", "WI", "MI"}),
    ("Brookshire Grocery", "Spring Market", r"^spring market", "any", {"TX", "LA", "AR"}),
    ("Brookshire Grocery", "Reasor's", r"^reasor'?s", "any", {"OK"}),
    ("Heritage Grocers", "Cardenas", r"^cardenas", "any", None),
    ("Heritage Grocers", "Tony's Fresh Market", r"^tony'?s fresh market", "any", None),
    ("Heritage Grocers", "El Rancho", r"^el rancho supermercado", "any", None),
    ("Grocery Outlet", "Grocery Outlet", r"grocery outlet", "any", None),
    ("Big Y", "Big Y", r"^big y\b", "any", None),
    ("Bashas'", "Bashas'", r"bashas", "any", None),
    ("Bashas'", "Food City (AZ)", r"^food city", "any", {"AZ"}),
    ("K-VA-T (Food City)", "Food City", r"^food city", "any", None),
    ("Harps", "Harps", r"^harps", "any", None),
    ("Fareway", "Fareway", r"fareway", "any", None),
    ("Northeast Grocery", "Price Chopper", r"^price chopper", "brand", {"NY", "VT", "NH", "MA", "CT", "PA"}),
    ("Northeast Grocery", "Market 32", r"market ?32", "any", None),
    ("Northeast Grocery", "Tops", r"^tops( friendly markets| markets)?$", "brand", {"NY", "PA", "VT"}),
    ("Lowes Foods (Alex Lee)", "Lowes Foods", r"lowes foods", "any", None),
    ("Rouses", "Rouses", r"rouses", "any", None),
    ("Festival Foods", "Festival Foods", r"festival foods", "any", None),
    ("Woodman's", "Woodman's", r"woodman'?s", "any", None),
    ("Market Basket (DeMoulas)", "Market Basket", r"^market basket", "any", {"MA", "NH", "ME"}),
    ("Gelson's", "Gelson's", r"gelson", "any", None),
    ("Natural Grocers", "Natural Grocers", r"natural grocers", "any", None),
    ("Fresh Thyme", "Fresh Thyme", r"fresh thyme", "any", None),
    ("Save A Lot", "Save A Lot", r"save[- ]?a[- ]?lot", "any", None),
    ("UNFI", "Cub", r"^cub( foods)?$", "any", {"MN"}),
    # C&S closed its SpartanNash acquisition 22 Sep 2025.
    ("C&S Wholesale Grocers", "Family Fare", r"family fare", "any", None),
    ("C&S Wholesale Grocers", "D&W Fresh Market", r"d ?& ?w fresh", "any", None),
    ("C&S Wholesale Grocers", "Martin's Super Markets", r"^martin'?s( super ?markets?)?$", "any", {"MI", "IN"}),
    ("C&S Wholesale Grocers", "Grand Union", r"grand union", "any", None),
    ("C&S Wholesale Grocers", "Forest Hills Foods", r"forest hills foods", "any", {"MI"}),
    ("C&S Wholesale Grocers", "Family Fresh Market", r"family fresh market", "any", {"MN", "WI", "ND", "SD", "IA", "NE"}),
    ("Chedraui USA", "Smart & Final", r"smart ?(&|and|n) ?final", "any", None),
    ("Chedraui USA", "El Super", r"^el super\b", "any", None),
    ("Chedraui USA", "Fiesta Mart", r"fiesta mart", "any", None),
    ("The Fresh Market", "The Fresh Market", r"the fresh market", "any", None),
    ("Key Food", "Key Food", r"^key food", "any", None),
    ("IGA", "IGA", r"\biga\b", "any", None),
    ("Piggly Wiggly", "Piggly Wiggly", r"piggly wiggly", "any", None),
    # Regional chains whose brand strings are noisy in Overture.
    ("Market Basket (LA/TX)", "Market Basket", r"^market basket", "any", {"LA", "TX"}),
    ("Price Chopper (KC)", "Price Chopper", r"^price chopper", "any", {"MO", "KS"}),
    ("99 Ranch Market", "99 Ranch Market", r"99 ranch", "any", None),
    ("Food Giant", "Food Giant", r"^food giant", "any", None),
    ("Seafood City", "Seafood City", r"seafood city", "any", None),
    ("Presidente", "Presidente", r"^presidente", "any", {"FL"}),
    ("Ridley's", "Ridley's", r"^ridley'?s", "any", None),
    ("Fine Fare", "Fine Fare", r"^fine fare", "any", None),
    ("C-Town", "C-Town", r"^c-? ?town", "any", None),
    ("Key Food", "Key Food", r"^keyfood", "any", None),
    ("H Mart", "H Mart", r"^h ?mart\b", "any", None),
    ("Shop 'n Save (PA/WV)", "Shop 'n Save", r"^shop ?'?n'? ?save", "any", {"PA", "WV", "OH", "MD"}),
]

RULES = [(p, b, re.compile(rx), mode, regions) for p, b, rx, mode, regions in R]

# Operating footprints (lower-48 states) for chains whose names collide with unrelated
# local stores ("Stop N Shop" in Texas, "Food 4 Less" corner stores, ...). A rule match
# outside the footprint is treated as a name collision, not the chain.
_ALL48 = set("AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
FOOTPRINT = {
    "Ahold Delhaize": set("ME NH VT MA RI CT NY NJ PA DE MD DC VA WV NC SC GA TN KY".split()),
    "Kroger": _ALL48 - set("ME NH VT MA RI CT NY NJ MN ND SD".split()),
    "Albertsons": _ALL48 - set("FL GA SC NC TN KY OH MI IN WI MN IA MO AR AL MS NY WV".split()),
    "Publix": set("FL GA AL SC NC TN VA KY".split()),
    "H-E-B": {"TX"},
    "Wakefern": set("NJ NY PA CT DE MD MA RI".split()),
    "Giant Eagle": set("PA OH WV MD IN".split()),
    "Southeastern Grocers": set("FL GA AL LA MS SC".split()),
    "Hy-Vee": set("IA MO MN NE KS SD WI IL".split()),
    "Meijer": set("MI OH IN IL KY WI".split()),
    "Raley's": {"CA", "NV"},
    "Stater Bros": {"CA"},
    "Ingles": set("NC SC GA TN VA AL".split()),
    "Weis Markets": set("PA MD NJ NY DE VA WV".split()),
    "Wegmans": set("NY PA NJ MA MD VA NC DE CT DC".split()),
}


# Legal-entity names ("Seojung, Inc", "MBM Corporation") are only treated as corporate sites
# when nothing in the name says "shop"; known manufacturers / plants are always dropped.
CORP_SUFFIX = re.compile(r"\b(inc|llc|l\.l\.c|corp|corporation|company|ltd|co|enterprises?|holdings?)\.?$")
SHOPPY = re.compile(r"market|grocer|mart\b|store|supermarket|tienda|mercado|bodega|shop|produce|carniceria|bazaar|deli\b|food center|foods? (outlet|depot|city|giant|king)")
MANUFACTURER = re.compile(
    r"\bdivision\b|processing|\bplant\b|packing|packers|\bdistribution|\bheadquarters|\bcorporate\b|cargill|"
    r"monster energy|frito|pepsico|coca-cola|nestl|kraft|general mills|tyson|hormel|keystone foods|mccain foods|"
    r"cheese of|schreiber|sugar co|water co|food ?service|catering|\bbrokerage|\bimport(s|ers?)\b|\bexport|nursery|orchards")


def _norm(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .str.lower()
        .str.replace("’", "'", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def assign_parents(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `parent`, `banner`, `rule` columns. Vectorised over rules."""
    brand, name, region = _norm(df["brand"]), _norm(df["name"]), df["state"].fillna("")
    parent = pd.Series(pd.NA, index=df.index, dtype="object")
    banner = parent.copy()
    rule_id = pd.Series(-1, index=df.index)
    for i, (p, b, rx, mode, regions) in enumerate(RULES):
        todo = parent.isna()
        if not todo.any():
            break
        hit = brand[todo].str.contains(rx, regex=True)
        if mode == "any":
            hit |= name[todo].str.contains(rx)
        if regions:
            hit &= region[todo].isin(regions)
        idx = hit[hit].index
        parent[idx], banner[idx], rule_id[idx] = p, b, i
    fp = parent.map(FOOTPRINT)
    collision = pd.Series([isinstance(f, set) and r not in f for f, r in zip(fp, region)], index=df.index)
    parent[collision], banner[collision], rule_id[collision] = pd.NA, pd.NA, -1
    # Unmatched records that are clearly companies, plants or warehouses rather than shops
    # ("Cargill", "Kettle Foods Inc", "Equity Group KY Division").
    corp = parent.isna() & (name.str.contains(MANUFACTURER, regex=True)
                            | (name.str.contains(CORP_SUFFIX, regex=True) & ~name.str.contains(SHOPPY, regex=True)))
    parent[corp], banner[corp] = EXCLUDE, "manufacturer / corporate site"
    out = df.copy()
    out["footprint_collision"] = collision
    has_brand = (brand != "") & ~collision
    unmatched = parent.isna()
    # Branded but unmatched -> its own parent; unbranded -> Independent.
    # Overture brand strings sometimes carry a location suffix ("MOM's Organic Market - Waldorf").
    brand_title = (
        df["brand"].fillna("").str.replace("\u2019", "'", regex=False)
        .str.replace(r"\s+-\s+.*$|\s+#?\d+$", "", regex=True).str.strip()
    )
    out["parent"] = parent.where(~unmatched, np.where(has_brand, brand_title, INDEPENDENT))
    out["banner"] = banner.where(~unmatched, np.where(has_brand, brand_title, INDEPENDENT))
    out["rule"] = rule_id
    out["matched"] = ~unmatched
    return out


def dedupe(df: pd.DataFrame, res: int = 9) -> pd.DataFrame:
    """Collapse duplicate records of the same chain within ~one H3 res-9 ring (~350 m).

    Overture often has several records per store (main listing + pickup/pharmacy
    variants, or two source datasets). Independents are left alone.
    """
    df = df.sort_values("confidence", ascending=False).copy()
    df["_c9"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(df["lat"], df["lon"])]
    keep = np.ones(len(df), dtype=bool)
    claimed: dict[str, set] = {}
    for i, (p, c) in enumerate(zip(df["parent"], df["_c9"])):
        if p == INDEPENDENT:
            continue
        s = claimed.setdefault(p, set())
        if c in s:
            keep[i] = False
            continue
        s.update(h3.grid_disk(c, 1))
    return df[keep].drop(columns="_c9").sort_index()
