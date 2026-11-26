# Who owns America's grocery map?

Overture Maps release **2026-11-18.0** · H3 resolution 4 · each hex is coloured by the winner across the hex plus its 6 neighbours (`grid_disk k=1`) · only hexes that contain a store are drawn (3,408 to 3,440 hexes).

## Run

```bash
pip install -r requirements.txt
cd src
python run.py --skip-buildings   # fast pass (~30 s): typical store sizes, no footprint join
python run_buildings.py          # one-off footprint scan (~5 min, cached)
python run.py                    # full run; --indep-per-sqft 400 sets the flat rate
```

Every expensive step is cached to `cache/*.parquet`: the places pull, the building row-group index, building candidates, store→building matches and the hex winners.

| Module | Role |
|---|---|
| `data.py` | Finds the latest release, pulls the places with DuckDB (httpfs + spatial), adds Census states |
| `parents.py` | Ordered regex rules (brand / name, brand-only for ambiguous names, state limits), operating footprints, exclusions, de-duplication |
| `buildings.py` | Footprint join: row groups pruned by parquet bbox stats, bbox-size prefilter, bbox range join + `ST_Contains`, area in EPSG:5070 (`always_xy`) |
| `analysis.py` | Hex winners with neighbour smoothing; revenue spread over each chain's stores by sqft; $/sqft sanity check |
| `viz.py`, `plots.py` | All figures |
| `data/parent_revenue.csv` | 53 parents with revenue, fiscal year, basis, source URL and notes |

## Headline findings

**Hexes won by each metric.** Parents with their own colour; "Other chains" covers the long tail.

| Parent | Count | Sqft | Revenue | Count→Revenue |
|---|---:|---:|---:|---:|
| Walmart | 572 | 1,897 | 1,673 | **+1,101** |
| Albertsons | 503 | 311 | 379 | −124 |
| Kroger | 232 | 125 | 284 | +52 |
| Ahold Delhaize | 295 | 167 | 149 | −146 |
| H-E-B | 76 | 52 | 124 | +48 |
| Hy-Vee | 67 | 42 | 113 | +46 |
| Publix | 116 | 63 | 93 | −23 |
| Meijer | 3 | 54 | 58 | +55 |
| Aldi | 208 | 5 | 11 | **−197** |
| IGA | 173 | 32 | 31 | −142 |
| Piggly Wiggly | 140 | 11 | 11 | −129 |
| Save A Lot | 58 | 6 | 0 | −58 |
| Other chains | 766 | 448 | 349 | −417 |
| Independent | 92 | 165 | 112 | +20 |

1. **47% of hexes change hands between the count map and the revenue map.** Walmart picks up 1,125 of them.
2. **Walmart wins by store count in only 17% of hexes, but by floor space in 56% and by revenue in 49%.** A Supercenter averages about 150k sqft and roughly $71M a year in grocery sales. One store outweighs several neighbourhood supermarkets.
3. **Square footage overstates Walmart.** Its implied grocery sales are about $470/sqft, against $620–840 for Kroger, Publix and Hy-Vee. Moving from sqft to revenue gives ground back to the high-productivity grocers: Kroger 125→284, H-E-B 52→124, Hy-Vee 42→113, Publix 63→93.
4. **Small-format and rural banners shrink.** Aldi drops from 208 hexes to 11, IGA from 173 to 31, Piggly Wiggly from 140 to 11 and Save A Lot from 58 to 0. They have many stores but small boxes and modest sales.
5. **Five regions hold against Walmart on every metric:** H-E-B in central and south Texas, Publix in peninsular Florida, Hy-Vee in Iowa, Meijer in lower Michigan, and Albertsons/Safeway in the Pacific Northwest and northern California. Kroger keeps Colorado (King Soopers), Ohio/Indiana and the Los Angeles basin (Ralphs).
6. **Surprises:**
   - Ahold Delhaize's Food Lion belt in North Carolina and Virginia wins by count but loses most of it to Walmart on revenue.
   - Albertsons dominates the Mountain West by count, then cedes much of Idaho, Montana and Wyoming.
   - Aldi's few revenue wins (11 hexes) are mostly in northern Minnesota.
7. **How many independents there are.** About 62% of the store records are unbranded independents. Counted as a single bloc they would win 99% of hexes by count. Each independent therefore competes as its own owner, and tan marks places where no chain outweighs a single local store, mostly the rural Great Plains and Mountain West.

**Replication (store name, by count).** Safeway (304 hexes), Food Lion (213), Kroger (188), Publix (154), Albertsons (127), IGA (123), H-E-B (112) and Hy-Vee (111) lead. Walmart barely shows (63) for two reasons:
- Its records are split across "Walmart", "Walmart Supercenter" and "Neighborhood Market".
- Most Supercenters are tagged `department_store`, which the original category filter excludes.

## Audit (`out/audit_parent_banner_counts.csv`)

- **Records:** 100,177 candidate places → 84,768 after exclusions → 82,649 stores after de-duplicating chain records within about 350 m.
- **Rule matches:** 44% of records match a parent rule. The rest are 51.5k independent records and 4.6k branded-but-unmatched records, each of which becomes its own parent. The biggest of those are Foodtown, Lowe's Market, Bravo, Superior Grocers, Food Depot and Vallarta.
- **Exclusions:**

  | Reason | Records |
  |---|---:|
  | Legal-entity, manufacturer and plant listings (e.g. Cargill, Monster Energy tagged `grocery_store`) | 6,147 |
  | In-store departments and pickup duplicates | 4,727 |
  | Convenience stores | 1,687 |
  | Clubs and foodservice | 1,412 |
  | Defunct banners (Fresh & Easy, Bi-Lo, Sweetbay…) | 194 |
  | Pharmacies | 182 |
  | Dollar stores | 104 |

- **Ownership verified by web search, and how the rules changed:**
  - **Winn-Dixie:** Aldi bought Southeastern Grocers in 2024 but sold Winn-Dixie and Harveys to a C&S-led consortium in **Feb 2025**. They now sit under "Southeastern Grocers", not Aldi.
  - **King Kullen:** the Stop & Shop deal was **terminated in 2020**, so King Kullen is not under Ahold.
  - **SpartanNash banners:** C&S closed its acquisition on 22 Sep 2025, so Family Fare, D&W and Martin's go to C&S.
  - **Chedraui USA:** Smart & Final, El Super and Fiesta Mart.
  - **Heritage Grocers:** Cardenas, Tony's and El Rancho.
  - **Reasor's:** part of Brookshire Grocery.
  - **Super One Foods (MN):** owned by Miner's, not Brookshire's "Super 1".
- **Operating footprints:** name collisions outside a chain's footprint (e.g. "Stop N Shop" in Texas) are reassigned, not counted as the chain.

## Square footage

- **Match rate:** 59.7% of stores sit inside an Overture building footprint, and 51.8% are reliable (4,000–320,000 sqft). The other 48% use the chain's median.
- **By chain:** Kroger 88% reliable, Albertsons 91%, Walmart 76%, Aldi 56%, independents 41%.
- **Scan:** 8,107 parquet row groups (158M buildings) contain a store. The `bbox` column is read first and `geometry` only for matching row groups. About 5 minutes.

## Revenue sanity (`out/revenue_sanity_per_sqft.csv`)

- **Inside the $300–900/sqft range:** Walmart $473, Kroger $717, Albertsons $581, Publix $620, Ahold $443, Aldi $399, Hy-Vee $839, Meijer $524.
- **Above the range:**
  - **H-E-B $1,317, Wegmans $1,002:** both are known as very high-volume operators (about $110M per store), so these look real.
  - **Northgate $1,003:** Overture misses about a third of its stores.
- **Below the range:** Southeastern Grocers $217, Save A Lot $106, C&S $277, K-VA-T $281, Lowes Foods $227, Festival $298.
  - These are stale revenue figures (2016–2022) or stale store listings. Overture still shows about 480 Winn-Dixie records, many since converted to Aldi.
  - Licensee formats sit in strip-mall footprints that overstate store size.
- **Store-coverage fix:** Overture lists only 191 of about 635 Trader Joe's and 248 of about 530 Whole Foods. Spreading the full revenue over those stores would make each one absorb the missing stores' sales (about $1,900/sqft). These two chains get a coverage adjustment from a `us_store_count` column in the CSV.
- **Flat-rate stores:** independents and unlisted chains get $400/sqft, a parameter. Floor area is capped at 30k sqft, because larger footprints are usually mixed-use buildings or mis-tagged warehouses.
- **Implied totals:** $1.32T, of which independents are about $317B. That is on the high side of the roughly $1.1T US food-at-home market, so treat independent wins as an upper bound.

## Caveats

- **Places data quality:**
  - Overture places are crowd- and AI-sourced: there are stale listings (closed or converted stores), duplicate records, and brand tags that are sometimes wrong (e.g. "Shoprite South Africa" on US ShopRites).
  - The rules also check the name field, and footprints guard against collisions.
- **Taxonomy change:** the new Overture taxonomy has no `supermarket` category; it is folded into `grocery_store`. The replication uses `taxonomy.primary ILIKE '%grocer%'`. Aldi and Walmart records tagged discount, department or superstore are added only for the extensions (flag `strict=False`). Target is left out.
- **Revenue inputs:** they mix company-reported figures with NRF/Forbes estimates and fiscal years from 2016 to 2026. Only Kroger and Albertsons exclude fuel. Walmart is the grocery line of Walmart U.S. (FY ended Jan 2026, $285.5B). Kroger's announced Giant Eagle deal (expected to close in 2027) is not applied.
- **Footprints:** they measure the whole building, not the sales floor. In-line stores pick up their neighbours' space.
- **Hex winners are a local plurality:** the dominance map shows the middle half of winners hold only 26–42% of local sales (median 33%).

## Outputs (`out/`)

| File | What it shows |
|---|---|
| `01_replication_store_name_count.png` | Faithful recreation: store name, by count |
| `02a_parent_by_count.png`, `02b_parent_by_sqft.png`, `02c_parent_by_revenue.png` | Parent company under each metric, same palette |
| `03_dominance_revenue_share.png` | Revenue winner; opacity is the winner's share of local sales |
| `04_flips_count_to_revenue.png` | Hexes that flip from count to revenue (46.7%) |
| `05_share_of_hexes_by_metric.png` | Bar chart of hexes won per metric |
| `06_1…06_8_empire_*.png` | Walmart, Kroger, Albertsons, Publix, Ahold Delhaize, H-E-B, Aldi, Meijer |
| `audit_parent_banner_counts.csv` | Parent / banner / kind record counts |
| `hexes_won_by_metric.csv`, `share_of_hexes_by_metric.csv` | Winner tallies |
| `sqft_by_parent.csv` | Store count, reliable-match share and median sqft per parent |
| `revenue_sanity_per_sqft.csv` | Implied $/sqft per parent |
| `run_log.txt` | Full log of the last run |
