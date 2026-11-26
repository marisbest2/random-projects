"""Standalone entry point for the slow building-footprint scan (all candidate places)."""
import data, buildings
rel = (data.CACHE / "release.txt").read_text().strip()
pl = data.places(rel)
m = buildings.match_stores(rel, pl)
print("matched:", m.bid.notna().mean().round(3), "of", len(m))
