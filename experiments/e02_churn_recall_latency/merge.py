"""Merge per-dataset shards into output.json."""
import json, glob, os

here = os.path.dirname(os.path.abspath(__file__))
shards = sorted(glob.glob(os.path.join(here, "output_*.json")))
shards = [s for s in shards if "legacy" not in s and "/output.json" not in s]
print(f"merging {len(shards)} shards:")
all_rows = []
for s in shards:
    with open(s) as f:
        rows = json.load(f)
    print(f"  {os.path.basename(s)}: {len(rows)} rows")
    all_rows.extend(rows)
out = os.path.join(here, "output.json")
with open(out, "w") as f:
    json.dump(all_rows, f, indent=2)
print(f"\n{len(all_rows)} total rows -> {out}")
