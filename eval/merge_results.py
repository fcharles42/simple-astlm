#!/usr/bin/env python3
import csv
from pathlib import Path

BASE = Path(__file__).parent
inputs = sorted(BASE.glob("results_*.csv"))
output = BASE / "results_all.csv"

rows = []
fieldnames = None

for path in inputs:
    with open(path) as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Merged {len(inputs)} files → {output} ({len(rows)} rows)")
