#!/usr/bin/env python3
"""Jones et al. (2023) LLJ typing, pass two: label every night.

Runs the classification over the whole pass-one record, writes one row per
night, and reports the type frequencies next to the paper's published ones.

Those frequencies are the check worth having: they are exact, and they do
not depend on any climatology beyond the thresholds. If the subregion
polygons are wrong, the frequencies move.

    python pass2.py --nights $SCRATCH/llj_nights \\
        --thresholds $SCRATCH/llj_thresholds.nc \\
        --regions regions.yaml \\
        --out $SCRATCH/llj_types.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter

import numpy as np
import xarray as xr

from classify import (NO_LLJ, UNCLASSIFIED, classify_night, load_regions,
                          region_masks)

NIGHT_PATTERN = re.compile(r"^llj_(\d{4})(\d{2})(\d{2})T(\d{4})\.nc$")

# Counts from Supplementary Table 1, "LLJ Total" row, over the same 6504
# nights this pipeline produces. The main text's "(2070 total cases)" is the
# Andes count in the wrong slot -- the total is 5133. Supp Table 1, Table 1's
# stated percentages, and the stationary distribution of Table 1's transition
# matrix (1847/1200/50/2091 nights) all agree to within 1%, so these are the
# numbers to compare against.
PAPER_COUNTS = {"Central": 1827, "Northern": 1187, "Peru": 49, "Andes": 2070}
PAPER_TOTAL = sum(PAPER_COUNTS.values())        # 5133 LLJ nights
PAPER_NO_LLJ = 6504 - PAPER_TOTAL               # 1371
PAPER_ANDES_X_PCT = 5.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--regions", default="regions.yaml")
    ap.add_argument("--out", required=True, help="CSV of one row per night")
    args = ap.parse_args()

    regions = load_regions(args.regions)
    thr = xr.open_dataset(args.thresholds, engine="h5netcdf")

    files = sorted(f for f in os.listdir(args.nights) if NIGHT_PATTERN.match(f))
    if not files:
        raise SystemExit(f"no llj_*.nc in {args.nights}")
    print(f"{len(files)} nights, thresholds for months {list(thr['month'].values)}")

    masks = None
    rows = []
    counts: Counter = Counter()
    unknown_sigs: Counter = Counter()
    started = time.perf_counter()

    for k, name in enumerate(files, 1):
        m = NIGHT_PATTERN.match(name)
        year, month, day = (int(m.group(i)) for i in (1, 2, 3))

        with xr.open_dataset(os.path.join(args.nights, name),
                             engine="h5netcdf") as ds:
            if masks is None:
                masks = region_masks(regions, ds["latitude"].values,
                                     ds["longitude"].values)
                print("region gridpoints: "
                      + ", ".join(f"{n}={int(v.sum())}"
                                  for n, v in sorted(masks.items())))
            speed = ds["speed850"].values
            shear = ds["shear"].values

        t = thr.sel(month=month)
        result = classify_night(speed, shear, t["thr_speed850"].values,
                                t["thr_shear"].values, masks)

        counts[result.label] += 1
        if result.label == UNCLASSIFIED:
            for c in result.crws:
                unknown_sigs["".join(str(x) for x in sorted(c.signature))] += 1

        rows.append({
            "date": f"{year:04d}-{month:02d}-{day:02d}",
            "type": result.label,
            "andes_x": int(result.is_andes_x),
            "n_crws": len(result.crws),
            "signatures": "|".join(
                "".join(str(x) for x in sorted(c.signature)) for c in result.crws
            ),
            "core_lat_idx": "" if result.core is None else result.core[0],
            "core_lon_idx": "" if result.core is None else result.core[1],
        })

        if k % 500 == 0:
            rate = k / (time.perf_counter() - started)
            print(f"  {k}/{len(files)}  {rate:.0f} nights/s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({time.perf_counter() - started:.0f}s)")

    # ---- frequencies against the paper ----
    cases = sum(v for k, v in counts.items() if k not in (NO_LLJ, UNCLASSIFIED))
    print(f"\n{len(files)} nights: {counts[NO_LLJ]} no-LLJ, {cases} LLJ, "
          f"{counts[UNCLASSIFIED]} unclassified")
    print(f"  paper (Supp Table 1): {PAPER_NO_LLJ} no-LLJ, {PAPER_TOTAL} LLJ")

    print(f"\n{'type':<10} {'count':>7} {'paper':>7} {'err':>8}   "
          f"{'% LLJ':>7} {'paper %':>8}")
    errs = []
    for name, want in PAPER_COUNTS.items():
        n = counts.get(name, 0)
        rel = (n - want) / want
        errs.append(abs(rel))
        pct = 100.0 * n / cases if cases else float("nan")
        print(f"{name:<10} {n:>7} {want:>7} {100 * rel:>+7.1f}%   "
              f"{pct:>6.1f}% {100 * want / PAPER_TOTAL:>7.1f}%")
    print(f"\n  total LLJ {cases} vs {PAPER_TOTAL} "
          f"({100 * (cases - PAPER_TOTAL) / PAPER_TOTAL:+.1f}%)")
    print(f"  mean |error| across types: {100 * sum(errs) / len(errs):.1f}%")

    andes = counts.get("Andes", 0)
    andes_x = sum(r["andes_x"] for r in rows)
    if andes:
        pct = 100.0 * andes_x / andes
        print(f"\nAndes-X: {andes_x} = {pct:.1f}% of Andes types "
              f"(paper {PAPER_ANDES_X_PCT}%)")

    if unknown_sigs:
        print("\nunclassified overlap signatures (not in the paper's lists):")
        for sig, n in unknown_sigs.most_common(10):
            print(f"  {sig or '(none)'}: {n}")
        print("A large count here points at the subregion polygons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
