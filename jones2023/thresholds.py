#!/usr/bin/env python3
"""Jones et al. (2023) LLJ typing, pass two step one: monthly thresholds.

An LLJ gridpoint is one where the nighttime 850-hPa wind speed and the
850-700 hPa shear both exceed the 75th percentile of their monthly frequency
distributions. Those percentiles are climatological, so they need the whole
record: this reads every pass-one night and writes, per calendar month and
per gridpoint, the two threshold fields.

Per gridpoint rather than pooled over the domain, since the paper identifies
LLJ "in every grid point" -- pooling would let the Andes ribbon dominate the
distribution and push the threshold too high everywhere else.

    python thresholds.py --nights $SCRATCH/llj_nights \\
        --out $SCRATCH/llj_thresholds.nc
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import xarray as xr

NIGHT_PATTERN = re.compile(r"^llj_(\d{4})(\d{2})(\d{2})T\d{4}\.nc$")
SEASON_MONTHS = (11, 12, 1, 2, 3)
PERCENTILE = 75.0
FIELDS = ("speed850", "shear")


def nights_by_month(directory: str) -> dict[int, list[str]]:
    """Group pass-one files by the calendar month in their label."""
    groups: dict[int, list[str]] = {}
    for name in sorted(os.listdir(directory)):
        m = NIGHT_PATTERN.match(name)
        if m is None:
            continue
        month = int(m.group(2))
        groups.setdefault(month, []).append(os.path.join(directory, name))
    return dict(sorted(groups.items()))


def stack_month(paths: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load one month's nights into (night, lat, lon) arrays per field.

    ~1300 nights x 171 x 108 x 4 bytes is under 100 MB per field, so a
    straight load is simpler than streaming and fits comfortably.
    """
    stacks = {f: [] for f in FIELDS}
    lat = lon = None
    for path in paths:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            if lat is None:
                lat = ds["latitude"].values
                lon = ds["longitude"].values
            for f in FIELDS:
                # Transpose to (lat, lon); the files store (lon, lat).
                stacks[f].append(ds[f].transpose("latitude", "longitude").values)
    return ({f: np.stack(v, axis=0) for f, v in stacks.items()}, lat, lon)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", required=True, help="pass-one output directory")
    ap.add_argument("--out", required=True, help="threshold NetCDF to write")
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--pooled", action="store_true",
                    help="one threshold per month over all nights AND all "
                         "gridpoints, instead of one per gridpoint. Per "
                         "gridpoint is exceeded ~25%% of nights everywhere by "
                         "construction, which makes the 5-gridpoint minimum "
                         "nearly free; pooled is selective in absolute terms.")
    args = ap.parse_args()

    groups = nights_by_month(args.nights)
    if not groups:
        raise SystemExit(f"no llj_*.nc files in {args.nights}")

    found = sorted(groups)
    print(f"{sum(len(v) for v in groups.values())} nights across months {found}")
    missing = [m for m in SEASON_MONTHS if m not in groups]
    if missing:
        print(f"warning: no nights for month(s) {missing}")

    thr = {f: [] for f in FIELDS}
    months: list[int] = []
    lat = lon = None

    for month, paths in groups.items():
        mark = time.perf_counter()
        stacks, lat, lon = stack_month(paths)
        for f in FIELDS:
            # nanpercentile: pass-one writes NaN as its fill value.
            if args.pooled:
                # One scalar, broadcast back over the grid so the output shape
                # and everything downstream stay identical.
                value = np.nanpercentile(stacks[f], args.percentile)
                thr[f].append(np.full(stacks[f].shape[1:], value))
            else:
                thr[f].append(
                    np.nanpercentile(stacks[f], args.percentile, axis=0))
        months.append(month)
        held = sum(a.nbytes for a in stacks.values()) / 1e6
        print(f"  month {month:2d}: {len(paths):5d} nights, {held:.0f} MB, "
              f"{time.perf_counter() - mark:.1f}s")

    out = xr.Dataset(
        {
            f"thr_{f}": (("month", "latitude", "longitude"),
                         np.stack(thr[f], axis=0).astype("float32"))
            for f in FIELDS
        },
        coords={"month": months, "latitude": lat, "longitude": lon},
    )
    for f in FIELDS:
        out[f"thr_{f}"].attrs = {
            "units": "m s-1",
            "long_name": f"{args.percentile:g}th percentile of nightly {f}",
        }
    out.attrs["percentile"] = args.percentile
    out.attrs["source"] = (
        "per calendar month, all years and all gridpoints pooled"
        if args.pooled else
        "per gridpoint, per calendar month, all years pooled")
    out.attrs["n_nights"] = str({m: len(p) for m, p in groups.items()})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_netcdf(args.out, engine="h5netcdf")
    print(f"\nwrote {args.out}")

    for f in FIELDS:
        a = out[f"thr_{f}"].values
        print(f"  thr_{f}: min {np.nanmin(a):.2f}  "
              f"median {np.nanmedian(a):.2f}  max {np.nanmax(a):.2f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())