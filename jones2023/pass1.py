#!/usr/bin/env python3
"""Jones et al. (2023) LLJ typing, pass one: nightly wind fields.

Computes, for each 23-09 UTC night in the extended austral summer, the two
fields the LLJ definition thresholds on:

    speed850  nighttime mean 850-hPa wind speed          (m/s)
    shear     nighttime mean 850-700 hPa shear magnitude (m/s)

Both come from nighttime means of the *components* (U850, V850, U700, V700),
not from means of hourly speeds -- speed is nonlinear in the components, so
the order matters. See the paper's "Low-level jet identification".

The 75th-percentile thresholds are climatological, so nothing can be
classified until every night exists. That is pass two.

    python pass1.py --start 1979-11-01 --end 2022-04-01
    python pass1.py --shard $I --of 16          # one worker of a pool
"""
import argparse
import os
# import sys
import time

import numpy as np
import xarray as xr

from grids import domain_grid
from wxread import BBox, Calculation, Era5Local, Era5Rda, Request, plan, shard

# The paper identifies LLJ at every gridpoint in this domain.
DOMAIN = BBox(lat_min=-30.0, lat_max=12.5, lon_min=-81.75, lon_max=-55.0)

# Nighttime is 23-09 UTC (19:00-05:00 Bolivia local time), one window per day,
# labelled by the day it starts on.
NIGHT_STEP = "1D"
NIGHT_WINDOW = "10h"
NIGHT_OFFSET = "23h"

# Extended austral summer, 1 November - 31 March.
SEASON_MONTHS = (11, 12, 1, 2, 3)

LEVELS = (700, 850)


def night_fields(ds: xr.Dataset, hours=None, target=None) -> xr.Dataset:
    """Nighttime mean components, speed at 850 hPa, and 850-700 hPa shear.

    Components are averaged over the window first, then combined, because
    speed and shear are both nonlinear in u and v.

    `hours` restricts the average to those UTC hours. Passing (0, 6) mimics
    6-hourly model output, which samples only 00Z and 06Z inside a 23-09Z
    window -- the control needed before comparing against MESACLIP, so that
    a sampling penalty is not mistaken for a model bias.

    `target` is a (lat, lon) pair to interpolate onto. Regridding happens
    after the time mean (both linear, so they commute, and this is 4 interps
    per night instead of 40) but BEFORE speed and shear are formed, because
    those are not linear: the speed of a regridded wind is not the regridded
    speed of that wind.
    """
    if hours is not None:
        ds = ds.sel(time=ds["time"].dt.hour.isin(list(hours)))
        if ds.sizes.get("time", 0) == 0:
            raise ValueError(f"no timesteps at hours {tuple(hours)} in this window")
    n_samples = int(ds.sizes["time"])

    mean = ds.mean("time", keep_attrs=False)

    grid_note = "native"
    if target is not None:
        tlat, tlon = target
        # interp wants ascending coordinates; ERA5 latitude descends. Sort in,
        # interpolate, sort back, so the output keeps the N..S convention every
        # downstream reader already assumes.
        mean = (mean.sortby("latitude")
                    .interp(latitude=np.sort(np.asarray(tlat)),
                            longitude=np.asarray(tlon),
                            method="linear")
                    .sortby("latitude", ascending=False))
        grid_note = f"regridded to {len(tlat)} lat x {len(tlon)} lon"

    # drop_vars here, not after assembly: each component carries its own
    # scalar `level` coord (850 or 700), and putting them in one Dataset
    # would make xarray try to reconcile 850 against 700.
    u850 = mean["u"].sel(level=850).drop_vars("level")
    v850 = mean["v"].sel(level=850).drop_vars("level")
    u700 = mean["u"].sel(level=700).drop_vars("level")
    v700 = mean["v"].sel(level=700).drop_vars("level")

    speed850 = np.hypot(u850, v850)

    # ASSUMPTION: shear is the magnitude of the vector difference between the
    # two levels. The paper says "vertical wind shear (700-850 hPa)" without
    # defining it. The difference of magnitudes would be negative in a jet
    # (850 faster than 700), so a 75th-percentile threshold on it would select
    # the weakest shear -- which cannot be the intent.
    shear = np.hypot(u700 - u850, v700 - v850)

    out = xr.Dataset({
        "speed850": speed850,
        "shear": shear,
        # Components kept so plots can draw the wind vector, and so the shear
        # definition can be revisited without re-reading the collection.
        "u850": u850, "v850": v850, "u700": u700, "v700": v700,
    })
    out = out.drop_vars("level", errors="ignore")

    out["speed850"].attrs = {
        "units": "m s-1",
        "long_name": "nighttime mean 850 hPa wind speed",
    }
    out["shear"].attrs = {
        "units": "m s-1",
        "long_name": "nighttime mean 850-700 hPa wind shear magnitude",
    }
    for name, lev in (("u850", 850), ("v850", 850), ("u700", 700), ("v700", 700)):
        out[name].attrs = {
            "units": "m s-1",
            "long_name": f"nighttime mean {'eastward' if name[0] == 'u' else 'northward'}"
                         f" wind at {lev} hPa",
        }

    out.attrs["window_utc"] = "23-09"
    out.attrs["method"] = "components averaged, then regridded, then combined"
    out.attrs["n_samples"] = n_samples
    out.attrs["sample_hours"] = "all" if hours is None else ",".join(
        str(h) for h in hours)
    out.attrs["grid"] = grid_note
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1979-11-01")
    ap.add_argument("--end", default="2022-04-01", help="exclusive")
    ap.add_argument("--source", choices=("rda", "local"), default="rda")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="./llj_nights")
    ap.add_argument("--all-months", action="store_true",
                    help="ignore the Nov-Mar season filter")
    ap.add_argument("--regrid", choices=("none", "n80"), default="none",
                    help="n80 interpolates onto ERA-Interim's 1.125 deg "
                         "Gaussian grid, the one the paper's spatial criteria "
                         "were tuned on")
    ap.add_argument("--sample-hours", default=None,
                    help="restrict the nighttime mean to these UTC hours, "
                         "e.g. 0,6 to mimic 6-hourly model output")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    cls = Era5Local if args.source == "local" else Era5Rda
    source = cls(root=args.root) if args.root else cls()

    calc = Calculation(
        name="llj",
        variables=("u", "v"),
        step=NIGHT_STEP,
        window=NIGHT_WINDOW,
        offset=NIGHT_OFFSET,
        levels=LEVELS,
    )
    req = Request(
        calc, args.start, args.end, DOMAIN,
        months=None if args.all_months else SEASON_MONTHS,
    )
    steps = shard(plan(req, source), args.shard, args.of)

    hours = None
    if args.sample_hours:
        hours = tuple(int(h) for h in args.sample_hours.split(","))

    target = domain_grid() if args.regrid == "n80" else None

    os.makedirs(args.out, exist_ok=True)
    print(f"{len(steps)} nights, shard {args.shard} of {args.of} -> {args.out}")
    print(f"sample hours: {'all' if hours is None else hours}")
    if target is None:
        print("grid: native")
    else:
        print(f"grid: N80, {len(target[0])} lat x {len(target[1])} lon "
              f"= {len(target[0]) * len(target[1])} gridpoints")

    totals = {"graph": 0.0, "compute": 0.0, "write": 0.0}
    done = skipped = 0

    for step in steps:
        dest = os.path.join(args.out, step.label.replace("/", "_") + ".nc")
        if os.path.exists(dest):
            skipped += 1
            continue
        if step.missing:
            print(f"{step.label}: SKIP, missing {step.missing}")
            continue

        mark = time.perf_counter()
        ds = step.open()
        totals["graph"] += (graph := time.perf_counter() - mark)

        mark = time.perf_counter()
        result = night_fields(ds, hours=hours, target=target).compute()
        totals["compute"] += (compute := time.perf_counter() - mark)

        mark = time.perf_counter()
        tmp = dest + ".tmp"
        result.to_netcdf(tmp, engine="h5netcdf")
        os.replace(tmp, dest)
        totals["write"] += (write := time.perf_counter() - mark)

        ds.close()
        done += 1
        line = (f"{step.label}: n={result.attrs['n_samples']} "
                f"max speed {float(result.speed850.max()):.1f} "
                f"shear {float(result.shear.max()):.1f} m/s")
        if args.timing:
            line += f"  [graph {graph:.2f}s compute {compute:.2f}s write {write:.2f}s]"
        print(line)

    print(f"\n{done} written, {skipped} already present")
    if args.timing and done:
        wall = sum(totals.values())
        print(f"{wall:.1f}s total, {wall / done:.2f}s/night")
        for phase, secs in totals.items():
            print(f"  {phase:8s} {secs:7.2f}s  {100 * secs / wall:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
