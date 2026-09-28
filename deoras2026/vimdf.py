#!/usr/bin/env python3
"""Vertically Integrated Moisture Deficit Flux over overnight windows.

    python vimdf.py --out ./out --start 1990-01-01 --end 1990-02-01

    # one shard of a PBS array job
    python vimdf.py ... --shard $PBS_ARRAY_INDEX --of 32
"""
import argparse
import os
# import sys
import time

import numpy as np
import xarray as xr

from wxread import BBox, Calculation, Era5Local, Era5Rda, Request, plan, shard

G = 9.80665  # m s-2
RD_RV = 0.621981  # Rd/Rv
T_0 = 273.16  # K, triple point
T_ICE = 250.16  # K, below this the IFS uses the ice saturation curve

# 850-300 hPa on ERA5's 37-level pressure grid: 50 hPa spacing down to 750,
# then 25 hPa below that. Not uniform, so don't generate this with range().
LEVELS = (300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850)
P_TOP = 30000.0  # Pa, top of the integration
P_BASE = 85000.0  # Pa, nominal base; the actual base is min(sp, P_BASE)

print("STARTING VIMDF RUN")

def qsat(t: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """Saturation specific humidity, kg/kg.

    Tetens coefficients and the mixed-phase blend follow the IFS formulation
    that produced ERA5, so qs is consistent with the reanalysis' own q. The
    paper does not state which saturation formula it used; this is a choice.
    """
    ew = 611.21 * np.exp(17.502 * (t - T_0) / (t - 32.19))
    ei = 611.21 * np.exp(22.587 * (t - T_0) / (t + 0.7))
    alpha = (((t - T_ICE) / (T_0 - T_ICE)).clip(0.0, 1.0)) ** 2
    es = alpha * ew + (1.0 - alpha) * ei
    return RD_RV * es / (p - (1.0 - RD_RV) * es)


def _column_integral(f: xr.DataArray, p: xr.DataArray, p_bot: xr.DataArray):
    """Trapezoidal integral of f dp from P_TOP down to p_bot.

    Each layer is clipped to the integration bounds and f is linearly
    interpolated to the clipped edges. A column whose surface falls inside
    the layer stack therefore contributes a partial bottom layer, and layers
    entirely below ground get dp = 0 rather than a NaN that would poison the
    whole column.
    """

    def edge(a: xr.DataArray, lower: bool) -> xr.DataArray:
        sl = slice(None, -1) if lower else slice(1, None)
        return a.isel(level=sl).drop_vars("level").rename(level="layer")

    p0, p1 = edge(p, True), edge(p, False)  # p0 < p1, pressure ascending
    f0, f1 = edge(f, True), edge(f, False)

    a = np.minimum(np.maximum(p0, P_TOP), p_bot)
    b = np.minimum(np.maximum(p1, P_TOP), p_bot)
    dp = b - a  # zero for layers outside [P_TOP, p_bot]

    span = p1 - p0
    fa = f0 + (f1 - f0) * (a - p0) / span
    fb = f0 + (f1 - f0) * (b - p0) / span
    return (0.5 * (fa + fb) * dp).sum("layer")


def vimdf(ds: xr.Dataset) -> xr.Dataset:
    """VIMDF per Deoras et al. Eq. (1): (1/g) integral of u (qs - q) dp.

    Returns eastward/northward components in kg m-1 s-1, averaged over the
    window's timesteps. No climatology enters here; anomalies against a daily
    climatology are a reduction over many of these outputs.
    """
    # Integrate with pressure increasing, so dp > 0 and the sign is physical.
    ds = ds.sortby("level")
    p = ds["level"] * 100.0  # hPa -> Pa

    deficit = qsat(ds["t"], p) - ds["q"]

    # Integrate to the surface where terrain intrudes into the layer.
    p_bot = np.maximum(np.minimum(ds["sp"], P_BASE), P_TOP)

    fu = _column_integral(deficit * ds["u"], p, p_bot) / G
    fv = _column_integral(deficit * ds["v"], p, p_bot) / G

    out = xr.Dataset(
        {
            "vimdf_u": fu.mean("time"),
            "vimdf_v": fv.mean("time"),
        }
    )
    out["vimdf_mag"] = np.hypot(out["vimdf_u"], out["vimdf_v"])
    for name in out.data_vars:
        out[name].attrs = {"units": "kg m-1 s-1", "long_name": name}
    out.attrs["levels_hPa"] = f"{min(LEVELS)}-{max(LEVELS)}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default="1990-02-01")
    ap.add_argument("--bbox", nargs=4, type=float, default=[-40, -10, -70, -45],
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--out", default="./vimdf_out")
    ap.add_argument("--source", choices=("rda", "local"), default="rda",
                    help="rda = the shared collection; local = a regional extraction")
    ap.add_argument("--root", default=None,
                    help="override the source's default root path")
    ap.add_argument("--step", default="1D")
    ap.add_argument("--window", default="1D")
    ap.add_argument("--offset", default=None)
    ap.add_argument("--timing", action="store_true",
                    help="report per-phase wall time for each step")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()

    calc = Calculation(
        name="vimdf",
        variables=("u", "v", "q", "t", "sp"),
        step=args.step,
        window=args.window,
        offset=args.offset,
        levels=LEVELS,
    )
    cls = Era5Local if args.source == "local" else Era5Rda
    source = cls(root=args.root) if args.root else cls()

    req = Request(calc, args.start, args.end, BBox(*args.bbox))
    steps = shard(plan(req, source), args.shard, args.of)

    os.makedirs(args.out, exist_ok=True)
    totals = {"graph": 0.0, "compute": 0.0, "write": 0.0}
    done = 0

    for step in steps:
        if step.missing:
            print(f"{step.label}: SKIP, missing {step.missing}")
            continue

        # step.open() only builds the dask graph; the reads happen in compute.
        mark = time.perf_counter()
        ds = step.open()
        totals["graph"] += (graph := time.perf_counter() - mark)

        mark = time.perf_counter()
        result = vimdf(ds).compute()
        totals["compute"] += (compute := time.perf_counter() - mark)

        dest = os.path.join(args.out, step.label.replace("/", "_") + ".nc")
        mark = time.perf_counter()
        result.to_netcdf(dest, engine="h5netcdf")
        totals["write"] += (write := time.perf_counter() - mark)

        ds.close()
        done += 1
        line = f"{step.label}: {dest}  |mag| max {float(result.vimdf_mag.max()):.1f}"
        if args.timing:
            line += (f"  [graph {graph:.2f}s  compute {compute:.2f}s  "
                     f"write {write:.2f}s]")
        print(line)

    if args.timing and done:
        wall = sum(totals.values())
        print(f"\n{done} steps in {wall:.1f}s ({wall / done:.2f}s/step)")
        for phase, secs in totals.items():
            print(f"  {phase:8s} {secs:7.2f}s  {100 * secs / wall:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

print("ENDING VIMDF RUN")
