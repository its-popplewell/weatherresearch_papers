#!/usr/bin/env python3
"""Jones et al. (2023) LLJ typing: region inclusion and night labelling.

Two jobs:

  1. INCLUSION -- turn the five digitized subregion polygons into boolean
     masks on the analysis grid, once, so every night reuses them.

  2. LABELLING -- for one night, find the contiguous regions of high wind
     speed (CRWS), keep the ones meeting the paper's size criteria, work out
     which subregions each overlaps, and map that signature to a type.

Thresholds are inputs here: an LLJ gridpoint is one where nighttime 850-hPa
speed and 850-700 shear both exceed the 75th percentile of their monthly
distributions. Computing those percentiles over the whole record is a
separate step.

    python classify.py --regions _regions.yaml \\
        --night $SCRATCH/llj_nights/llj_19900115T2300.nc \\
        --thresholds $SCRATCH/llj_thresholds.nc
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import yaml
from scipy import ndimage

# Paper's criteria, "Low-level jet identification".
CRWS_SPEED = 10.0       # m/s, contour defining a contiguous region of wind speed
MIN_CRWS_POINTS = 5
MIN_LLJ_POINTS = 5
# MIN_CRWS_POINTS = 43    # minimum size of the CRWS itself
# MIN_LLJ_POINTS = 43     # minimum LLJ gridpoints inside the CRWS

# Overlap signatures for a single CRWS, from "Low-level jet form".
CENTRAL_SIGS = {frozenset({1}), frozenset({2}), frozenset({1, 2}),
                frozenset({2, 3}), frozenset({1, 2, 3})}
NORTHERN_SIGS = {frozenset({4}), frozenset({5}), frozenset({4, 5}),
                 frozenset({3, 4}), frozenset({3, 4, 5})}
ANDES_SIGS = {frozenset({1, 2, 3, 4, 5}), frozenset({1, 2, 3, 4}),
              frozenset({2, 3, 4}), frozenset({2, 3, 4, 5})}
PERU_SIGS = {frozenset({3})}

# A single CRWS spanning the eastern Andes is the Andes-X subtype.
ANDES_X_SIGS = ANDES_SIGS

NO_LLJ = "no-LLJ"
UNCLASSIFIED = "unclassified"


# --------------------------------------------------------------------------
# 1. Inclusion
# --------------------------------------------------------------------------

def load_regions(path: str) -> dict[int, np.ndarray]:
    """Read the polygon YAML into {region number: (n, 2) array of lon/lat}."""
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    out = {}
    for key, entry in doc["regions"].items():
        verts = np.asarray(entry["vertices"], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(f"region {key}: want (n, 2) vertices, got {verts.shape}")
        out[int(key)] = verts
    return dict(sorted(out.items()))


def points_in_polygon(px: np.ndarray, py: np.ndarray,
                      poly: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon, vectorized over flat point arrays.

    Counts how many polygon edges a ray cast in +x from each point crosses;
    odd means inside. Written out rather than taken from matplotlib or
    shapely so this module needs no extra dependency.
    """
    inside = np.zeros(px.shape, dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)   # closes the ring

    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:                            # horizontal edge, no crossing
            continue
        # Does the point's latitude fall within this edge's span?
        straddles = (py >= np.minimum(ay, by)) & (py < np.maximum(ay, by))
        if not straddles.any():
            continue
        # x of the edge at the point's latitude
        xcross = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (px < xcross)
    return inside


def region_masks(regions: dict[int, np.ndarray],
                 lat: np.ndarray, lon: np.ndarray) -> dict[int, np.ndarray]:
    """Boolean (lat, lon) mask per region. Longitudes are matched on the
    -180..180 convention the polygons use, so a 0..360 axis is converted."""
    lon = np.asarray(lon, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    glon, glat = np.meshgrid(lon, np.asarray(lat, dtype=float))
    flat_x, flat_y = glon.ravel(), glat.ravel()

    masks = {}
    for num, poly in regions.items():
        masks[num] = points_in_polygon(flat_x, flat_y, poly).reshape(glat.shape)
    return masks


# --------------------------------------------------------------------------
# 2. Labelling
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Crws:
    """One contiguous region of wind speed that met both size criteria."""

    index: int
    signature: frozenset
    n_points: int
    n_llj: int
    category: str          # central / northern / peru / andes / unclassified
    is_andes_x: bool


@dataclass(frozen=True)
class NightType:
    """The classification of one night."""

    label: str             # no-LLJ / Central / Northern / Peru / Andes
    is_andes_x: bool
    crws: tuple            # tuple[Crws, ...], in label order
    core: tuple | None     # (lat index, lon index) of max speed in the LLJ


def _single_category(signature: frozenset) -> str:
    if signature in ANDES_SIGS:
        return "andes"
    if signature in CENTRAL_SIGS:
        return "central"
    if signature in NORTHERN_SIGS:
        return "northern"
    if signature in PERU_SIGS:
        return "peru"
    return UNCLASSIFIED


def _combine(categories: set[str]) -> str:
    """Multiple-CRWS rules: Central alone (or with Peru), Northern alone (or
    with Peru), Andes when both branches are present or any CRWS is itself
    an Andes signature."""
    if "andes" in categories:
        return "Andes"
    if "central" in categories and "northern" in categories:
        return "Andes"
    if categories <= {"central", "peru"} and "central" in categories:
        return "Central"
    if categories <= {"northern", "peru"} and "northern" in categories:
        return "Northern"
    if categories == {"peru"}:
        return "Peru"
    return UNCLASSIFIED


def classify_night(speed850: np.ndarray, shear: np.ndarray,
                   thr_speed: np.ndarray, thr_shear: np.ndarray,
                   masks: dict[int, np.ndarray]) -> NightType:
    """Label one night. Arrays are (lat, lon) and share a grid.

    Order follows Supplementary Fig. 1: identify LLJ at every gridpoint, mask
    out everything outside the five subregions, and only THEN find CRWS. The
    mask is applied before CRWS detection, not as an overlap test afterward,
    so a CRWS cannot reach a subregion from outside the corridor -- which
    otherwise lets, for instance, the Caribbean easterlies deliver a Northern
    night by touching region 5 from the east.
    """
    in_any_region = np.logical_or.reduce(list(masks.values()))

    # An LLJ gridpoint exceeds both monthly 75th percentiles, inside the mask.
    is_llj = (speed850 > thr_speed) & (shear > thr_shear) & in_any_region

    # Contiguous regions of wind speed above the fixed 10 m/s contour, also
    # confined to the mask. 8-connectivity: diagonal neighbours join, which
    # matters on a coarse grid where a feature spanning a few cells
    # diagonally would otherwise fragment below the size minimum.
    labelled, n_found = ndimage.label(
        (speed850 > CRWS_SPEED) & in_any_region,
        structure=np.ones((3, 3), dtype=int),
    )
    if n_found == 0:
        return NightType(NO_LLJ, False, (), None)

    kept: list[Crws] = []
    for idx in range(1, n_found + 1):
        blob = labelled == idx

        # No overlap test needed now -- every blob is inside the mask by
        # construction.
        n_points = int(blob.sum())
        if n_points < MIN_CRWS_POINTS:
            continue
        n_llj = int((blob & is_llj).sum())
        if n_llj < MIN_LLJ_POINTS:
            continue

        signature = frozenset(
            num for num, mask in masks.items() if (blob & mask).any()
        )
        kept.append(Crws(
            index=idx,
            signature=signature,
            n_points=n_points,
            n_llj=n_llj,
            category=_single_category(signature),
            is_andes_x=signature in ANDES_X_SIGS,
        ))

    if not kept:
        return NightType(NO_LLJ, False, (), None)

    label = _combine({c.category for c in kept})
    andes_x = label == "Andes" and any(c.is_andes_x for c in kept)

    # Core: the strongest 850-hPa wind among LLJ points in the kept CRWS.
    keep_mask = np.zeros_like(is_llj)
    for c in kept:
        keep_mask |= labelled == c.index
    core_area = keep_mask & is_llj
    core = None
    if core_area.any():
        masked = np.where(core_area, speed850, -np.inf)
        core = tuple(int(i) for i in np.unravel_index(masked.argmax(), masked.shape))

    return NightType(label, andes_x, tuple(kept), core)


# --------------------------------------------------------------------------
# CLI: inclusion report, plus one night if given
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="regions.yaml")
    ap.add_argument("--night", default=None, help="one pass-one NetCDF")
    ap.add_argument("--thresholds", default=None,
                    help="NetCDF with thr_speed850 / thr_shear by month")
    args = ap.parse_args()

    regions = load_regions(args.regions)
    print(f"{len(regions)} regions from {args.regions}")
    for num, poly in regions.items():
        print(f"  {num}: {len(poly)} vertices  "
              f"lon {poly[:, 0].min():.2f}..{poly[:, 0].max():.2f}  "
              f"lat {poly[:, 1].min():.2f}..{poly[:, 1].max():.2f}")

    if args.night is None:
        print("\nno --night given; pass one to build masks on its grid")
        return 0

    import xarray as xr
    ds = xr.open_dataset(args.night, engine="h5netcdf")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    masks = region_masks(regions, lat, lon)

    print(f"\ninclusion on the {len(lat)} x {len(lon)} grid:")
    total = 0
    for num, mask in masks.items():
        total += int(mask.sum())
        print(f"  region {num}: {int(mask.sum()):5d} gridpoints")
    overlap = sum(m.astype(int) for m in masks.values())
    print(f"  union: {int((overlap > 0).sum())} gridpoints "
          f"(sum {total}, so {int((overlap > 1).sum())} claimed twice)")

    if args.thresholds is None:
        print("\nno --thresholds given; cannot label without the percentiles")
        return 0

    m = re.match(r"^llj_(\d{4})(\d{2})(\d{2})T\d{4}\.nc$", os.path.basename(args.night))
    if m is None:
        raise SystemExit(f"cannot read a date from {os.path.basename(args.night)}")
    month = int(m.group(2))
    thr = xr.open_dataset(args.thresholds, engine="h5netcdf").sel(month=month)
    result = classify_night(
        ds["speed850"].values, ds["shear"].values,
        thr["thr_speed850"].values, thr["thr_shear"].values, masks,
    )
    print(f"\n{os.path.basename(args.night)}: {result.label}"
          + ("  (Andes-X)" if result.is_andes_x else ""))
    for c in result.crws:
        print(f"  CRWS {c.index}: regions {sorted(c.signature)} "
              f"{c.n_points} pts, {c.n_llj} LLJ -> {c.category}")
    if result.core is not None:
        i, j = result.core
        print(f"  core at {lat[i]:.2f}N {lon[j]:.2f}E, "
              f"{ds['speed850'].values[i, j]:.1f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
