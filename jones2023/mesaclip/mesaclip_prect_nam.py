# %% [markdown]
# # MESACLIP PRECT over North America
#
# Assumes `regrid_mesaclip_prect.sh` has already run and produced regridded
# PRECC/PRECL files on the CAM-FV 0.23x0.31 grid.
#
# Everything here is cheap -- the expensive part (the remap) already happened
# in the shell script. Nothing in this file touches the native ne120 grid.

# %%
import glob

import cf_xarray  # noqa: F401  -- registers the .cf accessor
import numpy as np
import xarray as xr

REGRID_DIR = "/glade/campaign/univ/udrt0019/regridded"
OUT_DIR = "/glade/campaign/univ/udrt0019"

CASE = "b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.010"

# North America box (0-360 longitude, matching the CAM convention)
LAT_SLICE = slice(0, 50)
LON_SLICE = slice(225, 305)

# %% [markdown]
# ## Load

# %%
def load_var(case, var):
    """Open all decades of one regridded variable as a single dataset."""
    pattern = f"{REGRID_DIR}/{case}.cam.h0.{var}.*.fv0.23x0.31.nc"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")
    print(f"{var}: {len(files)} files")
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        use_cftime=True,
        chunks={"time": 12},
    )


dsc = load_var(CASE, "PRECC")
dsl = load_var(CASE, "PRECL")

# %% [markdown]
# ## Recenter the time axis
#
# CAM stamps monthly means at the *end* of the averaging interval, so a
# January mean carries a 1 February timestamp. Left uncorrected this shifts
# every seasonal composite by one month. The fix is to use the midpoint of
# `time_bnds`.
#
# If `time_bnds` did not survive the remap, fall back to reading it from the
# original native-grid file instead -- it is a function of time only, so the
# regridding is irrelevant to it.

# %%
def recenter_time(ds):
    attrs = ds.time.attrs.copy()
    encoding = ds.time.encoding.copy()
    bnds = ds.cf.get_bounds("time")
    bnds_dim = ds.cf.get_bounds_dim_name("time")
    ds = ds.assign_coords(time=bnds.mean(bnds_dim))
    ds.time.attrs = attrs
    ds.time.encoding = encoding
    return ds


dsc = recenter_time(dsc)
dsl = recenter_time(dsl)

# Confirm the shift did what you expect: the first timestamp should now sit
# mid-January 2006, not 1 February 2006.
print("first three timestamps:", dsc.time.values[:3])

# %% [markdown]
# ## Build PRECT
#
# `PRECC` (convective) and `PRECL` (large-scale) are both in m/s.
# `86400 * 1000` converts to mm/day.
#
# Remapping is linear, so summing after the remap is identical to summing
# before it.

# %%
np.testing.assert_array_equal(dsc.time.values, dsl.time.values)

prect = (dsc["PRECC"] + dsl["PRECL"]) * 86400.0 * 1000.0
prect.name = "PRECT"
prect.attrs = {
    "long_name": "Total precipitation rate (convective + large-scale)",
    "units": "mm/day",
    "regrid_method": "ESMF bilinear, ne120np4 -> CAM-FV 0.23x0.31",
    "regrid_note": "bilinear is not area-conservative; do not use for closed water budgets",
}

# %% [markdown]
# ## Subset to North America

# %%
pr_nam = prect.sel(lat=LAT_SLICE, lon=LON_SLICE).load()
print(pr_nam.sizes)
pr_nam

# %% [markdown]
# ## Sanity checks
#
# Precipitation must be non-negative and physically plausible. Bilinear
# interpolation can in principle undershoot, though with weights this close
# to a simple triangular average it rarely matters in practice.

# %%
print(f"min  {float(pr_nam.min()):.4f} mm/day")
print(f"max  {float(pr_nam.max()):.4f} mm/day")
print(f"NaNs {int(pr_nam.isnull().sum())}")

neg = int((pr_nam < 0).sum())
print(f"negative values: {neg}")
if neg:
    print("  -> bilinear undershoot; clip with pr_nam.clip(min=0) if needed")

# Area-weighted regional mean. The map file's area_b is all zeros (ESMF does
# not populate it for bilinear maps), so compute cos(lat) weights directly.
# Safe here because the 0-50N subset excludes the FV polar cap cells.
weights = np.cos(np.deg2rad(pr_nam.lat))
nam_mean = pr_nam.weighted(weights).mean(("lat", "lon"))
print(f"\ndomain-mean PRECT: {float(nam_mean.mean()):.3f} mm/day")
print("annual cycle (mm/day):")
print(nam_mean.groupby("time.month").mean().values.round(2))

# %% [markdown]
# ## Save

# %%
outfile = f"{OUT_DIR}/{CASE}.cam.h0.PRECT.NAM.fv0.23x0.31.nc"

pr_nam.to_netcdf(
    outfile,
    encoding={"PRECT": {"zlib": True, "complevel": 4, "dtype": "float32"}},
)
print(f"wrote {outfile}")

# %% [markdown]
# ## Quick look

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    figsize=(9, 6), subplot_kw={"projection": ccrs.PlateCarree()}
)
pr_nam.mean("time").plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="YlGnBu",
    cbar_kwargs={"label": "PRECT (mm/day)"},
)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
ax.set_title(f"Mean total precipitation\n{CASE}")
plt.tight_layout()
plt.show()
