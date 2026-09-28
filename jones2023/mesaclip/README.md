# MESACLIP / iHESP CESM-HR regridding

Pipeline for getting MESACLIP atmospheric output off the native ne120np4
spectral-element grid and onto a regular lat-lon grid for analysis.

PRECT (total precipitation) is worked through end-to-end as the example.
The pipeline is variable-agnostic — see "Adapting to other variables" below
for what to change.

## Why this exists

MESACLIP CAM output is on the ne120np4 spectral-element grid: unstructured,
with a single `ncol = 777602` dimension instead of `(lat, lon)`. Most
analysis tooling (cartopy plotting, area-averaging, comparison against
reanalysis) wants a rectilinear grid. So the data has to be remapped first.

The wrong way to do this is nearest-neighbour interpolation, which is what
you get if you hand the lon/lat arrays to xESMF as a point cloud
(`locstream_in=True`). That discards the grid geometry, so nearest-neighbour
becomes the only available method, and the result duplicates some source
points while dropping others. Use a proper ESMF map file instead.

## Files

| File | What it does |
|---|---|
| `regrid_mesaclip_prect.sh` | Applies the ESMF map to native-grid files. The expensive step. Run once per variable. |
| `mesaclip_prect_nam.py` | Post-processing: time recentering, unit conversion, regional subset, sanity checks. Cheap. `# %%` cell markers, so it opens as a notebook or runs as a script. |

## Paths

| | |
|---|---|

| Map file | `/glade/work/avclarke/MESACLIP_regrid/map_ne120_to_0.23x0.31_bilinear.nc` |
| Source data (historical 1920–2005) | `/glade/campaign/collections/gdex/data/d651007/` |
| Source data (RCP6.0 2006–2100) | `/glade/campaign/collections/gdex/data/d651008/` |
| Regridded output | `/glade/campaign/univ/udrt0019/regridded/` |
| Final subsets | `/glade/campaign/univ/udrt0019/` |
| Project page | https://project.cgd.ucar.edu/projects/MESACLIP/ |

## The map file

Originally from Nan Rosenbloom's MESACLIP mapfiles directory. Prefer it over
generating your own — matching the project's own diagnostics matters more
than a marginal accuracy gain.

Verified properties:

- `src_grid_dims = 777602` — the ne120np4 GLL node count. **Any map you use
  must have this**, or it was built for a different grid.
- `dst_grid_dims = 1152, 768` — the CAM finite-volume 0.23°×0.31° grid
  (0.3125° lon × 0.23468° lat). 
- `map_method = "Bilinear remapping"`, `normalization = "destarea"`
- Row sums are exactly 1.0 everywhere; no masked or under-weighted cells.
- Latitude runs −90 to +90 inclusive (768 points), so the first and last rows
  are FV polar cap cells, not full-width cells.
- `area_a` / `area_b` are all zeros — ESMF does not populate these for
  bilinear maps. Doesn't matter, because `ncremap` computes real destination
  areas and writes them to the output `area` variable. Use that.

**Bilinear is not area-conservative.** Fine for means, spatial patterns,
trends, and variability. Not valid for a closed budget calculation — the
area-integral of a flux will not match the native grid. If you need
conservation, build a separate map:

```bash
GenerateCSMesh --res 120 --file ne120.g
ncremap -a se2fv_flx -s ne120.g -g dst_SCRIP.nc -m map_conservative.nc
```

Use `se2fv_flx` (monotone, flux-appropriate) for fluxes, `se2fv_stt` for
state variables.

## Running it

```bash
# Casper interactive; this is I/O bound, not CPU bound
execcasper -A <project> -l select=1:ncpus=4:mem=40GB -l walltime=2:00:00
bash regrid_mesaclip_prect.sh
```

Edit at the top of the script: `MEMBER`, `OUTDIR`, the variable list, and
which scenario block is uncommented. It skips files that already exist, so
it's safe to re-run after an interruption.

Then run the notebook. Two diagnostics to actually read:

- **First three timestamps** should be mid-January, mid-February,
  mid-March. If they're the 1st of the following month, the time recentering
  didn't happen — see below.
- **Annual cycle** should be physically sensible for your variable and
  region.

## Two things that will bite you

**CAM stamps monthly means at the end of the averaging interval.** A January
mean carries a 1 February timestamp. Uncorrected, every seasonal composite is
shifted by a month. The `recenter_time()` function fixes this using the
`time_bnds` midpoint. This applies to all `h0` monthly output, every
variable — don't remove it.

**`cf_xarray` was removed as a dependency.** `time_bnds` and its `nbnd`
dimension are hardcoded, since every file in this workflow comes from the
same `ncremap` call. If you point this at files from elsewhere, check those
names.

## Adapting to other variables

Most of the pipeline doesn't care what variable you're remapping. What
changes:

**In the shell script** — the variable list:

```bash
for VAR in PRECC PRECL; do    # -> TREFHT, PSL, U, Z3, ...
```

Anything dimensioned `(time, ncol)` works unchanged. 3D fields dimensioned
`(time, lev, ncol)` also remap fine — `ncremap` treats `lev` as a passthrough
dimension — but note the output is on CAM's hybrid sigma-pressure levels, so
you'll need `PS`, `hyam`, `hybm`, `P0` alongside it for any pressure-level
interpolation.

**In the notebook:**

- *Unit conversion.* PRECT-specific: PRECC and PRECL are both m/s, and
  `* 86400 * 1000` converts to mm/day. Most variables need nothing
  (TREFHT in K, PSL in Pa). Check the `units` attribute.
- *Combining variables.* PRECT = PRECC + PRECL is unusual. Most variables are
  a single field, so `load_var()` once and skip the sum. Remapping is linear,
  so summing before or after the remap gives identical answers.
- *Sanity checks.* The non-negativity check is precipitation-specific.
  Substitute something appropriate — plausible range for temperature,
  hemispheric asymmetry for a dynamical field, whatever catches a bad remap
  for your variable.
- *Region.* `LAT_SLICE` / `LON_SLICE` are set to a North America box
  (0–50°N, 225–305°E). Longitude is 0–360, matching CAM's convention.

**Weighting.** Use the `area` variable from the regridded file rather than
`cos(lat)`:

```python
area = xr.open_dataset(f)["area"].sel(lat=LAT_SLICE, lon=LON_SLICE)
regional_mean = da.weighted(area).mean(("lat", "lon"))
```

`cos(lat)` is a fine approximation away from the poles, but `area` is exact
and handles the FV polar caps correctly if the domain is ever widened.

**This map is atmosphere-only.** It will not work on ocean or sea-ice output,
which is on the POP/CICE tripole `t12` grid, nor on land-model output with a
different `ncol`. Those need their own map files. If `ncremap` errors about a
dimension mismatch, check that the input really is CAM output with
`ncol = 777602`.

## Status

Done:

- RCP6.0, member 010, PRECC + PRECL, 2006–2025 → `PRECT.NAM.fv0.23x0.31.nc`
- Domain-mean PRECT 3.74 mm/day, May–Oct maximum, no NaNs or negatives

Not done:

- Historical 1920–2005 (uncomment the block in the shell script, change
  `CASE` in the notebook)
- Members 001–009 — wrap the shell script in a loop over `MEMBER` and submit
  as a PBS array job. Read the map from campaign, not from a home directory,
  so concurrent jobs aren't all hitting the same home filesystem.
- The `cos(lat)` weighting in the notebook still needs swapping for `area`
- Comparison against the earlier `nearest_s2d` output, to quantify how much
  the interpolation method mattered
