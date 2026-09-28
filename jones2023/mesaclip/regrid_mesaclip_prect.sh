#!/bin/bash
# Remap MESACLIP / iHESP CESM-HR precipitation from the native ne120np4
# spectral-element grid to the CAM-FV 0.23x0.31 grid, using the
# project ESMF bilinear map file.
#
# Run on Casper (interactive is fine; this is I/O bound, not CPU bound):
#   execcasper -A <your_project> -l select=1:ncpus=4:mem=40GB -l walltime=2:00:00
#   bash regrid_mesaclip_prect.sh
#
# Produces one regridded file per variable per decade in $OUTDIR.
# Combine them afterward with mesaclip_prect_nam.py.
#
# Written by: AVC

set -euo pipefail

module load nco

# ---------------------------------------------------------------- settings --

# Path to the ESMF map file (src=ne120np4 777602 nodes, dst=1152x768 FV).
# NOTE: the filename contains dots, hence the quoting throughout this script.
MAP="/glade/work/$USER/regrid/map_ne120_to_0.23x0.31_bilinear.nc"

OUTDIR=/glade/campaign/univ/udrt0019/regridded
mkdir -p "$OUTDIR"

# Ensemble member, zero-padded (001 .. 010)
MEMBER=010

# Pick ONE scenario block by commenting the other out.

# --- RCP6.0, 2006-2100 (RDA d651008) ---
CASE="b.e13.BRCP60C5.ne120_t12.cesm-ihesp-hires1.0.46-2006-2100.${MEMBER}"
ROOT="/glade/campaign/collections/gdex/data/d651008/${CASE}/atm/proc/tseries/month_1"
YEARS="200601-201512 201601-202512"

# --- Historical, 1920-2005 (RDA d651007) ---
# CASE="b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.46-1920-2005.${MEMBER}"
# ROOT="/glade/campaign/collections/gdex/data/d651007/${CASE}/atm/proc/tseries/month_1"
# YEARS="192001-192912 193001-193912 194001-194912 195001-195912 196001-196912 197001-197912 198001-198912 199001-199912 200001-200512"

# ------------------------------------------------------------- sanity check --

if [[ ! -f "$MAP" ]]; then
  echo "ERROR: map file not found: $MAP" >&2
  exit 1
fi

# The source grid dimension must be 777602 (ne120np4 GLL node count).
# If this prints anything else, the map is for a different grid and the
# remap below will either fail or produce garbage.
echo "--- map file source/destination dimensions ---"
ncks -H -C -v src_grid_dims,dst_grid_dims "$MAP"
echo "--- map method ---"
ncks -M "$MAP" | grep -iE 'map_method|normalization' || true
echo

# ------------------------------------------------------------------- remap --

for VAR in PRECC PRECL; do
  for YR in $YEARS; do
    IN="${ROOT}/${CASE}.cam.h0.${VAR}.${YR}.nc"
    OUT="${OUTDIR}/${CASE}.cam.h0.${VAR}.${YR}.fv0.23x0.31.nc"

    if [[ ! -f "$IN" ]]; then
      echo "MISSING INPUT, skipping: $IN" >&2
      continue
    fi
    if [[ -f "$OUT" ]]; then
      echo "exists, skipping: $(basename "$OUT")"
      continue
    fi

    echo "remapping ${VAR} ${YR} ..."
    # -v limits the regridder to the variable of interest (plus coordinates
    # and time_bnds, which ncremap carries through automatically).
    ncremap -v "${VAR}" -m "$MAP" "$IN" "$OUT"
  done
done

echo
echo "--- verification: check that time_bnds survived ---"
FIRST=$(ls "${OUTDIR}"/*.PRECC.*.fv0.23x0.31.nc | head -1)
ncks -m "$FIRST" | grep -iE 'time_bnds|^lat |^lon ' || \
  echo "WARNING: time_bnds not found -- see note in mesaclip_prect_nam.py"

echo
echo "Done. Output in: $OUTDIR"
