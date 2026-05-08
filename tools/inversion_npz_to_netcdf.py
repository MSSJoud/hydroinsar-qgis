#!/usr/bin/env python3
"""Convert grouped/layered inversion NPZ outputs into plugin-ready NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np


GROUPED_FALLBACK_NAMES = ("Load_total", "Sg")
LAYERED_FALLBACK_NAMES = ("S0", "Ss", "Sd", "Sg", "Sr")


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return cleaned.strip("_") or "var"


def infer_mode(data: np.lib.npyio.NpzFile) -> str:
    if "x_layered" in data.files:
        return "layered"
    if "x_final" in data.files:
        return "grouped"
    if "x_prior" in data.files:
        return "grouped"
    raise ValueError("Could not infer NPZ mode. Expected either 'x_layered' or 'x_prior'.")


def extract_field_names(data: np.lib.npyio.NpzFile, mode: str) -> list[str]:
    if "field_names" in data.files:
        return [sanitize_name(name) for name in data["field_names"].tolist()]
    if mode == "layered":
        return list(LAYERED_FALLBACK_NAMES)
    return list(GROUPED_FALLBACK_NAMES)


def create_variable(ds, name, values, dims, units="mm", long_name=None):
    var = ds.createVariable(name, "f4", dims, zlib=True, complevel=4, fill_value=np.nan)
    var[:] = values.astype(np.float32)
    var.units = units
    var.long_name = long_name or name
    return var


def write_time_variable(ds, time_values: np.ndarray, time_origin: str | None):
    time_var = ds.createVariable("time", "f8", ("time",))
    time_var[:] = np.asarray(time_values, dtype=np.float64)
    time_var.long_name = "time"
    if time_origin:
        time_var.units = f"days since {time_origin}"
        time_var.calendar = "standard"
    else:
        time_var.units = "days"
        time_var.comment = "Relative day offsets. Set --time-origin during export for calendar dates."


def export_npz_to_netcdf(input_path: Path, output_path: Path, mode: str, time_origin: str | None) -> None:
    src = np.load(input_path, allow_pickle=True)
    resolved_mode = infer_mode(src) if mode == "auto" else mode
    field_names = extract_field_names(src, resolved_mode)

    if resolved_mode == "layered":
        cube = src["x_layered"].astype(np.float32)
    else:
        if "x_final" in src.files:
            cube = src["x_final"].astype(np.float32)
        else:
            cube = src["x_prior"].astype(np.float32)

    if cube.ndim != 4:
        raise ValueError(f"Expected data cube with shape (time, variable, y, x), got {cube.shape}.")

    time_values = np.asarray(src["time"])
    lat = np.asarray(src["lat"], dtype=np.float32)
    lon = np.asarray(src["lon"], dtype=np.float32)

    if lat.shape != lon.shape:
        raise ValueError(f"Latitude/longitude shapes must match, got {lat.shape} and {lon.shape}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", cube.shape[0])
        ds.createDimension("y", cube.shape[2])
        ds.createDimension("x", cube.shape[3])

        write_time_variable(ds, time_values, time_origin)

        lat_var = ds.createVariable("lat", "f4", ("y", "x"), zlib=True, complevel=4)
        lon_var = ds.createVariable("lon", "f4", ("y", "x"), zlib=True, complevel=4)
        lat_var[:] = lat
        lon_var[:] = lon
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        for idx, field_name in enumerate(field_names):
            create_variable(
                ds,
                name=field_name,
                values=cube[:, idx, :, :],
                dims=("time", "y", "x"),
                long_name=field_name,
            )

        if "y_obs" in src.files:
            create_variable(ds, "insar_observed", src["y_obs"], ("time", "y", "x"), long_name="Observed InSAR deformation")
        if "y_pred" in src.files:
            create_variable(ds, "insar_predicted", src["y_pred"], ("time", "y", "x"), long_name="Predicted InSAR deformation")
        if "residual" in src.files:
            create_variable(ds, "insar_residual", src["residual"], ("time", "y", "x"), long_name="Observed minus predicted InSAR deformation")
        if "d_prior" in src.files:
            create_variable(ds, "insar_prior", src["d_prior"], ("time", "y", "x"), long_name="Stage 2 prior deformation")
        if "d_final" in src.files:
            create_variable(ds, "insar_final", src["d_final"], ("time", "y", "x"), long_name="Stage 2 final deformation")
        if "tws" in src.files:
            create_variable(ds, "TWS", src["tws"], ("time", "y", "x"), long_name="Total water storage")
        if "load_total" in src.files:
            create_variable(ds, "Load_total", src["load_total"], ("time", "y", "x"), long_name="Grouped load total")
        if "sg" in src.files:
            create_variable(ds, "Sg_grouped", src["sg"], ("time", "y", "x"), long_name="Grouped groundwater signal")

        ds.source_npz = str(input_path)
        ds.export_mode = resolved_mode
        ds.history = "Converted from inversion NPZ output for W3RA Explorer."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to grouped or layered inversion .npz file.")
    parser.add_argument("--output", required=True, help="Output NetCDF path.")
    parser.add_argument("--mode", choices=("auto", "grouped", "layered"), default="auto")
    parser.add_argument(
        "--time-origin",
        default=None,
        help="Optional calendar origin in YYYY-MM-DD for the stored day offsets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_npz_to_netcdf(
        input_path=Path(args.input),
        output_path=Path(args.output),
        mode=args.mode,
        time_origin=args.time_origin,
    )
