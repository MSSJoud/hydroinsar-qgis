# HydroInSAR QGIS

HydroInSAR QGIS is the developed successor to the original `W3RAExplorer` plugin, extending it from a NetCDF time-series viewer into a QGIS frontend for W3RA-InSAR grouped, layered, and hybrid backend workflows.

It remains lightweight for map-side exploration, while adding backend launch tools for grouped Stage 1, grouped-to-layered export, Stage 2 residual processing, and plugin-ready NetCDF export.

It now supports two NetCDF styles:

- legacy yearly W3RA variables such as `Sg_EU_2010`, `Sg_EU_2011`, with matching `time_YYYY`
- standard 3D variables such as `S0`, `Ss`, `Sd`, `Sg`, `Sr`, `Load_total`, stored on `(time, y, x)` or `(time, lat, lon)`

## What It Does

- opens a NetCDF file from QGIS
- lets you click on the map to inspect the nearest grid cell
- plots the raw time series and anomalies
- overlays optional fits: linear, polynomial, exponential, Gaussian smoothing, Fourier
- includes a backend runner for grouped Stage 1, grouped-to-layered export, Stage 2 residual runs, and NPZ-to-NetCDF export
- works well for W3RA grids and for grouped/layered inversion products exported from your backend

## Current Repo Status

The plugin entry point is valid and the Python files compile successfully. The main limitation is that this repository is only the plugin code: full runtime verification still requires a local QGIS installation with the needed Python packages available inside the QGIS Python environment.

## Requirements

- QGIS 3.x
- Python modules available in the QGIS Python environment:
  - `numpy`
  - `matplotlib`
  - `netCDF4`
  - `PyQt5`
  - `scipy`

Optional conversion utilities also use:

- `geopandas`
- `shapely`
- `pandas`
- `scipy.io`

## Install In QGIS

1. Install from the packaged plugin ZIP in this repository root:
   - `hydroinsar_qgis.zip`
2. Or create your own QGIS-ready ZIP from the full plugin folder.
   Important:
   The ZIP should contain the entire plugin package, not just a single `.py` file.
   It should include at least:
   - `__init__.py`
   - `metadata.txt`
   - `W3RAExplorer.py`
   - `backend_dialog.py`
   - `netcdf2shp.py`
   - `tools/`
3. In QGIS, open `Plugins -> Manage and Install Plugins -> Install from ZIP`.
4. Select the ZIP file.
5. Enable `W3RA Explorer`.

### Build The ZIP On Windows

If you cloned the repo on Windows, the safest approach is to zip the whole plugin folder with an underscore package name:

```powershell
Copy-Item .\hydroinsar-qgis .\hydroinsar_qgis -Recurse
Remove-Item .\hydroinsar_qgis\.git -Recurse -Force
Compress-Archive -Path .\hydroinsar_qgis -DestinationPath .\hydroinsar_qgis.zip -Force
```

Then verify:

```powershell
tar -tf .\hydroinsar_qgis.zip
```

## Run The Plugin

1. Open QGIS.
2. Add a base map or any layer that helps you click in the correct geographic area.
3. Start the plugin from the toolbar or plugin menu.
4. Choose a NetCDF file.
5. Click near the grid cell you want to inspect.
6. Pick the variable from the dialog.
7. Review the plotted series and optional regression overlays.

The plugin reads the series directly from the NetCDF file. You do not need to convert the grid to a dense point layer for the basic click-and-plot workflow.

## Run The Backend From QGIS

The plugin now adds a second action: `W3RA Backend Runner`.

Use it when you want to:

- run grouped Stage 1 on Bologna InSAR/W3RA inputs
- derive layered inference from grouped results
- run the Stage 2 residual Swin step
- export grouped or layered `.npz` outputs to a plugin-ready NetCDF file

Recommended path inside the dialog:

1. Run `Stage 1 Grouped`.
2. Run `Export Layered Inference` if you want `S0`, `Ss`, `Sd`, `Sg`, `Sr`.
3. Optionally run `Stage 2 Residual` if you want the hybrid grouped refinement.
4. Run `Export NPZ To NetCDF`.
5. Click `Load NetCDF In Explorer`.
6. Click on the map to inspect grouped or layered time series.

## Convert W3RA `.mat` Output To NetCDF

Use the existing conversion script:

```bash
python3 tools/w3ra_mat_2netcdf.py \
  --input-dir /path/to/w3ra_mat_dir \
  --latlon-file /path/to/LatLon.mat \
  --output-file /tmp/W3RA_2010_2024.nc
```

## Convert Inversion Output To Plugin-Ready NetCDF

This repo now includes a backend bridge for grouped or layered inversion outputs:

```bash
python3 tools/inversion_npz_to_netcdf.py \
  --input /home/ubuntu/work/insar_mcmc/outputs_layered_inference_from_grouped_full/layered_inference_from_grouped.npz \
  --output /tmp/layered_inference_for_qgis.nc \
  --mode layered \
  --time-origin 2017-01-04
```

Or for the grouped Stage 1 product:

```bash
python3 tools/inversion_npz_to_netcdf.py \
  --input /home/ubuntu/work/insar_mcmc/outputs_stage1_bologna_real_full_grouped_quick/stage1_bologna_real_results.npz \
  --output /tmp/grouped_inference_for_qgis.nc \
  --mode grouped \
  --time-origin 2017-01-04
```

After export, open the resulting `.nc` file with the plugin and click on the map.

## Backend Development Direction

The most practical backend path is:

1. run the grouped Stage 1 inversion
2. optionally derive layered outputs from the grouped posterior
3. export the result to a standard NetCDF product
4. let QGIS visualize and interrogate that product

This avoids pushing millions of dense InSAR points into the QGIS canvas when your real goal is grouped or gridded water-content time series.

## Rendering Advice For Dense InSAR

For dense InSAR points, do not use shapefiles with full time-series arrays per feature. That is slow in both storage and rendering.

Prefer:

- gridded NetCDF or raster layers for map display
- decimated preview points only when needed
- on-click retrieval from the original cube
- tiled products for heavy workflows
- GeoPackage instead of Shapefile if you must store vector features

In your case, the existing grouped and tiled inversion outputs are already a better visualization target than raw dense InSAR points.
