import os
import re

import netCDF4 as nc
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsMapToolIdentify, QgsVertexMarker
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QAction, QComboBox, QDialog, QFileDialog, QInputDialog, QLabel, QLineEdit, QMessageBox, QVBoxLayout
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from .backend_dialog import BackendRunnerDialog


YEAR_SUFFIX_RE = re.compile(r"(.+)_\d{4}$")
MAX_POINT_LAYER_FEATURES = 100000
LATITUDE_NAMES = ("lat", "latitude", "y")
LONGITUDE_NAMES = ("lon", "longitude", "x")
FRIENDLY_VARIABLE_NAMES = {
    "S0": "Surface/Shallow storage",
    "Ss": "Shallow soil water",
    "Sd": "Deep soil water",
    "Sg": "Groundwater",
    "Sr": "Surface water / runoff store",
    "Load_total": "Total water load",
    "Shallow_water": "Grouped shallow water",
    "Groundwater": "Grouped groundwater",
    "Deep_water": "Grouped deep water",
    "insar_observed": "InSAR observed",
    "insar_predicted": "InSAR predicted",
    "insar_residual": "InSAR residual",
    "insar_prior": "InSAR prior",
    "insar_final": "InSAR final",
}


def _find_variable_name(dataset, candidates):
    lower_lookup = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    return None


def _coordinate_names(dataset):
    lat_name = _find_variable_name(dataset, LATITUDE_NAMES)
    lon_name = _find_variable_name(dataset, LONGITUDE_NAMES)
    if not lat_name or not lon_name:
        raise ValueError("NetCDF file must contain latitude/longitude coordinates, such as 'lat'/'lon'.")
    return lat_name, lon_name


def _coordinate_arrays(dataset):
    lat_name, lon_name = _coordinate_names(dataset)
    return dataset.variables[lat_name][:], dataset.variables[lon_name][:]


def _time_dim_name(variable):
    for dim in variable.dimensions:
        if dim.lower().startswith("time"):
            return dim
    return variable.dimensions[0] if variable.ndim in (2, 3) else None


def _is_time_series_variable(name, variable, coordinate_names):
    if name in coordinate_names or variable.ndim not in (2, 3):
        return False
    return _time_dim_name(variable) is not None


def _has_gridded_time_series(dataset):
    coordinate_names = set(_coordinate_names(dataset))
    return any(
        _is_time_series_variable(name, variable, coordinate_names) and variable.ndim == 3
        for name, variable in dataset.variables.items()
    )


def _list_base_variables(dataset):
    coordinate_names = set(_coordinate_names(dataset))
    base_names = set()
    for name, variable in dataset.variables.items():
        if not _is_time_series_variable(name, variable, coordinate_names):
            continue
        match = YEAR_SUFFIX_RE.match(name)
        base_names.add(match.group(1) if match else name)
    return sorted(base_names)


def _variable_display_name(dataset, var_name):
    friendly = FRIENDLY_VARIABLE_NAMES.get(var_name)
    long_name = None
    if var_name in dataset.variables:
        long_name = getattr(dataset.variables[var_name], "long_name", None)
    if friendly and long_name and long_name != var_name and long_name != friendly:
        return f"{friendly} ({var_name})"
    if friendly:
        return f"{friendly} ({var_name})"
    if long_name and long_name != var_name:
        return f"{long_name} ({var_name})"
    return var_name


def _variable_choices(dataset):
    return [(_variable_display_name(dataset, name), name) for name in _list_base_variables(dataset)]


def _nearest_grid_index(latitudes, longitudes, lat, lon, unstructured=False):
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    if unstructured and latitudes.ndim == 1 and longitudes.ndim == 1 and latitudes.shape == longitudes.shape:
        dist2 = (latitudes - lat) ** 2 + (longitudes - lon) ** 2
        return int(np.argmin(dist2)), 0
    if latitudes.ndim == 1 and longitudes.ndim == 1:
        return int(np.abs(latitudes - lat).argmin()), int(np.abs(longitudes - lon).argmin())
    if latitudes.ndim == 2 and longitudes.ndim == 2:
        dist2 = (latitudes - lat) ** 2 + (longitudes - lon) ** 2
        return np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    raise ValueError("Expected coordinate arrays to be either both 1D or both 2D.")


def _load_time_values(dataset, time_name):
    if time_name not in dataset.variables:
        return None
    time_var = dataset.variables[time_name]
    raw = np.asarray(time_var[:])
    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    if units:
        try:
            return np.asarray(nc.num2date(raw, units=units, calendar=calendar))
        except Exception:
            return raw
    return raw


def _extract_standard_series(dataset, var_name, lat_idx, lon_idx):
    variable = dataset.variables[var_name]
    data = np.asarray(variable[:])
    time_name = _time_dim_name(variable)
    time_axis = variable.dimensions.index(time_name)
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)

    if data.ndim == 2:
        values = data[:, lat_idx]
    else:
        values = data[:, lat_idx, lon_idx]

    times = _load_time_values(dataset, time_name)
    if times is None:
        times = np.arange(values.shape[0])
    return np.asarray(times), np.asarray(values)


def _extract_yearly_series(dataset, var_name, lat_idx, lon_idx):
    year_pairs = []
    for name, variable in dataset.variables.items():
        match = YEAR_SUFFIX_RE.match(name)
        if match and match.group(1) == var_name and variable.ndim == 3:
            year_pairs.append((int(name.split("_")[-1]), name))

    all_times = []
    all_values = []
    for year, variable_name in sorted(year_pairs):
        time_name = f"time_{year}"
        if time_name not in dataset.variables:
            continue
        data = np.asarray(dataset.variables[variable_name][:])
        if data.ndim == 2:
            values = data[:, lat_idx]
        else:
            values = data[:, lat_idx, lon_idx]
        times = _load_time_values(dataset, time_name)
        if times is None:
            times = np.arange(values.shape[0])
        all_times.extend(np.asarray(times).tolist())
        all_values.extend(values.tolist())
    return np.asarray(all_times), np.asarray(all_values)


def extract_point_series_at_index(dataset, var_name, lat_idx, lon_idx):
    if var_name in dataset.variables and dataset.variables[var_name].ndim in (2, 3):
        return _extract_standard_series(dataset, var_name, lat_idx, lon_idx)
    return _extract_yearly_series(dataset, var_name, lat_idx, lon_idx)


def extract_point_series(dataset, var_name, lat, lon):
    latitudes, longitudes = _coordinate_arrays(dataset)
    unstructured = var_name in dataset.variables and dataset.variables[var_name].ndim == 2
    lat_idx, lon_idx = _nearest_grid_index(latitudes, longitudes, lat, lon, unstructured=unstructured)

    return extract_point_series_at_index(dataset, var_name, lat_idx, lon_idx)


def _iter_grid_points(latitudes, longitudes, unstructured=False, max_features=MAX_POINT_LAYER_FEATURES):
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    if unstructured and latitudes.ndim == 1 and longitudes.ndim == 1 and latitudes.shape == longitudes.shape:
        total = latitudes.size
        stride = max(1, int(np.ceil(total / max_features))) if total > max_features else 1
        for point_idx in range(0, latitudes.size, stride):
            lat = float(latitudes[point_idx])
            lon = float(longitudes[point_idx])
            if np.isfinite(lat) and np.isfinite(lon):
                yield point_idx, 0, lat, lon, stride
        return

    if latitudes.ndim == 1 and longitudes.ndim == 1:
        total = latitudes.size * longitudes.size
        stride = max(1, int(np.ceil(np.sqrt(total / max_features)))) if total > max_features else 1
        for lat_idx in range(0, latitudes.size, stride):
            lat = float(latitudes[lat_idx])
            if not np.isfinite(lat):
                continue
            for lon_idx in range(0, longitudes.size, stride):
                lon = float(longitudes[lon_idx])
                if np.isfinite(lon):
                    yield lat_idx, lon_idx, lat, lon, stride
        return

    if latitudes.ndim == 2 and longitudes.ndim == 2 and latitudes.shape == longitudes.shape:
        total = latitudes.size
        stride = max(1, int(np.ceil(np.sqrt(total / max_features)))) if total > max_features else 1
        for lat_idx in range(0, latitudes.shape[0], stride):
            for lon_idx in range(0, latitudes.shape[1], stride):
                lat = float(latitudes[lat_idx, lon_idx])
                lon = float(longitudes[lat_idx, lon_idx])
                if np.isfinite(lat) and np.isfinite(lon):
                    yield lat_idx, lon_idx, lat, lon, stride
        return

    raise ValueError("Expected coordinate arrays to be either both 1D or matching 2D grids.")


def _dialog_exec(dialog):
    exec_fn = getattr(dialog, "exec", None) or dialog.exec_
    return exec_fn()


def _map_point_to_dataset_lonlat(canvas, point):
    source_crs = canvas.mapSettings().destinationCrs()
    target_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    if not source_crs.isValid() or source_crs == target_crs:
        return point.x(), point.y()
    transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance().transformContext())
    transformed = transform.transform(point)
    return transformed.x(), transformed.y()


class InteractivePlotDialog(QDialog):
    def __init__(self, x, y, anomalies, var_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Time Series Analysis")
        self.resize(800, 600)

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.anomalies = np.asarray(anomalies)
        self.var_name = var_name
        self.plot_x = self._numeric_x()
        self.x_labels = [self._format_x_label(value) for value in self.x]

        layout = QVBoxLayout(self)

        self.combo = QComboBox()
        self.combo.addItems(["None", "Linear", "Polynomial", "Exponential", "Gaussian Smoothing", "Fourier"])
        self.combo.currentIndexChanged.connect(self.update_plot)

        self.param_input = QLineEdit()
        self.param_input.setPlaceholderText("Degree (Poly), Sigma (Gaussian), Terms (Fourier)")
        self.param_input.textChanged.connect(self.update_plot)

        layout.addWidget(QLabel("Regression Type:"))
        layout.addWidget(self.combo)
        layout.addWidget(self.param_input)

        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.plot_data()

    def plot_data(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)

        ax1.plot(self.plot_x, self.y, "o-", label=self.var_name, color="red", markersize=4)
        ax1.set_title(f"{self.var_name} Time Series")
        ax1.set_ylabel("Value")
        ax1.grid(True)

        ax2.plot(self.plot_x, self.anomalies, "o--", label="Anomalies", color="blue", markersize=4)
        ax2.set_title("Anomalies")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Anomaly")
        ax2.grid(True)
        self._apply_time_ticks(ax2)

        self.ax1 = ax1
        self.canvas.draw()
        self.update_plot()

    def _numeric_x(self):
        if np.issubdtype(self.x.dtype, np.number):
            return self.x.astype(float)
        return np.arange(self.x.shape[0], dtype=float)

    def _format_x_label(self, value):
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%Y-%m-%d")
            except Exception:
                pass
        return str(value)

    def _apply_time_ticks(self, axis):
        if self.plot_x.size == 0:
            return
        tick_count = min(6, self.plot_x.size)
        tick_indices = np.linspace(0, self.plot_x.size - 1, tick_count, dtype=int)
        tick_indices = np.unique(tick_indices)
        axis.set_xticks(self.plot_x[tick_indices])
        axis.set_xticklabels([self.x_labels[idx] for idx in tick_indices], rotation=30, ha="right")
        self.fig.tight_layout()

    def update_plot(self):
        reg_type = self.combo.currentText()
        param = self.param_input.text().strip()

        while len(self.ax1.lines) > 1:
            self.ax1.lines[-1].remove()

        x = self.plot_x
        y = self.y.astype(float)

        try:
            if reg_type == "Linear":
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, x)
                self.ax1.plot(self.plot_x, y_fit, label="Linear Fit", linestyle="--")
            elif reg_type == "Polynomial":
                degree = int(param) if param else 2
                coeffs = np.polyfit(x, y, degree)
                y_fit = np.polyval(coeffs, x)
                self.ax1.plot(self.plot_x, y_fit, label=f"Poly deg={degree}", linestyle="--")
            elif reg_type == "Exponential":
                def model(x_val, a_val, b_val, c_val):
                    return a_val * np.exp(b_val * x_val) + c_val

                popt, _ = curve_fit(model, x, y, maxfev=10000)
                self.ax1.plot(self.plot_x, model(x, *popt), label="Exp Fit", linestyle="--")
            elif reg_type == "Gaussian Smoothing":
                sigma = float(param) if param else 3.0
                y_smooth = gaussian_filter1d(y, sigma)
                self.ax1.plot(self.plot_x, y_smooth, label=f"Gaussian sigma={sigma:g}", linestyle="--")
            elif reg_type == "Fourier":
                terms = int(param) if param else 5
                fft_coeffs = np.fft.rfft(y)
                fft_coeffs[terms:] = 0
                y_ifft = np.fft.irfft(fft_coeffs, n=y.shape[0])
                self.ax1.plot(self.plot_x, y_ifft, label=f"Fourier ({terms} terms)", linestyle="--")
        except Exception as exc:
            print("Regression update error:", exc)

        self.ax1.legend()
        self.canvas.draw()


class PointClickTool(QgsMapToolIdentify):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.marker = None

    def _grid_index_from_click(self, event):
        layer_id = QgsProject.instance().readEntry("W3RAExplorer", "PointLayerId")[0]
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
        if layer is None:
            return None

        try:
            results = self.identify(event.x(), event.y(), [layer], QgsMapToolIdentify.LayerSelection)
        except Exception:
            return None

        if not results:
            return None

        feature = getattr(results[0], "mFeature", None)
        if feature is None and hasattr(results[0], "feature"):
            feature = results[0].feature()
        if feature is None:
            return None

        try:
            return int(feature["grid_i"]), int(feature["grid_j"])
        except Exception:
            return None

    def canvasReleaseEvent(self, event):
        point = self.canvas.getCoordinateTransform().toMapCoordinates(event.pos())
        lon, lat = _map_point_to_dataset_lonlat(self.canvas, point)
        grid_index = self._grid_index_from_click(event)

        if self.marker:
            self.canvas.scene().removeItem(self.marker)

        self.marker = QgsVertexMarker(self.canvas)
        self.marker.setCenter(point)
        self.marker.setColor(QColor("red"))
        self.marker.setIconSize(12)
        self.marker.setIconType(QgsVertexMarker.ICON_CROSS)
        self.marker.setPenWidth(3)

        input_file = QgsProject.instance().readEntry("W3RAExplorer", "NetCDF_Path")[0]
        if not input_file or not os.path.exists(input_file):
            QMessageBox.critical(None, "Error", "NetCDF file not found. Please load a valid file.")
            return

        try:
            with nc.Dataset(input_file, "r") as dataset:
                variable_choices = _variable_choices(dataset)
                if not variable_choices:
                    QMessageBox.warning(None, "No Variables", "No time-dependent variables were found in this NetCDF file.")
                    return

                labels = [label for label, _ in variable_choices]
                selected_label, ok = QInputDialog.getItem(None, "Select Variable", "Variable:", labels, 0, False)
                if not ok:
                    return
                var_name = dict(variable_choices)[selected_label]

                if grid_index is not None:
                    x, y = extract_point_series_at_index(dataset, var_name, grid_index[0], grid_index[1])
                else:
                    x, y = extract_point_series(dataset, var_name, lat, lon)

            if y.size == 0:
                QMessageBox.warning(None, "No Data", "No data found at this location.")
                return

            y = np.asarray(y, dtype=float)
            anomalies = y - np.nanmean(y)
            dlg = InteractivePlotDialog(x, y, anomalies, var_name)
            _dialog_exec(dlg)
        except Exception as exc:
            QMessageBox.critical(None, "Error", str(exc))


class W3RAExplorer:
    def __init__(self, iface):
        self.iface = iface
        self.tool = None
        self.explorer_action = None
        self.backend_action = None
        self.backend_dialog = None
        self.point_layer = None

    def initGui(self):
        self.tool = PointClickTool(self.iface.mapCanvas())
        self.explorer_action = QAction("HydroInSAR Explorer", self.iface.mainWindow())
        self.explorer_action.triggered.connect(self.activate_plugin)
        self.iface.addToolBarIcon(self.explorer_action)
        self.iface.addPluginToMenu("&HydroInSAR Explorer", self.explorer_action)

        self.backend_action = QAction("HydroInSAR Backend Runner", self.iface.mainWindow())
        self.backend_action.triggered.connect(self.open_backend_dialog)
        self.iface.addToolBarIcon(self.backend_action)
        self.iface.addPluginToMenu("&HydroInSAR Explorer", self.backend_action)

    def activate_plugin(self):
        testdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hydroinsar_qgis_testdata"))
        start_dir = testdata_dir if os.path.isdir(testdata_dir) else ""
        input_file, _ = QFileDialog.getOpenFileName(None, "Select NetCDF File", start_dir, "NetCDF (*.nc)")
        if input_file:
            self.load_netcdf_for_explorer(input_file)

    def load_netcdf_for_explorer(self, input_file):
        try:
            point_layer, feature_count, stride = self.create_point_layer(input_file)
        except Exception as exc:
            QMessageBox.critical(None, "HydroInSAR Explorer", f"Could not load NetCDF grid:\n{exc}")
            return

        QgsProject.instance().writeEntry("W3RAExplorer", "NetCDF_Path", input_file)
        QgsProject.instance().writeEntry("W3RAExplorer", "PointLayerId", point_layer.id())
        self.point_layer = point_layer
        self.iface.setActiveLayer(point_layer)
        self.zoom_to_layer(point_layer)
        self.iface.mapCanvas().setMapTool(self.tool)

        sample_note = "" if stride == 1 else f"\nThe grid was sampled every {stride} cells to keep QGIS responsive."
        QMessageBox.information(
            None,
            "HydroInSAR Explorer",
            f"Loaded {feature_count} grid points. Click a point to inspect its time series.{sample_note}",
        )

    def create_point_layer(self, input_file):
        with nc.Dataset(input_file, "r") as dataset:
            lat_name, lon_name = _coordinate_names(dataset)

            base_vars = _list_base_variables(dataset)
            if not base_vars:
                raise ValueError("No time-dependent variables were found in this NetCDF file.")

            latitudes = dataset.variables[lat_name][:]
            longitudes = dataset.variables[lon_name][:]
            unstructured = not _has_gridded_time_series(dataset)

            layer_name = f"HydroInSAR grid - {os.path.basename(input_file)}"
            layer = QgsVectorLayer("Point?crs=EPSG:4326", layer_name, "memory")
            provider = layer.dataProvider()
            provider.addAttributes(
                [
                    QgsField("grid_i", QVariant.Int),
                    QgsField("grid_j", QVariant.Int),
                    QgsField("latitude", QVariant.Double),
                    QgsField("longitude", QVariant.Double),
                ]
            )
            layer.updateFields()

            features = []
            stride = 1
            for lat_idx, lon_idx, lat, lon, stride in _iter_grid_points(latitudes, longitudes, unstructured=unstructured):
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
                feature.setAttributes([lat_idx, lon_idx, lat, lon])
                features.append(feature)

            if not features:
                raise ValueError("No finite latitude/longitude points were found.")

            provider.addFeatures(features)
            layer.updateExtents()
            symbol = QgsMarkerSymbol.createSimple({"name": "circle", "color": "220,40,40", "size": "2.0"})
            layer.renderer().setSymbol(symbol)

        old_layer_id = QgsProject.instance().readEntry("W3RAExplorer", "PointLayerId")[0]
        old_layer = QgsProject.instance().mapLayer(old_layer_id) if old_layer_id else None
        if old_layer is not None:
            QgsProject.instance().removeMapLayer(old_layer.id())

        QgsProject.instance().addMapLayer(layer)
        return layer, len(features), stride

    def zoom_to_layer(self, layer):
        extent = layer.extent()
        canvas = self.iface.mapCanvas()
        layer_crs = layer.crs()
        canvas_crs = canvas.mapSettings().destinationCrs()
        if layer_crs.isValid() and canvas_crs.isValid() and layer_crs != canvas_crs:
            transform = QgsCoordinateTransform(layer_crs, canvas_crs, QgsProject.instance().transformContext())
            extent = transform.transformBoundingBox(extent)
        canvas.setExtent(extent)
        canvas.refresh()

    def open_backend_dialog(self):
        if self.backend_dialog is None:
            self.backend_dialog = BackendRunnerDialog(
                iface=self.iface,
                load_netcdf_callback=self.load_netcdf_for_explorer,
                parent=self.iface.mainWindow(),
            )
        self.backend_dialog.show()
        self.backend_dialog.raise_()
        self.backend_dialog.activateWindow()

    def unload(self):
        if self.explorer_action:
            self.iface.removeToolBarIcon(self.explorer_action)
            self.iface.removePluginMenu("&HydroInSAR Explorer", self.explorer_action)
        if self.backend_action:
            self.iface.removeToolBarIcon(self.backend_action)
            self.iface.removePluginMenu("&HydroInSAR Explorer", self.backend_action)
        if self.tool:
            self.iface.mapCanvas().unsetMapTool(self.tool)
        layer_id = QgsProject.instance().readEntry("W3RAExplorer", "PointLayerId")[0]
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
        if layer is not None:
            QgsProject.instance().removeMapLayer(layer.id())
