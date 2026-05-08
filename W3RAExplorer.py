import os
import re

import netCDF4 as nc
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject
from qgis.gui import QgsMapTool, QgsVertexMarker
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction, QComboBox, QDialog, QFileDialog, QInputDialog, QLabel, QLineEdit, QMessageBox, QVBoxLayout
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from .backend_dialog import BackendRunnerDialog


YEAR_SUFFIX_RE = re.compile(r"(.+)_\d{4}$")


def _time_dim_name(variable):
    for dim in variable.dimensions:
        if dim.lower().startswith("time"):
            return dim
    return variable.dimensions[0] if variable.ndim == 3 else None


def _list_base_variables(dataset):
    base_names = set()
    for name, variable in dataset.variables.items():
        if variable.ndim != 3:
            continue
        time_dim = _time_dim_name(variable)
        if time_dim is None:
            continue
        match = YEAR_SUFFIX_RE.match(name)
        base_names.add(match.group(1) if match else name)
    return sorted(base_names)


def _nearest_grid_index(latitudes, longitudes, lat, lon):
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
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
        values = np.asarray(dataset.variables[variable_name][:, lat_idx, lon_idx])
        times = _load_time_values(dataset, time_name)
        if times is None:
            times = np.arange(values.shape[0])
        all_times.extend(np.asarray(times).tolist())
        all_values.extend(values.tolist())
    return np.asarray(all_times), np.asarray(all_values)


def extract_point_series(dataset, var_name, lat, lon):
    if "lat" not in dataset.variables or "lon" not in dataset.variables:
        raise ValueError("NetCDF file must contain 'lat' and 'lon' coordinates.")

    latitudes = dataset.variables["lat"][:]
    longitudes = dataset.variables["lon"][:]
    lat_idx, lon_idx = _nearest_grid_index(latitudes, longitudes, lat, lon)

    if var_name in dataset.variables and dataset.variables[var_name].ndim == 3:
        return _extract_standard_series(dataset, var_name, lat_idx, lon_idx)
    return _extract_yearly_series(dataset, var_name, lat_idx, lon_idx)


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

        ax1.plot(self.x, self.y, "o-", label=self.var_name, color="red", markersize=4)
        ax1.set_title(f"{self.var_name} Time Series")
        ax1.set_ylabel("Value")
        ax1.grid(True)

        ax2.plot(self.x, self.anomalies, "o--", label="Anomalies", color="blue", markersize=4)
        ax2.set_title("Anomalies")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Anomaly")
        ax2.grid(True)

        self.ax1 = ax1
        self.canvas.draw()
        self.update_plot()

    def _numeric_x(self):
        if np.issubdtype(self.x.dtype, np.number):
            return self.x.astype(float)
        return np.arange(self.x.shape[0], dtype=float)

    def update_plot(self):
        reg_type = self.combo.currentText()
        param = self.param_input.text().strip()

        while len(self.ax1.lines) > 1:
            self.ax1.lines[-1].remove()

        x = self._numeric_x()
        y = self.y.astype(float)

        try:
            if reg_type == "Linear":
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, x)
                self.ax1.plot(self.x, y_fit, label="Linear Fit", linestyle="--")
            elif reg_type == "Polynomial":
                degree = int(param) if param else 2
                coeffs = np.polyfit(x, y, degree)
                y_fit = np.polyval(coeffs, x)
                self.ax1.plot(self.x, y_fit, label=f"Poly deg={degree}", linestyle="--")
            elif reg_type == "Exponential":
                def model(x_val, a_val, b_val, c_val):
                    return a_val * np.exp(b_val * x_val) + c_val

                popt, _ = curve_fit(model, x, y, maxfev=10000)
                self.ax1.plot(self.x, model(x, *popt), label="Exp Fit", linestyle="--")
            elif reg_type == "Gaussian Smoothing":
                sigma = float(param) if param else 3.0
                y_smooth = gaussian_filter1d(y, sigma)
                self.ax1.plot(self.x, y_smooth, label=f"Gaussian sigma={sigma:g}", linestyle="--")
            elif reg_type == "Fourier":
                terms = int(param) if param else 5
                fft_coeffs = np.fft.rfft(y)
                fft_coeffs[terms:] = 0
                y_ifft = np.fft.irfft(fft_coeffs, n=y.shape[0])
                self.ax1.plot(self.x, y_ifft, label=f"Fourier ({terms} terms)", linestyle="--")
        except Exception as exc:
            print("Regression update error:", exc)

        self.ax1.legend()
        self.canvas.draw()


class PointClickTool(QgsMapTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.marker = None

    def canvasReleaseEvent(self, event):
        point = self.canvas.getCoordinateTransform().toMapCoordinates(event.pos())
        lon, lat = _map_point_to_dataset_lonlat(self.canvas, point)

        if self.marker:
            self.canvas.scene().removeItem(self.marker)

        self.marker = QgsVertexMarker(self.canvas)
        self.marker.setCenter(point)
        self.marker.setColor(Qt.red)
        self.marker.setIconSize(12)
        self.marker.setIconType(QgsVertexMarker.ICON_CROSS)
        self.marker.setPenWidth(3)

        input_file = QgsProject.instance().readEntry("W3RAExplorer", "NetCDF_Path")[0]
        if not input_file or not os.path.exists(input_file):
            QMessageBox.critical(None, "Error", "NetCDF file not found. Please load a valid file.")
            return

        try:
            with nc.Dataset(input_file, "r") as dataset:
                base_vars = _list_base_variables(dataset)
                if not base_vars:
                    QMessageBox.warning(None, "No Variables", "No time-dependent 3D variables were found in this NetCDF file.")
                    return

                var_name, ok = QInputDialog.getItem(None, "Select Variable", "Variable:", base_vars, 0, False)
                if not ok:
                    return

                x, y = extract_point_series(dataset, var_name, lat, lon)

            if y.size == 0:
                QMessageBox.warning(None, "No Data", "No data found at this location.")
                return

            y = np.asarray(y, dtype=float)
            anomalies = y - np.nanmean(y)
            dlg = InteractivePlotDialog(x, y, anomalies, var_name)
            dlg.exec_()
        except Exception as exc:
            QMessageBox.critical(None, "Error", str(exc))


class W3RAExplorer:
    def __init__(self, iface):
        self.iface = iface
        self.tool = None
        self.explorer_action = None
        self.backend_action = None
        self.backend_dialog = None

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
        input_file, _ = QFileDialog.getOpenFileName(None, "Select NetCDF File", "", "NetCDF (*.nc)")
        if input_file:
            self.load_netcdf_for_explorer(input_file)

    def load_netcdf_for_explorer(self, input_file):
        QgsProject.instance().writeEntry("W3RAExplorer", "NetCDF_Path", input_file)
        self.iface.mapCanvas().setMapTool(self.tool)
        QMessageBox.information(
            None,
            "HydroInSAR Explorer",
            "NetCDF selected. Click on the map near the grid cell you want to inspect.",
        )

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
