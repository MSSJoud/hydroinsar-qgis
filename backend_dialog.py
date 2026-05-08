import os

from qgis.core import QgsProject
from qgis.PyQt.QtCore import QProcess, QSettings
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)


class BackendRunnerDialog(QDialog):
    def __init__(self, iface, load_netcdf_callback, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.load_netcdf_callback = load_netcdf_callback
        self.process = None
        self.pending_action = None

        self.setWindowTitle("W3RA Backend Runner")
        self.resize(980, 860)

        self.settings = QSettings()
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_backend_dir = "/home/ubuntu/work/insar_mcmc"
        self.default_python = "/home/ubuntu/anaconda3/envs/insar/bin/python"
        self.default_time_origin = "2017-01-04"

        self._build_ui()
        self._restore_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_general_group())
        layout.addWidget(self._build_stage1_group())
        layout.addWidget(self._build_layered_group())
        layout.addWidget(self._build_stage2_group())
        layout.addWidget(self._build_netcdf_group())

        layout.addWidget(QLabel("Backend Log"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        actions_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop Running Process")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_process)
        self.load_button = QPushButton("Load NetCDF In Explorer")
        self.load_button.clicked.connect(self.load_current_netcdf)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addWidget(self.load_button)
        layout.addLayout(actions_layout)

    def _build_general_group(self):
        group = QGroupBox("General")
        form = QFormLayout(group)

        self.python_edit = self._path_row(form, "Python Executable", self.default_python, pick_dir=False)
        self.backend_dir_edit = self._path_row(form, "Backend Repo Dir", self.default_backend_dir, pick_dir=True)
        self.time_origin_edit = QLineEdit(self.default_time_origin)
        form.addRow("Time Origin", self.time_origin_edit)
        return group

    def _build_stage1_group(self):
        group = QGroupBox("Stage 1 Grouped MCMC")
        form = QFormLayout(group)

        self.stage1_insar_edit = self._path_row(form, "InSAR NetCDF", "/mnt/data/mcma/01/insar_sub.nc", pick_dir=False)
        self.stage1_w3ra_edit = self._path_row(form, "W3RA NetCDF", "/mnt/data/mcma/01/w3ra_sub_anom.nc", pick_dir=False)
        self.stage1_output_edit = self._path_row(
            form,
            "Stage 1 Output Dir",
            "/home/ubuntu/work/insar_mcmc/outputs_stage1_bologna_real_full_grouped_quick",
            pick_dir=True,
        )

        self.stage1_iter_spin = QSpinBox()
        self.stage1_iter_spin.setRange(1, 500)
        self.stage1_iter_spin.setValue(6)
        form.addRow("Iterations", self.stage1_iter_spin)

        self.stage1_burn_spin = QSpinBox()
        self.stage1_burn_spin.setRange(0, 500)
        self.stage1_burn_spin.setValue(2)
        form.addRow("Burn In", self.stage1_burn_spin)

        run_button = QPushButton("Run Stage 1 Grouped")
        run_button.clicked.connect(self.run_stage1_grouped)
        form.addRow(run_button)
        return group

    def _build_layered_group(self):
        group = QGroupBox("Grouped To Layered Export")
        form = QFormLayout(group)

        self.grouped_results_edit = self._path_row(
            form,
            "Grouped Results NPZ",
            "/home/ubuntu/work/insar_mcmc/outputs_stage1_bologna_real_full_grouped_quick/stage1_bologna_real_results.npz",
            pick_dir=False,
        )
        self.layered_output_edit = self._path_row(
            form,
            "Layered Output Dir",
            "/home/ubuntu/work/insar_mcmc/outputs_layered_inference_from_grouped_full",
            pick_dir=True,
        )

        run_button = QPushButton("Export Layered Inference")
        run_button.clicked.connect(self.run_layered_export)
        form.addRow(run_button)
        return group

    def _build_stage2_group(self):
        group = QGroupBox("Stage 2 Residual Swin")
        form = QFormLayout(group)

        self.stage2_input_edit = self._path_row(
            form,
            "Stage 2 Stage1 NPZ",
            "/home/ubuntu/work/insar_mcmc/outputs_stage1_bologna_real_full_grouped_quick/stage1_bologna_real_results.npz",
            pick_dir=False,
        )
        self.stage2_output_edit = self._path_row(
            form,
            "Stage 2 Output Dir",
            "/home/ubuntu/work/insar_mcmc/outputs_stage2_bologna_real_full_grouped_quick",
            pick_dir=True,
        )

        self.stage2_device_combo = QComboBox()
        self.stage2_device_combo.addItems(["auto", "cpu", "cuda"])
        form.addRow("Device", self.stage2_device_combo)

        self.stage2_epochs_spin = QSpinBox()
        self.stage2_epochs_spin.setRange(1, 500)
        self.stage2_epochs_spin.setValue(4)
        form.addRow("Max Epochs", self.stage2_epochs_spin)

        run_button = QPushButton("Run Stage 2 Residual")
        run_button.clicked.connect(self.run_stage2)
        form.addRow(run_button)
        return group

    def _build_netcdf_group(self):
        group = QGroupBox("Plugin NetCDF Export")
        form = QFormLayout(group)

        self.export_npz_edit = self._path_row(
            form,
            "Input NPZ",
            "/home/ubuntu/work/insar_mcmc/outputs_layered_inference_from_grouped_full/layered_inference_from_grouped.npz",
            pick_dir=False,
        )
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["auto", "grouped", "layered"])
        self.export_mode_combo.setCurrentText("layered")
        form.addRow("NPZ Mode", self.export_mode_combo)
        self.export_nc_edit = self._path_row(form, "Output NetCDF", "/tmp/layered_inference_for_qgis.nc", pick_dir=False, save_file=True)

        button_layout = QHBoxLayout()
        export_button = QPushButton("Export NPZ To NetCDF")
        export_button.clicked.connect(self.run_netcdf_export)
        button_layout.addWidget(export_button)

        use_grouped_button = QPushButton("Use Grouped Result")
        use_grouped_button.clicked.connect(self.prefill_grouped_export)
        button_layout.addWidget(use_grouped_button)

        use_layered_button = QPushButton("Use Layered Result")
        use_layered_button.clicked.connect(self.prefill_layered_export)
        button_layout.addWidget(use_layered_button)

        form.addRow(button_layout)
        return group

    def _path_row(self, form, label, default_value, pick_dir=False, save_file=False):
        container = QHBoxLayout()
        edit = QLineEdit(default_value)
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self._browse_path(edit, pick_dir=pick_dir, save_file=save_file))
        container.addWidget(edit)
        container.addWidget(button)
        form.addRow(label, container)
        return edit

    def _browse_path(self, edit, pick_dir=False, save_file=False):
        start_dir = edit.text().strip() or os.path.expanduser("~")
        if pick_dir:
            chosen = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        elif save_file:
            chosen, _ = QFileDialog.getSaveFileName(self, "Select Output File", start_dir, "NetCDF (*.nc);;All Files (*)")
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "Select File", start_dir, "All Files (*)")
        if chosen:
            edit.setText(chosen)

    def _settings_key(self, suffix):
        return f"W3RAExplorer/{suffix}"

    def _restore_line_edit(self, edit, suffix):
        stored = self.settings.value(self._settings_key(suffix), "", type=str)
        if stored:
            edit.setText(stored)

    def _save_line_edit(self, edit, suffix):
        self.settings.setValue(self._settings_key(suffix), edit.text().strip())

    def _restore_settings(self):
        pairs = [
            (self.python_edit, "python"),
            (self.backend_dir_edit, "backend_dir"),
            (self.time_origin_edit, "time_origin"),
            (self.stage1_insar_edit, "stage1_insar"),
            (self.stage1_w3ra_edit, "stage1_w3ra"),
            (self.stage1_output_edit, "stage1_output"),
            (self.grouped_results_edit, "grouped_results"),
            (self.layered_output_edit, "layered_output"),
            (self.stage2_input_edit, "stage2_input"),
            (self.stage2_output_edit, "stage2_output"),
            (self.export_npz_edit, "export_npz"),
            (self.export_nc_edit, "export_nc"),
        ]
        for edit, suffix in pairs:
            self._restore_line_edit(edit, suffix)

        self.stage1_iter_spin.setValue(int(self.settings.value(self._settings_key("stage1_iter"), self.stage1_iter_spin.value())))
        self.stage1_burn_spin.setValue(int(self.settings.value(self._settings_key("stage1_burn"), self.stage1_burn_spin.value())))
        self.stage2_epochs_spin.setValue(int(self.settings.value(self._settings_key("stage2_epochs"), self.stage2_epochs_spin.value())))
        self.stage2_device_combo.setCurrentText(self.settings.value(self._settings_key("stage2_device"), self.stage2_device_combo.currentText()))
        self.export_mode_combo.setCurrentText(self.settings.value(self._settings_key("export_mode"), self.export_mode_combo.currentText()))

    def _save_settings(self):
        pairs = [
            (self.python_edit, "python"),
            (self.backend_dir_edit, "backend_dir"),
            (self.time_origin_edit, "time_origin"),
            (self.stage1_insar_edit, "stage1_insar"),
            (self.stage1_w3ra_edit, "stage1_w3ra"),
            (self.stage1_output_edit, "stage1_output"),
            (self.grouped_results_edit, "grouped_results"),
            (self.layered_output_edit, "layered_output"),
            (self.stage2_input_edit, "stage2_input"),
            (self.stage2_output_edit, "stage2_output"),
            (self.export_npz_edit, "export_npz"),
            (self.export_nc_edit, "export_nc"),
        ]
        for edit, suffix in pairs:
            self._save_line_edit(edit, suffix)

        self.settings.setValue(self._settings_key("stage1_iter"), self.stage1_iter_spin.value())
        self.settings.setValue(self._settings_key("stage1_burn"), self.stage1_burn_spin.value())
        self.settings.setValue(self._settings_key("stage2_epochs"), self.stage2_epochs_spin.value())
        self.settings.setValue(self._settings_key("stage2_device"), self.stage2_device_combo.currentText())
        self.settings.setValue(self._settings_key("export_mode"), self.export_mode_combo.currentText())

    def _append_log(self, message):
        self.log_edit.append(message)

    def _validate_program(self):
        program = self.python_edit.text().strip()
        if not program:
            QMessageBox.warning(self, "Missing Python", "Set the backend Python executable first.")
            return None
        if not os.path.exists(program):
            QMessageBox.warning(self, "Missing Python", f"Python executable not found:\n{program}")
            return None
        return program

    def _start_process(self, action_name, workdir, args, on_success=None):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Process Running", "Another backend task is already running.")
            return

        program = self._validate_program()
        if program is None:
            return

        if not os.path.isdir(workdir):
            QMessageBox.warning(self, "Missing Directory", f"Working directory does not exist:\n{workdir}")
            return

        self._save_settings()
        self.pending_action = {"name": action_name, "on_success": on_success}
        self.process = QProcess(self)
        self.process.setWorkingDirectory(workdir)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.started.connect(lambda: self._append_log(f"$ {program} {' '.join(args)}"))
        self.stop_button.setEnabled(True)
        self.process.start(program, args)

    def _read_stdout(self):
        if self.process:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            if text:
                self._append_log(text)

    def _read_stderr(self):
        if self.process:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            if text:
                self._append_log(text)

    def _process_finished(self, exit_code, exit_status):
        self.stop_button.setEnabled(False)
        action = self.pending_action or {}
        if exit_code == 0:
            self._append_log(f"[ok] {action.get('name', 'Task')} finished.")
            callback = action.get("on_success")
            if callback:
                callback()
        else:
            self._append_log(f"[failed] {action.get('name', 'Task')} exited with code {exit_code}.")
        self.pending_action = None
        self.process = None

    def stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self._append_log("[stopped] Process terminated by user.")

    def run_stage1_grouped(self):
        args = [
            os.path.join(self.backend_dir_edit.text().strip(), "stage1_bologna_real_mcmc.py"),
            "--insar-path",
            self.stage1_insar_edit.text().strip(),
            "--w3ra-path",
            self.stage1_w3ra_edit.text().strip(),
            "--output-dir",
            self.stage1_output_edit.text().strip(),
            "--n-iter",
            str(self.stage1_iter_spin.value()),
            "--burn",
            str(self.stage1_burn_spin.value()),
            "--mode",
            "grouped",
        ]
        self._start_process("Stage 1 grouped", self.backend_dir_edit.text().strip(), args, on_success=self._after_stage1)

    def _after_stage1(self):
        result_path = os.path.join(self.stage1_output_edit.text().strip(), "stage1_bologna_real_results.npz")
        self.grouped_results_edit.setText(result_path)
        self.stage2_input_edit.setText(result_path)
        self.export_npz_edit.setText(result_path)
        self.export_mode_combo.setCurrentText("grouped")

    def run_layered_export(self):
        args = [
            os.path.join(self.backend_dir_edit.text().strip(), "export_layered_inference_from_grouped.py"),
            "--grouped-results-path",
            self.grouped_results_edit.text().strip(),
            "--output-dir",
            self.layered_output_edit.text().strip(),
        ]
        self._start_process("Grouped to layered export", self.backend_dir_edit.text().strip(), args, on_success=self._after_layered_export)

    def _after_layered_export(self):
        layered_path = os.path.join(self.layered_output_edit.text().strip(), "layered_inference_from_grouped.npz")
        self.export_npz_edit.setText(layered_path)
        self.export_mode_combo.setCurrentText("layered")
        self.export_nc_edit.setText("/tmp/layered_inference_for_qgis.nc")

    def run_stage2(self):
        args = [
            os.path.join(self.backend_dir_edit.text().strip(), "stage2_bologna_real_residual.py"),
            "--stage1-results-path",
            self.stage2_input_edit.text().strip(),
            "--output-dir",
            self.stage2_output_edit.text().strip(),
            "--device",
            self.stage2_device_combo.currentText(),
            "--max-epochs",
            str(self.stage2_epochs_spin.value()),
        ]
        self._start_process("Stage 2 residual", self.backend_dir_edit.text().strip(), args, on_success=self._after_stage2)

    def _after_stage2(self):
        stage2_results = os.path.join(self.stage2_output_edit.text().strip(), "stage2_bologna_real_results.npz")
        self.export_npz_edit.setText(stage2_results)
        self.export_mode_combo.setCurrentText("grouped")
        self.export_nc_edit.setText("/tmp/stage2_grouped_inference_for_qgis.nc")

    def run_netcdf_export(self):
        args = [
            os.path.join(self.plugin_dir, "tools", "inversion_npz_to_netcdf.py"),
            "--input",
            self.export_npz_edit.text().strip(),
            "--output",
            self.export_nc_edit.text().strip(),
            "--mode",
            self.export_mode_combo.currentText(),
        ]
        time_origin = self.time_origin_edit.text().strip()
        if time_origin:
            args.extend(["--time-origin", time_origin])
        self._start_process("NPZ to NetCDF export", self.plugin_dir, args, on_success=self._after_netcdf_export)

    def _after_netcdf_export(self):
        path = self.export_nc_edit.text().strip()
        if os.path.exists(path):
            QgsProject.instance().writeEntry("W3RAExplorer", "NetCDF_Path", path)
            self._append_log(f"[ready] NetCDF available at {path}")

    def prefill_grouped_export(self):
        self.export_npz_edit.setText(self.grouped_results_edit.text().strip())
        self.export_mode_combo.setCurrentText("grouped")
        self.export_nc_edit.setText("/tmp/grouped_inference_for_qgis.nc")

    def prefill_layered_export(self):
        self.export_npz_edit.setText(os.path.join(self.layered_output_edit.text().strip(), "layered_inference_from_grouped.npz"))
        self.export_mode_combo.setCurrentText("layered")
        self.export_nc_edit.setText("/tmp/layered_inference_for_qgis.nc")

    def load_current_netcdf(self):
        path = self.export_nc_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Missing NetCDF", "Export a NetCDF file first, or choose an existing file.")
            return
        self.load_netcdf_callback(path)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)
