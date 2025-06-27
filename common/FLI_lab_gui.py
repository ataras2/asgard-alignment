import numpy as np
import time
import datetime
import sys
from pathlib import Path
import os
from astropy.io import fits
import json
import numpy as np
import matplotlib.pyplot as plt


# sys.path.insert(1, '/opt/FirstLightImaging/FliSdk/Python/demo/')
# import FliSdk_V2
# import FliCredOne
# import FliCredTwo
# import FliCredThree

# FLI_Cameras  import must be above PyQt 5 otherwise c library conflicts.
#from asgard_alignment import FLI_Cameras
from asgard_alignment import FLI_Cameras as FLI_Cameras
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QTextEdit, QFileDialog, QSlider
# from PyQt5.QtCore import QTimer, Qt
# from PyQt5.QtGui import QPixmap, QImage

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
import pyqtgraph as pg

# from astropy.io import fits


class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Initialize FLI camera
        self.camera = FLI_Cameras.fli() #0, roi=[None, None, None, None])
        config_file_name = os.path.join(
            self.camera.config_file_path, "default_cred1_config.json"
        )
        #self.camera.configure_camera(config_file_name)
        self.camera.start_camera()

        # Main layout as a 2x2 grid
        main_layout = QtWidgets.QGridLayout(self)

        # Text input fields for min and max cut values
        self.min_cut_input = QtWidgets.QLineEdit()
        self.min_cut_input.setPlaceholderText("Min Cut")
        self.min_cut_input.returnPressed.connect(self.update_image_cuts)

        self.max_cut_input = QtWidgets.QLineEdit()
        self.max_cut_input.setPlaceholderText("Max Cut")
        self.max_cut_input.returnPressed.connect(self.update_image_cuts)

        # Button for auto cuts
        self.auto_cut_button = QtWidgets.QPushButton("Auto Cuts")
        self.auto_cut_button.clicked.connect(self.auto_cut)

        # Checkbox for image reduction
        self.reduce_image_checkbox = QtWidgets.QCheckBox("Reduce Image")
        self.reduce_image_checkbox.setChecked(True)  # Default to checked

        # Control layout for cut settings
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(QtWidgets.QLabel("Min Cut"))
        control_layout.addWidget(self.min_cut_input)
        control_layout.addWidget(QtWidgets.QLabel("Max Cut"))
        control_layout.addWidget(self.max_cut_input)
        control_layout.addWidget(self.auto_cut_button)
        control_layout.addWidget(self.reduce_image_checkbox)

        # Camera frame placeholder (G[0,0])
        self.camera_view = pg.PlotWidget()
        self.camera_image = pg.ImageItem()
        self.camera_view.addItem(self.camera_image)

        self.camera_image.setLevels(
            (0, 255)
        )  # Set initial display range to 0-255 or any typical default range
        self.min_cut_input.setText("0")
        self.max_cut_input.setText("255")

        self.camera_view.setAspectLocked(True)
        main_layout.addWidget(self.camera_view, 0, 0)

        # Pixel value display label
        self.pixel_value_label = QtWidgets.QLabel("Pixel Value: N/A")
        main_layout.addWidget(self.pixel_value_label, 1, 0)

        # Connect mouse movement over image to show pixel values
        self.proxy = pg.SignalProxy(
            self.camera_view.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.show_pixel_value,
        )

        # DM images in a vertical column (G[0,1])
        dm_layout = QtWidgets.QVBoxLayout()
        self.dm_images = []
        for i in range(4):
            dm_label = QtWidgets.QLabel(f"DM {i + 1}")
            dm_view = pg.PlotWidget()
            dm_image = pg.ImageItem()
            dm_view.addItem(dm_image)
            dm_view.setAspectLocked(True)
            dm_layout.addWidget(dm_label)
            dm_layout.addWidget(dm_view)
            self.dm_images.append(dm_image)
        main_layout.addLayout(dm_layout, 0, 1)

        # Command line prompt area (G[1,0])
        command_layout = QtWidgets.QVBoxLayout()
        self.prompt_history = QtWidgets.QTextEdit()
        self.prompt_history.setReadOnly(True)
        self.prompt_input = QtWidgets.QLineEdit()
        self.prompt_input.returnPressed.connect(self.handle_command)
        command_layout.addWidget(self.prompt_history)
        command_layout.addWidget(self.prompt_input)
        main_layout.addLayout(command_layout, 1, 0)

        # 4x2 button grid (G[1,1])
        button_layout = QtWidgets.QGridLayout()
        button_functions = [
            ("Start Camera", self.camera.start_camera),
            ("Stop Camera", self.camera.stop_camera),
            ("Save Config", self.save_camera_config),
            ("Load Config", self.load_camera_config),
            ("Save Images", self.save_images),
            ("Build Dark", self.build_dark),
            ("Get Bad Pixels", self.get_bad_pixels),
        ]
        for i, (text, func) in enumerate(button_functions):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(func)
            button_layout.addWidget(button, i // 2, i % 2)
        main_layout.addLayout(button_layout, 1, 1)

        # Integrate control layout (with cut settings) into the main layout
        main_layout.addLayout(
            control_layout, 2, 0, 1, 2
        )  # Positioned below the main camera view and button grid

        # Adjust row and column stretch to set proportions
        main_layout.setRowStretch(0, 3)
        main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)

        # Status LED
        self.status_led = QtWidgets.QLabel()
        self.status_led.setFixedSize(20, 20)
        self.update_led(False)
        main_layout.addWidget(
            self.status_led, 1, 1, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight
        )

        # Timer for camera updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_image)
        self.timer.start(100)

        # Command history and completer
        self.command_history = []
        self.history_index = -1
        self.completer = QtWidgets.QCompleter(self)
        self.prompt_input.setCompleter(self.completer)
        self.update_completer()

    def auto_cut(self):
        """Automatically adjust the cut range for optimal viewing based on percentiles."""
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        img = self.camera.get_image(
            apply_manual_reduction=apply_manual_reduction, which_index=-1
        )

        if img is not None:
            min_cut = np.percentile(img, 1)
            max_cut = np.percentile(img, 99)

            # Update text inputs with calculated cuts
            self.min_cut_input.setText(str(int(min_cut)))
            self.max_cut_input.setText(str(int(max_cut)))

            # Apply the new cut range
            self.update_image_cuts()
            self.prompt_history.append(
                "Auto cut applied based on the 1st and 99th percentiles of the image."
            )
        else:
            self.prompt_history.append("No image data available for auto cut.")

    def update_image_cuts(self):
        """Update the min and max cuts for the displayed image based on text input values."""
        try:
            min_cut = int(self.min_cut_input.text())
            max_cut = int(self.max_cut_input.text())
            self.camera_image.setLevels((min_cut, max_cut))
        except ValueError:
            self.prompt_history.append("Invalid input for min or max cut.")

    def update_camera_image(self):
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        img = self.camera.get_image(
            apply_manual_reduction=apply_manual_reduction, which_index=-1
        )
        if img is not None:
            self.camera_image.setImage(img)
            self.update_image_cuts()  # Apply current cut values to image

    def update_led(self, running):
        self.status_led.setStyleSheet(
            "background-color: green;" if running else "background-color: red;"
        )

    def show_pixel_value(self, event):
        """Display pixel value at mouse hover position."""
        pos = event[0]  # event[0] holds the QtGui.QGraphicsSceneMouseEvent
        mouse_point = self.camera_view.getViewBox().mapSceneToView(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())

        # Check if mouse is within the image bounds
        if 0 <= x < self.camera_image.width() and 0 <= y < self.camera_image.height():
            img_data = (
                self.camera_image.image
            )  # Assuming the image data is stored in this attribute
            pixel_value = img_data[y, x]  # Note: y comes first for row-major order
            self.pixel_value_label.setText(f"Pixel Value: {pixel_value}")
        else:
            self.pixel_value_label.setText("Pixel Value: N/A")

    def handle_command(self):
        command = self.prompt_input.text()
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        self.prompt_history.append(f"> {command}")

        try:
            if command.lower() == "autocuts":
                self.auto_cut()
                self.prompt_history.append(
                    "Image cut ranges updated based on current image."
                )
            else:
                response = self.camera.send_fli_cmd(command)
                self.prompt_history.append(f"Command executed. Reply:\n{response}")
        except Exception as e:
            self.prompt_history.append(f"Error: {str(e)}")

    def update_completer(self):
        model = QtGui.QStandardItemModel(self.completer)
        for cmd in FLI_Cameras.cred1_command_dict.keys():
            item = QtGui.QStandardItem(cmd)
            model.appendRow(item)
        self.completer.setModel(model)

    def reinitialize_camera(self, config_file_path):
        self.camera.stop_camera()
        self.camera.exit_camera()
        self.camera = FLI_Cameras.fli()#0, roi=[None, None, None, None])
        self.camera.configure_camera(config_file_path)
        self.camera.start_camera()
        self.prompt_history.append(
            f"Camera reinitialized with config: {config_file_path}"
        )

    def save_camera_config(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Config", "", "Config Files (*.json)"
        )
        if file_name:
            config_file = self.camera.get_camera_config()
            with open(file_name, "w") as f:
                json.dump(config_file, f)

    def load_camera_config(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Config", "", "Config Files (*.json)"
        )
        if file_name:
            self.reinitialize_camera(file_name)

    def save_images(self):
        self.timer.stop()
        self.camera.start_camera()  # gui freezes if camera is off
        apply_manual_reduction = self.reduce_image_checkbox.isChecked()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Images", "", "Image Files (*.fits)"
        )
        self.camera.save_fits(
            fname=file_name,
            number_of_frames=100,
            apply_manual_reduction=apply_manual_reduction,
        )
        self.timer.start(100)

    def build_dark(self):
        self.timer.stop()
        self.camera.start_camera()  # gui freezes if camera is off
        self.camera.build_manual_dark(no_frames=100)
        self.timer.start(100)

    def get_bad_pixels(self):
        self.timer.stop()
        self.camera.start_camera()  # gui freezes if camera is off
        bad_pixels = self.camera.get_bad_pixel_indicies(
            no_frames=100, std_threshold=100, flatten=False
        )
        self.camera.build_bad_pixel_mask(bad_pixels=bad_pixels, set_bad_pixels_to=0)
        self.timer.start(100)

    def closeEvent(self, event):
        self.camera.send_fli_cmd("set gain 1")
        self.camera.stop_camera()
        self.camera.exit_camera()
        event.accept()

    def __del__(self):
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.send_fli_cmd("set gain 1")
            self.camera.stop_camera()
            self.camera.exit_camera()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AOControlApp()
    window.setWindowTitle("AO Control GUI")
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
