#!/usr/bin/env python3
"""
Baldr Slow GUI — capture, inspect "lucky" frames, and apply cautious LO/HO iterate steps.

Assumptions
-----------
- A per-beam TOML exists and contains: pupil masks (mask, secondary, exterior),
  I2A, I0, N0, I2M_LO, I2M_HO, M2C_LO, M2C_HO under
  [beam{beam_id}.{phasemask}.ctrl_model]. Shapes may vary (nAct=140 or 144).
- A shared-memory camera stream is available at /dev/shm/baldr{beam_id}.im.shm.
- DM control is via asgard_alignment.DM_shm_ctrl.dmclass with .send_data(cmd: np.ndarray).

What it does
------------
- Load config → build state (masks, mappings) and reference I0.
- Capture N frames to a local buffer; compute an exterior-mask metric timeseries.
- Select "lucky" (>q_lucky) and "unlucky" (<q_unlucky) subsets; show means.
- Recompute I0_ref (image-domain, normalized) and I0_dm_ref (DM-domain) from lucky stack.
- Apply a *single* LO and/or HO iterate step on demand: Δcmd = M2C_*(@) I2M_*(@)(I2A @ mean_img_norm − I0_dm_ref).

Notes
-----
- Shapes/orientations of I2A, I2M_*, M2C_* differ across builds. Helper functions
  try to do the right thing; verify in the log panel on first run.
- This app is intentionally slow & step-based.
"""
import argparse
import datetime
import os
import sys
import time
import toml
import numpy as np

from typing import Tuple, Optional

# Qt / plotting
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# Optional: FITS saving
try:
    from astropy.io import fits
    HAVE_FITS = True
except Exception:
    HAVE_FITS = False

# Baldr/Asgard libs (runtime environment must provide these)
try:
    from xaosim.shmlib import shm
except Exception as e:
    shm = None
    _SHM_IMPORT_ERR = e

try:
    from asgard_alignment.DM_shm_ctrl import dmclass
except Exception as e:
    dmclass = None
    _DM_IMPORT_ERR = e


# --------------------------
# Helper math / shape guards
# --------------------------

def to_bool_mask(arr, shape: Tuple[int, int]):
    a = np.asarray(arr)
    if a.dtype != bool:
        # tolerate 0/1 numeric masks
        a = a.astype(bool)
    if a.ndim == 1 and a.size == (shape[0] * shape[1]):
        a = a.reshape(shape)
    if a.shape != shape:
        raise ValueError(f"Mask shape mismatch. Expected {shape}, got {a.shape}.")
    return a


def image_to_dm(I2A: np.ndarray, img_flat: np.ndarray) -> np.ndarray:
    """Map image (flattened) → DM space using I2A. Tries both orientations.
    Returns vector length nAct.
    """
    I2A = np.asarray(I2A)
    img_flat = np.asarray(img_flat).reshape(-1)
    # Try (nAct x Npix) @ (Npix) -> (nAct)
    if I2A.ndim == 2 and I2A.shape[1] == img_flat.size:
        return I2A @ img_flat
    # Try (Npix x nAct).T @ (Npix) -> (nAct)
    if I2A.ndim == 2 and I2A.shape[0] == img_flat.size:
        return I2A.T @ img_flat
    raise ValueError(f"I2A has incompatible shape {I2A.shape} for image of {img_flat.size} pixels.")


def modes_from_signal(I2M: np.ndarray, sig_dm: np.ndarray) -> np.ndarray:
    """Map DM-space signal → modal coefficients. Tries common orientations."""
    I2M = np.asarray(I2M)
    sig_dm = np.asarray(sig_dm).reshape(-1)
    # (nModes x nAct) @ (nAct) -> (nModes)
    if I2M.ndim == 2 and I2M.shape[1] == sig_dm.size:
        return I2M @ sig_dm
    # (nAct x nModes).T @ (nAct) -> (nModes)
    if I2M.ndim == 2 and I2M.shape[0] == sig_dm.size:
        return I2M.T @ sig_dm
    raise ValueError(f"I2M has incompatible shape {I2M.shape} for nAct={sig_dm.size}.")


def modes_to_cmd(M2C: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Map modal vector → DM commands. Handles (nAct x nModes) or (nModes x nAct).
    Returns (nAct,).
    """
    M2C = np.asarray(M2C)
    modes = np.asarray(modes).reshape(-1)
    # (nAct x nModes) @ (nModes) -> (nAct)
    if M2C.ndim == 2 and M2C.shape[1] == modes.size:
        return M2C @ modes
    # (nModes x nAct).T @ (nModes) -> (nAct)
    if M2C.ndim == 2 and M2C.shape[0] == modes.size:
        return M2C.T @ modes
    raise ValueError(f"M2C has incompatible shape {M2C.shape} for nModes={modes.size}.")


# --------------------------
# Config loader
# --------------------------

def load_config(toml_path: str, beam_id: int, phasemask: str, img_shape_hint: Optional[Tuple[int,int]] = None):
    d = toml.load(toml_path)
    b = d.get(f"beam{beam_id}", {})
    cm = (b.get(phasemask, {}) or {}).get("ctrl_model", {})

    # Masks
    pupil_mask = b.get("pupil_mask", {}).get("mask", None)
    secondary_mask = b.get("pupil_mask", {}).get("secondary", None)
    exterior_mask = b.get("pupil_mask", {}).get("exterior", None)

    # Mappings & refs
    I2A = np.array(b.get("I2A"))
    I0 = np.array(cm.get("I0")) if cm.get("I0") is not None else None
    N0 = np.array(cm.get("N0")) if cm.get("N0") is not None else None

    I2M_LO = np.array(cm.get("I2M_LO")) if cm.get("I2M_LO") is not None else None
    I2M_HO = np.array(cm.get("I2M_HO")) if cm.get("I2M_HO") is not None else None

    # IMPORTANT: Use the correct keys for M2C
    M2C_LO = np.array(cm.get("M2C_LO")) if cm.get("M2C_LO") is not None else None
    M2C_HO = np.array(cm.get("M2C_HO")) if cm.get("M2C_HO") is not None else None

    inner_pupil_filt = np.array(cm.get("inner_pupil_filt")) if cm.get("inner_pupil_filt") is not None else None

    # Shape checks later when first image arrives (when img_shape_hint is None).
    return {
        "pupil_mask": pupil_mask,
        "secondary_mask": secondary_mask,
        "exterior_mask": exterior_mask,
        "I2A": I2A,
        "I0": I0,
        "N0": N0,
        "I2M_LO": I2M_LO,
        "I2M_HO": I2M_HO,
        "M2C_LO": M2C_LO,
        "M2C_HO": M2C_HO,
        "inner_pupil_filt": inner_pupil_filt,
    }


# --------------------------
# Worker: capture a batch
# --------------------------
class CaptureWorker(QtCore.QThread):
    batch_ready = QtCore.pyqtSignal(object)  # emits np.ndarray of shape (N,H,W)
    log = QtCore.pyqtSignal(str)

    def __init__(self, shm_path: str, N: int, dt_s: float, timeout_s: float):
        super().__init__()
        self.shm_path = shm_path
        self.N = int(N)
        self.dt_s = float(dt_s)
        self.timeout_s = float(timeout_s)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        if shm is None:
            self.log.emit(f"ERROR: xaosim.shmlib not available: {_SHM_IMPORT_ERR}")
            return
        try:
            c = shm(self.shm_path, nosem=False)
        except Exception as e:
            self.log.emit(f"ERROR: cannot open SHM {self.shm_path}: {e}")
            return

        imgs = []
        t0 = time.time()
        while (len(imgs) < self.N) and (not self._stop):
            try:
                frame = c.get_data()
            except Exception as e:
                self.log.emit(f"ERROR: get_data failed: {e}")
                break
            if frame is None:
                time.sleep(0.001)
                continue
            imgs.append(np.array(frame, copy=True))
            if self.dt_s > 0:
                time.sleep(self.dt_s)
            if (time.time() - t0) > self.timeout_s:
                self.log.emit("WARN: capture timed out before reaching N frames")
                break
        if len(imgs) > 0:
            stack = np.stack(imgs, axis=0)
            self.batch_ready.emit(stack)
        else:
            self.log.emit("WARN: no frames captured")


# --------------------------
# Main GUI
# --------------------------
class BaldrSlowGUI(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(f"Baldr Slow GUI — beam {args.beam_id} / {args.phasemask}")

        # State
        self.config = None
        self.img_shape = None
        self.I0_ref_img = None   # image-domain normalized ref (H,W)
        self.I0_dm_ref = None    # DM-domain ref (nAct,)
        self.dm = None
        self.cmd = None
        self.last_stack = None   # (N,H,W)
        self.ext_signal = None

        # Build UI
        self._build_ui()
        self._wire_signals()

        # DM
        if dmclass is None:
            self.log(f"WARN: DM class not available: {_DM_IMPORT_ERR}")
        else:
            try:
                self.dm = dmclass(beam_id=args.beam_id, main_chn=3)
                self.log("DM connected (channel 3 for poke / iterate)")
            except Exception as e:
                self.log(f"ERROR: DM init failed: {e}")

        self.reload_config()

    # ---------- UI ----------
    def _build_ui(self):
        pg.setConfigOptions(antialias=True)
        layout = QtWidgets.QGridLayout(self)

        # Controls panel
        ctrl = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrl)

        self.le_toml = QtWidgets.QLineEdit(self.args.toml_file)
        self.le_shm = QtWidgets.QLineEdit(f"/dev/shm/baldr{self.args.beam_id}.im.shm")
        self.sb_N = QtWidgets.QSpinBox(); self.sb_N.setRange(10, 200000); self.sb_N.setValue(self.args.captures)
        self.dsb_dt = QtWidgets.QDoubleSpinBox(); self.dsb_dt.setRange(0.0, 1.0); self.dsb_dt.setDecimals(4); self.dsb_dt.setSingleStep(0.001); self.dsb_dt.setValue(self.args.dt_ms/1000.0)
        self.dsb_timeout = QtWidgets.QDoubleSpinBox(); self.dsb_timeout.setRange(0.1, 3600.0); self.dsb_timeout.setValue(10.0)
        self.dsb_lucky = QtWidgets.QDoubleSpinBox(); self.dsb_lucky.setRange(0.5, 1.0); self.dsb_lucky.setSingleStep(0.01); self.dsb_lucky.setValue(self.args.lucky)
        self.dsb_unlucky = QtWidgets.QDoubleSpinBox(); self.dsb_unlucky.setRange(0.0, 0.5); self.dsb_unlucky.setSingleStep(0.01); self.dsb_unlucky.setValue(self.args.unlucky)
        self.dsb_amp = QtWidgets.QDoubleSpinBox(); self.dsb_amp.setRange(0.0, 10.0); self.dsb_amp.setValue(self.args.amplitude)

        self.cb_norm = QtWidgets.QCheckBox("Normalize mean image before mapping (divide by sum)")
        self.cb_norm.setChecked(True)

        self.btn_reload = QtWidgets.QPushButton("Reload config")
        self.btn_capture = QtWidgets.QPushButton("Capture N frames")
        self.btn_recompute_I0 = QtWidgets.QPushButton("Recompute I0 from lucky")
        self.cb_use_new_I0 = QtWidgets.QCheckBox("Use new I0 for iterate")
        self.btn_iterate_LO = QtWidgets.QPushButton("Step iterate LO")
        self.btn_iterate_HO = QtWidgets.QPushButton("Step iterate HO")
        self.btn_reset_dm = QtWidgets.QPushButton("Reset DM (zeros)")
        self.btn_save = QtWidgets.QPushButton("Save FITS (means & ext TS)")

        form.addRow("TOML path:", self.le_toml)
        form.addRow("SHM path:", self.le_shm)
        form.addRow("N frames:", self.sb_N)
        form.addRow("Inter-frame sleep [s]:", self.dsb_dt)
        form.addRow("Timeout [s]:", self.dsb_timeout)
        form.addRow("Lucky quantile:", self.dsb_lucky)
        form.addRow("Unlucky quantile:", self.dsb_unlucky)
        form.addRow("Iterate amplitude:", self.dsb_amp)
        form.addRow(self.cb_norm)
        form.addRow(self.btn_reload)
        form.addRow(self.btn_capture)
        form.addRow(self.btn_recompute_I0)
        form.addRow(self.cb_use_new_I0)
        form.addRow(self.btn_iterate_LO)
        form.addRow(self.btn_iterate_HO)
        form.addRow(self.btn_reset_dm)
        form.addRow(self.btn_save)

        # Plots/images
        self.img_lucky = pg.ImageItem(); self.lut_lucky = pg.HistogramLUTWidget(); self.lut_lucky.setImageItem(self.img_lucky)
        self.img_unlucky = pg.ImageItem(); self.lut_unlucky = pg.HistogramLUTWidget(); self.lut_unlucky.setImageItem(self.img_unlucky)
        self.img_all = pg.ImageItem(); self.lut_all = pg.HistogramLUTWidget(); self.lut_all.setImageItem(self.img_all)

        view_lucky = pg.PlotWidget(); view_lucky.addItem(self.img_lucky); view_lucky.setTitle("Mean lucky")
        view_unlucky = pg.PlotWidget(); view_unlucky.addItem(self.img_unlucky); view_unlucky.setTitle("Mean unlucky")
        view_all = pg.PlotWidget(); view_all.addItem(self.img_all); view_all.setTitle("Mean all")

        self.plot_ext = pg.PlotWidget(title="Exterior-signal timeseries")
        self.plot_ext_h = pg.PlotWidget(title="Exterior-signal histogram")

        # Logs
        self.te_log = QtWidgets.QPlainTextEdit(); self.te_log.setReadOnly(True); self.te_log.setMaximumBlockCount(1000)

        # Layout grid
        layout.addWidget(ctrl, 0, 0, 3, 1)

        # Top row: images + LUTs
        imgs_layout = QtWidgets.QGridLayout()
        w_imgs = QtWidgets.QWidget(); w_imgs.setLayout(imgs_layout)
        imgs_layout.addWidget(view_lucky, 0, 0)
        imgs_layout.addWidget(self.lut_lucky, 0, 1)
        imgs_layout.addWidget(view_unlucky, 1, 0)
        imgs_layout.addWidget(self.lut_unlucky, 1, 1)
        imgs_layout.addWidget(view_all, 2, 0)
        imgs_layout.addWidget(self.lut_all, 2, 1)

        layout.addWidget(w_imgs, 0, 1, 2, 2)

        # Bottom row: plots + log
        layout.addWidget(self.plot_ext, 2, 1)
        layout.addWidget(self.plot_ext_h, 2, 2)
        layout.addWidget(self.te_log, 3, 0, 1, 3)

    def _wire_signals(self):
        self.btn_reload.clicked.connect(self.reload_config)
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_recompute_I0.clicked.connect(self.on_recompute_I0)
        self.btn_iterate_LO.clicked.connect(lambda: self.on_iterate(which="LO"))
        self.btn_iterate_HO.clicked.connect(lambda: self.on_iterate(which="HO"))
        self.btn_reset_dm.clicked.connect(self.on_reset_dm)
        self.btn_save.clicked.connect(self.on_save)

    # ---------- Logging ----------
    def log(self, s: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.te_log.appendPlainText(f"[{ts}] {s}")
        # also print to stdout
        print(s)

    # ---------- Config & DM ----------
    def reload_config(self):
        path = self.le_toml.text().strip()
        try:
            self.config = load_config(path, self.args.beam_id, self.args.phasemask)
        except Exception as e:
            self.log(f"ERROR: load_config failed: {e}")
            return
        # Reset references; will be finalized after first capture
        self.I0_ref_img = None
        self.I0_dm_ref = None
        self.cmd = None
        self.log("Config reloaded. Capture a batch to initialize references.")

    def ensure_dm_ready(self, nAct_guess: Optional[int] = None):
        if self.dm is None:
            self.log("WARN: DM not initialized; iterate/reset disabled.")
            return False
        if self.cmd is None:
            nAct = nAct_guess
            # Try infer from M2C_LO / M2C_HO / I2A
            for key in ("M2C_LO", "M2C_HO"):
                M2C = self.config.get(key)
                if M2C is not None and np.ndim(M2C) == 2:
                    nAct = M2C.shape[0] if M2C.shape[0] in (140, 144) else M2C.shape[1]
                    break
            if nAct is None:
                I2A = self.config.get("I2A")
                if I2A is not None:
                    if I2A.shape[0] in (140, 144):
                        nAct = I2A.shape[0]
                    elif I2A.shape[1] in (140, 144):
                        nAct = I2A.shape[1]
            if nAct is None:
                nAct = 144  # fallback
            self.cmd = np.zeros(nAct, dtype=float)
            self.log(f"Initialized DM command vector with nAct={nAct}.")
        return True

    # ---------- Capture & compute ----------
    def on_capture(self):
        shm_path = self.le_shm.text().strip()
        N = int(self.sb_N.value())
        dt_s = float(self.dsb_dt.value())
        timeout_s = float(self.dsb_timeout.value())

        self.log(f"Starting capture: N={N}, sleep={dt_s}s, timeout={timeout_s}s…")
        self.btn_capture.setEnabled(False)

        self.worker = CaptureWorker(shm_path, N, dt_s, timeout_s)
        self.worker.batch_ready.connect(self.on_batch)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(lambda: self.btn_capture.setEnabled(True))
        self.worker.start()

    def on_batch(self, stack: np.ndarray):
        self.last_stack = stack
        H, W = stack.shape[1:]
        if self.img_shape is None:
            self.img_shape = (H, W)
            self.log(f"Image shape set to {self.img_shape}.")
            # Validate masks now
            try:
                self.pupil_mask = to_bool_mask(self.config["pupil_mask"], self.img_shape)
                self.exterior_mask = to_bool_mask(self.config["exterior_mask"], self.img_shape)
            except Exception as e:
                self.log(f"ERROR: mask validation failed: {e}")
                return

        # Exterior-signal metric
        ext = stack[:, self.exterior_mask]
        self.ext_signal = ext.mean(axis=1)

        # Quantile thresholds
        qL = float(self.dsb_lucky.value())
        qU = float(self.dsb_unlucky.value())
        lucky_cut = np.quantile(self.ext_signal, qL)
        unlucky_cut = np.quantile(self.ext_signal, qU)

        idx_lucky = self.ext_signal > lucky_cut
        idx_unlucky = self.ext_signal < unlucky_cut

        if not np.any(idx_lucky):
            self.log("WARN: no frames above lucky threshold; lowering threshold suggested.")
        if not np.any(idx_unlucky):
            self.log("WARN: no frames below unlucky threshold; raising threshold suggested.")

        mean_all = stack.mean(axis=0)
        mean_lucky = stack[idx_lucky].mean(axis=0) if np.any(idx_lucky) else mean_all
        mean_unlucky = stack[idx_unlucky].mean(axis=0) if np.any(idx_unlucky) else mean_all

        # Update images
        for img_item, arr in ((self.img_lucky, mean_lucky), (self.img_unlucky, mean_unlucky), (self.img_all, mean_all)):
            img_item.setImage(arr.T, autoLevels=True)  # transpose for pg's x/y convention

        # Update plots
        self.plot_ext.clear()
        self.plot_ext.plot(self.ext_signal, pen=pg.mkPen(width=1))
        self.plot_ext.addLine(y=lucky_cut, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        self.plot_ext.addLine(y=unlucky_cut, pen=pg.mkPen(style=QtCore.Qt.DashLine))

        self.plot_ext_h.clear()
        y, x = np.histogram(self.ext_signal, bins=50)
        self.plot_ext_h.plot(x[:-1], y, stepMode=True, fillLevel=0, brush=(150, 150, 250, 100))

        # Initialize references on first capture (using "lucky")
        if self.I0_ref_img is None:
            self.I0_ref_img = self._normalize(mean_lucky)
            try:
                self.I0_dm_ref = image_to_dm(self.config["I2A"], self.I0_ref_img.ravel())
            except Exception as e:
                self.log(f"ERROR: I0_dm_ref mapping failed: {e}")
                self.I0_dm_ref = None
            self.log("Initialized I0 references from first lucky stack.")

        self.log(f"Batch done. lucky={idx_lucky.sum()}, unlucky={idx_unlucky.sum()}, total={stack.shape[0]}")

    def _normalize(self, img2d: np.ndarray) -> np.ndarray:
        if not self.cb_norm.isChecked():
            return img2d
        s = float(img2d.sum())
        if s <= 0 or not np.isfinite(s):
            self.log("WARN: normalization skipped (non-positive or non-finite sum)")
            return img2d
        return img2d / s

    # ---------- I0 recompute ----------
    def on_recompute_I0(self):
        if self.last_stack is None:
            self.log("Capture first.")
            return
        ext = self.ext_signal
        qL = float(self.dsb_lucky.value())
        lucky_cut = np.quantile(ext, qL)
        idx_lucky = ext > lucky_cut
        if not np.any(idx_lucky):
            self.log("No lucky frames to recompute I0.")
            return
        mean_lucky = self.last_stack[idx_lucky].mean(axis=0)
        self.I0_ref_img = self._normalize(mean_lucky)
        try:
            self.I0_dm_ref = image_to_dm(self.config["I2A"], self.I0_ref_img.ravel())
            self.log("Recomputed I0 (image & DM domain) from lucky frames.")
        except Exception as e:
            self.log(f"ERROR: I0_dm_ref mapping failed: {e}")
            self.I0_dm_ref = None

    # ---------- Iterate ----------
    def on_iterate(self, which: str):
        if self.last_stack is None:
            self.log("Capture first.")
            return
        if self.I0_dm_ref is None:
            self.log("I0_dm_ref not ready—reload config or recompute I0.")
            return
        if not self.ensure_dm_ready():
            return
        mean_img = self._normalize(self.last_stack.mean(axis=0))
        try:
            sig_dm = image_to_dm(self.config["I2A"], mean_img.ravel()) - self.I0_dm_ref
        except Exception as e:
            self.log(f"ERROR: image_to_dm failed: {e}")
            return

        amp = float(self.dsb_amp.value())

        try:
            if which == "LO":
                I2M = self.config["I2M_LO"]; M2C = self.config["M2C_LO"]
            else:
                I2M = self.config["I2M_HO"]; M2C = self.config["M2C_HO"]
            if I2M is None or M2C is None:
                self.log(f"ERROR: Missing {which} mappings (I2M/M2C)")
                return
            modes = modes_from_signal(I2M, sig_dm)
            dcmd = modes_to_cmd(M2C, modes)
        except Exception as e:
            self.log(f"ERROR: {which} iterate mapping failed: {e}")
            return

        # Length guard
        if dcmd.shape != self.cmd.shape:
            # try to broadcast or trim/pad
            n = min(dcmd.size, self.cmd.size)
            dtmp = np.zeros_like(self.cmd)
            dtmp[:n] = dcmd[:n]
            dcmd = dtmp
            self.log("NOTE: adjusted Δcmd length to match DM.")

        self.cmd = self.cmd + amp * dcmd

        try:
            self.dm.send_data(self.cmd.astype(float))
            self.log(f"Applied one {which} iterate step (amp={amp}). RMS Δcmd={np.sqrt(np.mean((amp*dcmd)**2)):.4g}")
        except Exception as e:
            self.log(f"ERROR: DM send_data failed: {e}")

    # ---------- Reset / Save ----------
    def on_reset_dm(self):
        if not self.ensure_dm_ready():
            return
        self.cmd[:] = 0.0
        try:
            self.dm.send_data(self.cmd)
            self.log("DM reset to zeros.")
        except Exception as e:
            self.log(f"ERROR: DM reset failed: {e}")

    def on_save(self):
        if self.last_stack is None:
            self.log("Nothing to save; capture first.")
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.args.savepath, f"slowgui_{ts}.fits")
        os.makedirs(self.args.savepath, exist_ok=True)

        mean_all = self.last_stack.mean(axis=0)
        if self.ext_signal is None:
            self.ext_signal = np.zeros(self.last_stack.shape[0])

        if not HAVE_FITS:
            self.log("astropy not available; skipping FITS save.")
            return
        try:
            hdu0 = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU(mean_all, name='MEAN_ALL')
            # recompute these to ensure consistency with current thresholds
            ext = self.ext_signal
            qL = float(self.dsb_lucky.value()); qU = float(self.dsb_unlucky.value())
            lucky_cut = np.quantile(ext, qL); unlucky_cut = np.quantile(ext, qU)
            idx_lucky = ext > lucky_cut; idx_unlucky = ext < unlucky_cut
            mean_lucky = self.last_stack[idx_lucky].mean(axis=0) if np.any(idx_lucky) else mean_all
            mean_unlucky = self.last_stack[idx_unlucky].mean(axis=0) if np.any(idx_unlucky) else mean_all
            hdu2 = fits.ImageHDU(mean_lucky, name='MEAN_LUCKY')
            hdu3 = fits.ImageHDU(mean_unlucky, name='MEAN_UNLUCKY')
            hdu4 = fits.BinTableHDU.from_columns([
                fits.Column(name='ext_signal', format='D', array=ext),
            ], name='EXT_TS')
            hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
            hdul.writeto(out, overwrite=True)
            self.log(f"Saved {out}")
        except Exception as e:
            self.log(f"ERROR: FITS save failed: {e}")


# --------------------------
# Entry point
# --------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Baldr Slow GUI")
    default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")
    p.add_argument("--toml_file", type=str, default=default_toml, help="TOML file pattern (replace # with beam_id)")
    p.add_argument("--savepath", type=str, default=os.path.expanduser("~/baldr_slowgui_out"), help="Output dir for FITS saves")
    p.add_argument("--beam_id", type=int, default=1)
    p.add_argument("--phasemask", type=str, default="H3")
    p.add_argument("--captures", type=int, default=3000)
    p.add_argument("--dt_ms", type=float, default=1.0, help="sleep between grabs (ms)")
    p.add_argument("--lucky", type=float, default=0.99)
    p.add_argument("--unlucky", type=float, default=0.10)
    p.add_argument("--amplitude", type=float, default=0.1)
    args = p.parse_args(argv)

    # Resolve # in TOML pattern
    if '#' in args.toml_file:
        args.toml_file = args.toml_file.replace('#', f"{args.beam_id}")
    return args


def main(argv=None):
    args = parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    w = BaldrSlowGUI(args)
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# import numpy as np 
# import zmq
# import time
# import toml
# import os 
# import argparse
# import matplotlib.pyplot as plt
# import argparse
# import subprocess
# import glob

# from astropy.io import fits
# from scipy.signal import TransferFunction, bode
# from types import SimpleNamespace
# import asgard_alignment.controllino as co # for turning on / off source \
# from asgard_alignment.DM_shm_ctrl import dmclass
# import common.DM_basis_functions as dmbases
# import asgard_alignment.controllino as co
# import common.phasemask_centering_tool as pct
# import common.phasescreens as ps 
# import pyBaldr.utilities as util 
# import pyzelda.ztools as ztools
# import datetime
# from xaosim.shmlib import shm
# from asgard_alignment import FLI_Cameras as FLI


# tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# parser = argparse.ArgumentParser(description="Calibrate new baldr reference intensity flat with lucky imaging.")

# arg_default_toml = os.path.join("/usr/local/etc/baldr/", f"baldr_config_#.toml") 

# parser.add_argument(
#     "--toml_file",
#     type=str,
#     default=arg_default_toml,
#     help="TOML file pattern (replace # with beam_id) to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
# )
# # Camera shared memory path
# parser.add_argument(
#     "--savepath",
#     type=str,
#     default="/home/asg/ben_bld_data/15-9-25night6/",
#     help="where to save results . Default: /home/asg/ben_bld_data/15-9-25night6/"
# )

# # Beam ids: provided as a comma-separated string and converted to a list of ints.
# parser.add_argument(
#     "--beam_id",
#     type=int, #lambda s: [int(item) for item in s.split(",")],
#     default=1, # 1, 2, 3, 4],
#     help="which beam to apply. Default: 1"
# )

# parser.add_argument(
#     "--phasemask",
#     type=str,
#     default="H3",
#     help="which phasemask to use "
# )
# parser.add_argument(
#     "--user_input",
#     type=int,
#     default=0,
#     help="do you want to check plots and input if to update the flat? Default 0 (false)"
# )


# args=parser.parse_args()

# beam_id = args.beam_id 
# default_toml =  args.toml_file
# phasemask = args.phasemask
# savepath = args.savepath


# # Baldr RTC server addresses to update live RTC via zmq
# SERVER_ADDR_DICT = {1:"tcp://127.0.0.1:6662",
#                     2:"tcp://127.0.0.1:6663",
#                     3:"tcp://127.0.0.1:6664",
#                     4:"tcp://127.0.0.1:6665"}

# # the defined shared memory address for the baldr subframes
# global_camera_shm = f"/dev/shm/baldr{beam_id}.im.shm" 

# # open the 
# with open(default_toml.replace('#',f'{beam_id}'), "r") as f:

#     config_dict = toml.load(f)

#     baldr_pupils = config_dict.get("baldr_pupils", {})
    
#     crop_mode_offset  = config_dict.get("crop_mode_offset", {})
#     #  read in the current calibrated matricies 
#     pupil_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None) ).astype(bool)   # matrix bool
#     I2A = np.array( config_dict[f'beam{beam_id}']['I2A'] )
#     IM = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("IM", None) ).astype(float)
#     M2C = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("M2C", None) ).astype(float)

#     I2M_LO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
#     I2M_HO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)


#     M2C_LO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I2M_LO", None) ).astype(float)
#     M2C_HO = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I2M_HO", None) ).astype(float)


#     pupil_mask = config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("mask", None)

#     secondary_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("secondary", None) )

#     exterior_mask = np.array(config_dict.get(f"beam{beam_id}", {}).get("pupil_mask", {}).get("exterior", None) )

#     # # define our Tip/Tilt or lower order mode index on zernike DM basis 
#     LO = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("LO", None)

#     # tight (non-edge) pupil filter
#     inside_edge_filt = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("inner_pupil_filt", None) )#.astype(bool)
#     # clear pupil 
#     I0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("I0", None) )
#     N0 = np.array(config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("N0", None) )#.astype(bool)
#     # secondary filter
#     sec = np.array(config_dict.get(f"beam{beam_id}" , {}).get(f"{phasemask}", {}).get("ctrl_model",None).get("secondary", None) )
#     poke_amp = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("poke_amp", None)
#     camera_config = config_dict.get(f"beam{beam_id}", {}).get(f"{phasemask}", {}).get("ctrl_model", None).get("camera_config", None)


# # SETUP 
# ##########

# # shared memory camera object 
# c = shm(global_camera_shm, nosem=False)
# # deformable mirror 
# dm = dmclass( beam_id=beam_id, main_chn=3 ) # we poke on ch3 so we can close TT on chn 2 with rtc when building IM 
# zero_cmd = np.zeros( 144 )

# # how many do we capture 
# Ncaptures = 10000

# # Original I0
# I0_ref_0 = I2A @ I0 # we keep an original copy 
# I0_ref = I0_ref_0.copy()  # by default we use the original first  


# # parameters
# dt = 0.001 #s
# T = 1 #s
# alpha = 0.99
# amplitude = 0.1 # DM units 
# lucky_quantile = 0.99
# unlucky_qunatile = 0.5
# savefile_name = "/fits/file/name.fits"

# # state 
# capture_imgs = True
# iterate = False
# updated_I0 = False


# def reset():
#     global capture_imgs, iterate, updated_I0
#     capture_imgs = False
#     iterate = False
#     updated_I0 = False
#     I0_ref = I0_ref_0.copy()  # reset to original
#     cmd = zero_cmd.copy()
#     dm.send_data( cmd )
#     time.sleep(0.1)
#     print("reset done")

# # PROCESS THE IMAGES 
# ######################
# # go and capture
# img_buffer = []
# while capture_imgs:
#     t0 = time.time()
#     while t1-t0 < T:
#         time.sleep(dt)

#         img_buffer.append( c.get_data() )
#         t1=time.time()
#     print("capture done")    
#     imgs = img_buffer.copy()

    
# # PROCESS THE IMAGES 
# ######################    
# # external signal (light scattered outside of pupil by phasemask which scales with strehl)
# ext_signal = np.array( [np.mean( ii[exterior_mask.astype(bool)] ) for ii in imgs] )

# # cut off quantile for lucky images
# lucky_cutoff = np.quantile( ext_signal , 0.99) #10k samples => 99th perc. keeps 100 samples 

# # also look at the unlucky ones for reference that this is working 
# unlucky_cutoff = np.quantile( ext_signal , 0.10)

# # filter the lucky external pupil signals  
# lucky_ext_signals = np.array(ext_signal)[ext_signal > lucky_cutoff]

# # if np.mean(lucky_ext_signals) < 0.1 * np.mean( I0[exterior_mask] ) : # if the lucky exterior signals are less than 10% of the internal reference exterior signal then we warn the user 
# #     print("WARNING : lucky exterior signals are less than 10% of the internal reference exterior signal. May not be a good reference")

# # filter the images 
# lucky_imgs = np.array(imgs)[ext_signal > lucky_cutoff]

# unlucky_imgs = np.array(imgs)[ext_signal < unlucky_cutoff]


# # show the user the results 
# img_list = [ np.mean( lucky_imgs, axis=0), np.mean( unlucky_imgs, axis=0) ,np.mean( imgs, axis=0) ]
# title_list = [f"mean lucky onsky I0 (>{lucky_quantile} quantile)", f"mean unlucky onsky I0 (<{unlucky_qunatile} quantile)","mean all images"]
# util.nice_heatmap_subplots(im_list = img_list,
#                            title_list = title_list, 
#                            vlims=[[0,np.max(I0)] for _ in range(len(img_list))])    


# # aggregate our lucky and unlucky ones 
# I0_bad = np.mean( unlucky_imgs, axis=0)
# I0_unlucky_ref = I0_bad / np.sum( I0_bad )

# I0_new = np.mean( lucky_imgs, axis=0)
# I0_ref = I0_new / np.sum( I0_new ) 

# I0_dm_ref = I2A @ I0_ref.reshape(-1) 

# if update_I0:
#     I0_ref = I0_dm_ref.copy() 


# if iterate_HO:    
#     sig = np.mean( imgs, axis=0) / np.sum( np.mean( imgs, axis=0) ) - I0_ref

#     # update DM shape to amplitude * 
#     cmd = alpha * (cmd + amplitude * sig)
#     dm.send_data( cmd)
    

# if iterate_LO:    
#     # update DM shape to amplitude * 
#     sig = np.mean( imgs, axis=0) / np.sum( np.mean( imgs, axis=0) ) - I0_dm_ref
#     cmd_err = M2C_LO @ (I2M_LO @ (amplitude * sig) ) 
#     cmd = alpha * (cmd + cmd_err)
#     dm.send_data(amplitude * cmd)
    

# # at all times we show the user 
# # the last image in the buffer (updated not super quick)
# # the reference intensity I0_ref 
# # the last lucky image 
# # the signal 
# # the current DM command


# ####### FOR LATER 
# # ## Bright BOTT2 : 0.04 BOTP2 : -0.377
# # ## faint  BOTT2 : -0.031 BOTP2 : -0.373

# # # plot results 
# # img_list = [I0.reshape(32,32), I0_ref ,I0.reshape(32,32) - I0_ref, I0_unlucky_ref ]
# # title_list = ["original internal I0", "lucky onsky I0","delta", "unlucky onsky I0"]

# # util.nice_heatmap_subplots(im_list = img_list,
# #                            title_list = title_list, 
# #                            vlims=[[0,np.max(I0_ref)] for _ in range(len(img_list))])

# # img_fname = savepath + f"beam{beam_id}_I0_onsky_flat_{tstamp}.jpeg"
# # plt.savefig(img_fname, bbox_inches='tight')
# # if args.user_input:
# #     plt.show() # let the user review it 
# # else:
# #     plt.close()
# # # # On DM
# # # img_list = [util.get_DM_command_in_2D( I2A@I0 ), 
# # #             util.get_DM_command_in_2D(I2A @ (I0_ref.reshape(-1))) ,
# # #             util.get_DM_command_in_2D( I2A @(I0.reshape(32,32) - I0_ref).reshape(-1) )]
# # # title_list = ["original internal I0", "lucky onsky I0","delta"]

# # # util.nice_heatmap_subplots(im_list = img_list,
# # #                            title_list = title_list)

# # if args.user_input:
# #     update_flat = input("review the flat and enter 1 if we should update rtc")
# # else:
# #     update_flat = '1' # update automatically 
# # if update_flat=='1': 
# #     # connect to Baldr RTC socket and update I2A via ZMQ
# #     addr = SERVER_ADDR_DICT[beam_id] # "tcp://127.0.0.1:6662"  # this will change depending on if we are in simulation mode
# #     ctx = zmq.Context.instance()
# #     s = ctx.socket(zmq.REQ)
# #     s.RCVTIMEO = 5000  # ms
# #     s.SNDTIMEO = 5000  # ms
# #     s.connect(addr)

# #     # get the current config file 
# #     s.send_string('list_rtc_fields ""')
# #     rep = s.recv_json()

# #     # update the field (example!)
# #     #s.send_string('set_rtc_field "inj_signal.freq_hz",0.04')
# #     #rep = s.recv_json()


# #     # I0
# #     s.send_string(f'set_rtc_field "reference_pupils.I0",{I0_ref.reshape(-1).tolist()}')
# #     rep = s.recv_json()

# #     print( "sucess?", rep['ok'] )
# #     if not rep['ok'] :
# #         print( '  error: ', rep['error'] )

# #     #I0-dm
# #     s.send_string(f'set_rtc_field "reference_pupils.I0_dm",{I0_dm_ref.reshape(-1).tolist()}')
# #     rep = s.recv_json()

# #     print( "sucess?", rep['ok'] )
# #     if not rep['ok'] :
# #         print( '  error: ', rep['error'] )

# #     # I0-dm_runtime
# #     s.send_string(f'set_rtc_field "I0_dm_runtime",{I0_dm_ref.reshape(-1).tolist()}')
# #     rep = s.recv_json()

# #     print( "sucess?", rep['ok'] )
# #     if not rep['ok'] :
# #         print( '  error: ', rep['error'] )



# #     # close connection
# #     s.setsockopt(zmq.LINGER, 0)   # don't wait on unsent msgs
# #     s.close()  # closes the socket


# # # save the fits of the images 
# # from astropy.io import fits 
# # hdulist = fits.HDUList([])
# # hdu = fits.ImageHDU(imgs)
# # hdulist.append(hdu)
# # frame_fname = savepath + f"beam{beam_id}_I0_onsky_flat_{tstamp}.fits"
# # hdulist.writeto(frame_fname, overwrite=True)

# # print( f"saved the frames at {frame_fname}")

# # print("done")


