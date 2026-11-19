"""
Single-camera data collection over a row/column grid.

Steps through grid positions (row-major), shows a live preview, and saves one image per
position as <out>/<prefix>_r<row>_c<col>.png, plus a log.csv with timestamps.
Positions that already have a file are marked done, so a run can be stopped and resumed.

Keys (in the preview window):
  SPACE        save image at the current position and advance to the next empty one
  arrows/WASD  move the current position without saving (to retake one)
  n            skip to the next empty position
  c            open the camera driver settings dialog (exposure, WB, focus)
  q / ESC      stop (also: close the window, or Ctrl+C in the terminal); rerun to resume

Run with no arguments to pick the settings in a setup window (last used values are remembered).

Examples:
  python grid_capture.py
  python grid_capture.py --rows 5 --cols 8
  python grid_capture.py --rows 3 --cols 3 --cam 1 --out data/run2 --prefix pof
"""
import json
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
from serial.tools.list_ports import comports

FONT_SIZE = 24

# Capture backend per platform
BACKEND = (cv2.CAP_DSHOW if sys.platform == "win32"
           else cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)

FIELDS = [  # (name, label, type)
("rows", "Rows", int), ("cols", "Columns", int), ("cam0", "Left Camera", lambda s: int(s.split(":")[0])), ("cam1", "Right Camera", lambda s: int(s.split(":")[0])),
("out", "Output folder", str), ("mcu", "MCU Port", lambda s: s.split(":")[0].strip()), ("length", "Snapshot Length", int), ("prefix", "File prefix", str),
("width", "Width [px]", int), ("height", "Height [px]", int), ("fps", "FPS", int), ("fourcc", "FourCC", str),
]


def list_cams(probe=False, n=6):
    """[(index, name)] of cameras via cv2_enumerate_cameras (instant). If that fails or finds
    nothing, fall back to opening indices 0..n-1 if `probe`."""
    try:
        found = sorted({(c.index, c.name) for c in enumerate_cameras(BACKEND)})
        if found:
            return found
    except Exception:
        print("Exception while listing cameras")
    found = []
    for i in range(n if probe else 0):
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened() and cap.read()[0]:
            found.append((i, f"camera {i}"))
        cap.release()
    return found


def cam_label(i, name):
    return f"{i}: {name}"


def list_ports():
    """[(device, description)] of serial ports: COMx on Windows, /dev/tty* on Linux/macOS."""
    try:
        # Linux lists dozens of phantom /dev/ttyS* ports with description "n/a"; real USB ones have one
        return [(p.device, p.description) for p in sorted(comports()) if p.description != "n/a"]
    except Exception:
        print("Exception while listing serial ports")
        return []


def pick_cam(cams, idx, last_name=""):
    """Last used camera if it's still connected (matched by name), else the highest index,
    since a built-in laptop camera usually enumerates first."""
    for i, name in cams:
        if last_name and name == last_name:
            return i, name
    return cams[-1] if cams else (idx, last_name)


CONFIG_NAME = "config.json"  # written into every experiment folder
# Settings restored from an experiment's config when resuming. Devices (cameras, MCU port) are
# not restored, since they may have changed since the last session.
RESUME_KEYS = ("rows", "cols", "prefix", "width", "height", "fps", "fourcc", "length")


def is_saved(out, prefix, r, c, rows, cols, sensors=False):
    """True if both camera images of a position exist (and its sensor csv, if `sensors`)."""
    n = Path(fname(prefix, r, c, rows, cols))
    ok = all((out / "grid_data" / n.with_stem(n.stem + s)).exists() for s in ("_l", "_r"))
    return ok and (not sensors or (out / "sensor_data" / n.with_suffix(".csv").name).exists())


def saved_grid(out, cfg, sensors=False):
    """Boolean (rows, cols) array of the positions already saved in experiment folder `out`."""
    return np.array([[is_saved(out, cfg["prefix"], r, c, cfg["rows"], cfg["cols"], sensors)
                      for c in range(cfg["cols"])] for r in range(cfg["rows"])])


def load_config(out):
    """Saved config of experiment folder `out`; raises OSError/ValueError if missing or broken."""
    return json.loads((Path(out) / CONFIG_NAME).read_text())


def find_incomplete(root):
    """[(folder, saved, total)] of experiments under `root` that still have empty positions,
    newest first."""
    found = []
    for d in sorted(Path(root).glob("*/"), reverse=True):
        try:
            cfg = load_config(d)
            done = saved_grid(d, cfg)
        except (OSError, ValueError, KeyError):
            continue
        if not done.all():
            found.append((d, int(done.sum()), done.size))
    return found


def setup_dialog(defaults):
    """CustomTkinter form for the capture settings. Returns a dict, or None if cancelled.
    CustomTkinter scales widgets with the system DPI automatically."""
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.geometry("360x390")
    root.title("Grid capture setup")
    font = ctk.CTkFont(size=FONT_SIZE)
    msg = ctk.CTkFrame(root, fg_color="transparent")
    msg.pack(padx=1, pady=1)
    frm = ctk.CTkFrame(root, fg_color="transparent")
    frm.pack(padx=16, pady=16)
    vars_, result = {}, {}
    W = 260

    warn = ctk.CTkButton(msg, font=font, text="Check README.md before starting!", fg_color="red", hover=False)
    warn.grid(row=0, column=0, pady=4)
    
    for i, (name, label, _) in enumerate(FIELDS):
        ctk.CTkLabel(frm, font=font, text=label, anchor="w").grid(row=i, column=0, sticky="w", pady=4, padx=(0, 10))
        vars_[name] = ctk.StringVar(value=str(defaults[name]))
        if name == "cam0" or name == "cam1":
            box = ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_[name], width=W, values=[""])
            box.grid(row=i, column=1, pady=4)

            def detect(box=box, probe=True, name=name):
                root.configure(cursor="watch"); root.update()
                cams = list_cams(probe)
                root.configure(cursor="")
                if cams:
                    box.configure(values=[cam_label(*c) for c in cams])
                    vars_[name].set(cam_label(*pick_cam(cams, defaults[name], defaults.get(name + "_name", ""))))
                elif probe:
                    messagebox.showwarning("Grid capture", "No cameras found")
            detect(probe=False, name=name)
            ctk.CTkButton(frm, font=font, text="Refresh", width=80, command=detect).grid(row=i, column=2, padx=(8, 0))
        elif name == "mcu":
            box = ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_[name], width=W, values=[""])
            box.grid(row=i, column=1, pady=4)

            def detect_ports(box=box):
                ports = list_ports()
                labels = [f"{dev}: {desc}" for dev, desc in ports]
                box.configure(values=labels)
                current = vars_["mcu"].get().split(":")[0].strip()
                match = next((l for (dev, _), l in zip(ports, labels) if dev == current), None)
                # keep the saved port if it's still present, else the first one found
                if match or labels:
                    vars_["mcu"].set(match or labels[0])
                else:
                    vars_["mcu"].set(current)  # nothing found: leave editable text
            detect_ports()
            ctk.CTkButton(frm, font=font, text="Refresh", width=80, command=detect_ports).grid(row=i, column=2, padx=(8, 0))
        elif name == "fourcc":
            ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_[name], width=W,
                values=["MJPG", "YUY2"]).grid(row=i, column=1, pady=4)
        else:
            ctk.CTkEntry(frm, font=font, textvariable=vars_[name], width=W).grid(row=i, column=1, pady=4)
        if name == "out":
            ctk.CTkButton(frm, font=font, text="Browse", width=80, command=lambda: vars_["out"].set(
                filedialog.askdirectory(initialdir=vars_["out"].get() or ".") or vars_["out"].get())
            ).grid(row=i, column=2, padx=(8, 0))

    # Resume an incomplete experiment: its saved grid settings replace the ones above
    r = len(FIELDS)
    ctk.CTkLabel(frm, font=font, text="Resume", anchor="w").grid(row=r, column=0, sticky="w", pady=4, padx=(0, 10))
    vars_["resume"] = ctk.StringVar(value="")
    resume_box = ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_["resume"], width=W, values=[""])
    resume_box.grid(row=r, column=1, pady=4)

    def scan():
        found = find_incomplete(vars_["out"].get().strip() or ".")
        resume_box.configure(values=[""] + [f"{d}  ({n}/{t})" for d, n, t in found])
        if found and not vars_["resume"].get():
            vars_["resume"].set("")  # never resume unless asked
    scan()

    def browse_resume():
        d = filedialog.askdirectory(initialdir=vars_["out"].get() or ".")
        if d:
            vars_["resume"].set(d)
    rb = ctk.CTkFrame(frm, fg_color="transparent")
    rb.grid(row=r, column=2, padx=(8, 0))
    ctk.CTkButton(rb, font=font, text="Scan", width=60, command=scan).pack(side="left")
    ctk.CTkButton(rb, font=font, text="Browse", width=80, command=browse_resume).pack(side="left", padx=(4, 0))

    def start():
        try:
            for name, _, typ in FIELDS:
                result[name] = typ(vars_[name].get().strip())
            cam0_text = vars_["cam0"].get()
            cam1_text = vars_["cam1"].get()
            result["cam0_name"] = cam0_text.split(":", 1)[1].strip() if ":" in cam0_text else ""
            result["cam1_name"] = cam1_text.split(":", 1)[1].strip() if ":" in cam1_text else ""
            res = vars_["resume"].get().strip()
            if res.endswith(")") and "  (" in res:  # strip the "(saved/total)" label
                res = res.rsplit("  (", 1)[0]
            if res:
                try:
                    result.update({k: v for k, v in load_config(res).items() if k in RESUME_KEYS})
                except (OSError, ValueError):
                    raise AssertionError(f"no readable {CONFIG_NAME} in {res}")
            result["resume"] = res
            assert result["rows"] > 0 and result["cols"] > 0, "rows and columns must be > 0"
            assert result["out"] and result["prefix"], "output folder and prefix are required"
        except (ValueError, AssertionError) as e:
            result.clear()
            messagebox.showerror("Grid capture", f"Invalid setting: {e}")
            return
        root.destroy()

    btns = ctk.CTkFrame(frm, fg_color="transparent")
    btns.grid(row=len(FIELDS) + 1, column=0, columnspan=3, pady=(14, 0))
    ctk.CTkButton(btns, font=font, text="Start", width=100, command=start).pack(side="left", padx=6)
    ctk.CTkButton(btns, font=font, text="Cancel", width=100, fg_color="gray40", hover_color="gray30",
                  command=root.destroy).pack(side="left", padx=6)

    ctk.set_widget_scaling(0.5)
    ctk.set_window_scaling(1.5)

    root.bind("<Return>", lambda e: start())
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()
    return result or None


def open_cam(idx, w, h, fps, fourcc):
    """Opened cv2.VideoCapture, or None if the camera couldn't be opened."""
    cap = cv2.VideoCapture(idx, BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    print(f"cam {idx}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
          f" @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    return cap


class Camera:
    """Reads one camera in a background thread, so a stalled or unplugged device can't freeze
    the GUI. Keeps only the newest frame, and reopens the camera after it disconnects. A
    replugged camera may come back at a different index, so it is looked up again by name."""
    _all = []  # every Camera, so two cameras with the same name don't grab the same index

    def __init__(self, idx, name, w, h, fps, fourcc, label="", reconnect_after=1.0, retry_every=1.0):
        self.idx, self.name, self.label = idx, name, label or f"cam {idx}"
        self.mode = (w, h, fps, fourcc)
        self.shape = (h, w, 3)  # shape of the last good frame, for the placeholder image
        self.reconnect_after = reconnect_after  # seconds of failed reads before reopening
        self.retry_every = retry_every          # seconds between reopen attempts
        self.cap = None
        self._lock = threading.Lock()
        self._frame, self._t = None, 0.0
        self._stop = threading.Event()
        self._settings = threading.Event()
        Camera._all.append(self)
        self._thread = threading.Thread(target=self._run, name=self.label, daemon=True)
        self._thread.start()

    def _candidates(self):
        """Indices to try: the last one first, then others listed with the same name."""
        if not self.name:
            return [self.idx]
        cams = list_cams()
        if not cams:
            return [self.idx]
        taken = {c.idx for c in Camera._all if c is not self}
        same = [i for i, n in cams if n == self.name and i not in taken]
        return sorted(same, key=lambda i: i != self.idx)

    def _connect(self):
        for i in self._candidates():
            cap = open_cam(i, *self.mode)
            if cap is not None:
                if i != self.idx:
                    print(f"{self.label}: now at index {i} (was {self.idx})")
                self.idx, self.cap = i, cap
                print(f"{self.label}: connected")
                return True
        return False

    def _run(self):
        last_ok = 0.0
        while not self._stop.is_set():
            if self.cap is None:
                if not self._connect():
                    self._stop.wait(self.retry_every)
                    continue
                last_ok = time.time()
            if self._settings.is_set():  # done here so only this thread touches self.cap
                self._settings.clear()
                self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
            ok, frame = self.cap.read()
            now = time.time()
            if ok:
                last_ok = now
                self.shape = frame.shape
                with self._lock:
                    self._frame, self._t = frame, now
            elif now - last_ok > self.reconnect_after:
                print(f"{self.label}: lost, reconnecting...")
                self.cap.release()
                self.cap = None
            else:
                self._stop.wait(0.01)  # don't busy-spin on failed reads

    def latest(self, max_age=0.5):
        """Copy of the newest frame, or None if the camera is disconnected or stalled."""
        with self._lock:
            if self._frame is None or time.time() - self._t > max_age:
                return None
            frame = self._frame.copy()
        if frame.ndim == 2:  # IR / grayscale cameras give 1 channel, which hconcat/colour drawing can't mix
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        return frame

    def open_settings(self):
        """Open the driver settings dialog (exposure, WB, focus) on the reader thread."""
        self._settings.set()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if not self._thread.is_alive() and self.cap is not None:  # else read() is wedged in the driver
            self.cap.release()
        if self in Camera._all:
            Camera._all.remove(self)


class MirrorCamera:
    """Second 'camera' that shows another Camera's frames. For testing with one webcam, since
    a device can't be opened twice."""

    def __init__(self, src, label=""):
        self.src, self.label = src, label
        self.name = src.name

    idx = property(lambda self: self.src.idx)
    shape = property(lambda self: self.src.shape)

    def latest(self, max_age=0.5):
        return self.src.latest(max_age)

    def open_settings(self):
        pass

    def release(self):
        pass


def fname(prefix, r, c, rows, cols):
    nr, nc = len(str(rows - 1)), len(str(cols - 1))
    return f"{prefix}_r{r:0{max(nr, 2)}d}_c{c:0{max(nc, 2)}d}.png"


def next_empty(done, pos):
    """Next position after `pos` (row-major, wrapping) that has no image yet, or None."""
    rows, cols = done.shape
    i = pos[0] * cols + pos[1]
    for k in range(1, rows * cols + 1):
        j = (i + k) % (rows * cols)
        if not done.flat[j]:
            return divmod(j, cols)
    return None


def draw(frame, done, pos, msg):
    """Preview with a status line and a mini map of the grid (green = saved, red box = current)."""
    v = frame.copy()
    rows, cols = done.shape
    cell = max(8, min(30, 300 // max(rows, cols)))
    x0, y0 = v.shape[1] - cols * cell - 10, 10
    for r in range(rows):
        for c in range(cols):
            p0 = (x0 + c * cell, y0 + r * cell)
            p1 = (p0[0] + cell - 2, p0[1] + cell - 2)
            cv2.rectangle(v, p0, p1, (0, 200, 0) if done[r, c] else (80, 80, 80), -1)
            if (r, c) == tuple(pos):
                cv2.rectangle(v, p0, p1, (0, 0, 255), 2)
    status = f"row {pos[0]}  col {pos[1]}   {done.sum()}/{done.size} saved   {msg}"
    cv2.rectangle(v, (0, v.shape[0] - 40), (v.shape[1], v.shape[0]), (0, 0, 0), -1)
    cv2.putText(v, status, (10, v.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(v, "SPACE take snapshot | WASD move | n next square | q quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return v
