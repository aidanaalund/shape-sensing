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
import sys
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from cv2_enumerate_cameras import enumerate_cameras

FONT_SIZE = 24

# Capture backend per platform
BACKEND = (cv2.CAP_DSHOW if sys.platform == "win32"
           else cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)

FIELDS = [  # (name, label, type)
("rows", "Rows", int), ("cols", "Columns", int), ("cam", "Camera", lambda s: int(s.split(":")[0])),
("out", "Output folder", str), ("device", "Device Name", str), ("length", "Snapshot Window", int), ("prefix", "File prefix", str),
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
        pass
    found = []
    for i in range(n if probe else 0):
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened() and cap.read()[0]:
            found.append((i, f"camera {i}"))
        cap.release()
    return found


def cam_label(i, name):
    return f"{i}: {name}"


def pick_cam(cams, cfg):
    """Last used camera if it's still connected (matched by name), else the highest index,
    since a built-in laptop camera usually enumerates first."""
    for i, name in cams:
        if name == cfg.get("cam_name"):
            return i, name
    return cams[-1] if cams else (cfg["cam"], cfg.get("cam_name", ""))


def setup_dialog(defaults):
    """CustomTkinter form for the capture settings. Returns a dict, or None if cancelled.
    CustomTkinter scales widgets with the system DPI automatically."""
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    root.geometry("360x360")
    root.title("Grid capture setup")
    font = ctk.CTkFont(size=FONT_SIZE)  # CTkFont scales with ctk widget scaling
    frm = ctk.CTkFrame(root, fg_color="transparent")
    frm.pack(padx=16, pady=16)
    vars_, result = {}, {}
    W = 260
    for i, (name, label, _) in enumerate(FIELDS):
        ctk.CTkLabel(frm, font=font, text=label, anchor="w").grid(row=i, column=0, sticky="w", pady=4, padx=(0, 10))
        vars_[name] = ctk.StringVar(value=str(defaults[name]))
        if name == "cam":
            box = ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_[name], width=W, values=[""])
            box.grid(row=i, column=1, pady=4)

            def detect(box=box, probe=True):
                root.configure(cursor="watch"); root.update()
                cams = list_cams(probe)
                root.configure(cursor="")
                if cams:
                    box.configure(values=[cam_label(*c) for c in cams])
                    vars_["cam"].set(cam_label(*pick_cam(cams, defaults)))
                elif probe:
                    messagebox.showwarning("Grid capture", "No cameras found")
            detect(probe=False)
            ctk.CTkButton(frm, font=font, text="Refresh", width=80, command=detect).grid(row=i, column=2, padx=(8, 0))
        elif name == "fourcc":
            ctk.CTkComboBox(frm, font=font, dropdown_font=font, variable=vars_[name], width=W,
                values=["MJPG", "YUY2"]).grid(row=i, column=1, pady=4)
        else:
            ctk.CTkEntry(frm, font=font, textvariable=vars_[name], width=W).grid(row=i, column=1, pady=4)
        if name == "out":
            ctk.CTkButton(frm, font=font, text="Browse", width=80, command=lambda: vars_["out"].set(
                filedialog.askdirectory(initialdir=vars_["out"].get() or ".") or vars_["out"].get())
            ).grid(row=i, column=2, padx=(8, 0))

    def start():
        try:
            for name, _, typ in FIELDS:
                result[name] = typ(vars_[name].get().strip())
            cam_text = vars_["cam"].get()
            result["cam_name"] = cam_text.split(":", 1)[1].strip() if ":" in cam_text else ""
            assert result["rows"] > 0 and result["cols"] > 0, "rows and columns must be > 0"
            assert result["out"] and result["prefix"], "output folder and prefix are required"
        except (ValueError, AssertionError) as e:
            result.clear()
            messagebox.showerror("Grid capture", f"Invalid setting: {e}")
            return
        root.destroy()

    btns = ctk.CTkFrame(frm, fg_color="transparent")
    btns.grid(row=len(FIELDS), column=0, columnspan=3, pady=(14, 0))
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
    cap = cv2.VideoCapture(idx, BACKEND)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise SystemExit(f"camera {idx} did not open")
    print(f"cam {idx}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
          f" @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    return cap


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
    cv2.putText(v, "SPACE save | arrows/WASD move | n next empty | c cam settings | q stop", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return v
