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
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440   # cv2.waitKeyEx on Windows
SETTINGS_FILE = Path.home() / ".grid_capture.json"
FIELDS = [  # (name, label, type)
    ("rows", "Rows", int), ("cols", "Columns", int), ("cam", "Camera", lambda s: int(s.split(":")[0])),
    ("out", "Output folder", str), ("prefix", "File prefix", str),
    ("width", "Width [px]", int), ("height", "Height [px]", int), ("fps", "FPS", int), ("fourcc", "FourCC", str),
]


def list_cams(probe=False, n=6):
    """[(index, name)] of DirectShow cameras. Names come from pygrabber (instant, same order as
    cv2.CAP_DSHOW indices); without it, fall back to opening indices 0..n-1 if `probe`."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        pass
    found = []
    for i in range(n if probe else 0):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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
    """Tk form for the capture settings. Returns a dict, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Grid capture setup")
    frm = ttk.Frame(root, padding=12)
    frm.grid()
    vars_, result = {}, {}
    for i, (name, label, _) in enumerate(FIELDS):
        ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=2)
        vars_[name] = tk.StringVar(value=str(defaults[name]))
        if name == "cam":
            box = ttk.Combobox(frm, textvariable=vars_[name], width=27)
            box.grid(row=i, column=1, pady=2)

            def detect(box=box, probe=True):
                root.config(cursor="watch"); root.update()
                cams = list_cams(probe)
                root.config(cursor="")
                if cams:
                    box["values"] = [cam_label(*c) for c in cams]
                    vars_["cam"].set(cam_label(*pick_cam(cams, defaults)))
                elif probe:
                    messagebox.showwarning("Grid capture", "No cameras found")
            detect(probe=False)
            ttk.Button(frm, text="Refresh", command=detect).grid(row=i, column=2, padx=4)
        elif name == "fourcc":
            ttk.Combobox(frm, textvariable=vars_[name], width=27, values=["MJPG", "YUY2"]).grid(row=i, column=1, pady=2)
        else:
            ttk.Entry(frm, textvariable=vars_[name], width=30).grid(row=i, column=1, pady=2)
        if name == "out":
            ttk.Button(frm, text="Browse", command=lambda: vars_["out"].set(
                filedialog.askdirectory(initialdir=vars_["out"].get() or ".") or vars_["out"].get())
            ).grid(row=i, column=2, padx=4)

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

    btns = ttk.Frame(frm)
    btns.grid(row=len(FIELDS), column=0, columnspan=3, pady=(10, 0))
    ttk.Button(btns, text="Start", command=start).pack(side="left", padx=4)
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side="left", padx=4)
    root.bind("<Return>", lambda e: start())
    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()
    return result or None


def open_cam(idx, w, h, fps, fourcc):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--cam", type=int, help="OpenCV camera index")
    ap.add_argument("--out")
    ap.add_argument("--prefix")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--fourcc")
    ap.add_argument("--no-gui", action="store_true", help="don't show the setup window")
    a = ap.parse_args()

    # defaults < last used settings < command line; the setup window shows the result
    cfg = dict(rows=5, cols=5, cam=0, cam_name="", out="grid_data", prefix="img",
               width=1280, height=720, fps=30, fourcc="MJPG")
    try:
        cfg.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
    if not a.no_gui and (a.rows is None or a.cols is None):
        cfg = setup_dialog(cfg)
        if cfg is None:
            return
    elif a.cam is None:
        cfg["cam"], cfg["cam_name"] = pick_cam(list_cams(), cfg)
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=1))
    a = argparse.Namespace(**cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["file", "row", "col", "timestamp", "cam", "width", "height"])

    done = np.array([[(out / fname(a.prefix, r, c, a.rows, a.cols)).exists() for c in range(a.cols)]
                     for r in range(a.rows)])
    pos = [0, 0] if not done[0, 0] else list(next_empty(done, (0, 0)) or (0, 0))
    print(f"{done.sum()} of {done.size} positions already saved in {out}")

    cap = open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    win = "grid capture"
    msg, msg_t = "", 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("dropped frame")
                continue
            vis = draw(frame, done, pos, msg if time.time() - msg_t < 2 else "")
            scale = min(1.0, 1600 / vis.shape[1])
            cv2.imshow(win, cv2.resize(vis, None, fx=scale, fy=scale))
            k = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed with X
                break
            if k == -1:
                continue
            ch = chr(k & 0xFF).lower() if k < 256 else ""
            if ch == "q" or k == 27:
                break
            elif ch == " ":
                name = fname(a.prefix, pos[0], pos[1], a.rows, a.cols)
                cv2.imwrite(str(out / name), frame)
                log.writerow([name, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              a.cam, frame.shape[1], frame.shape[0]])
                log_f.flush()
                done[pos[0], pos[1]] = True
                print("saved", name)
                msg, msg_t = f"saved {name}", time.time()
                nxt = next_empty(done, pos)
                if nxt is None:
                    print("all positions saved")
                    msg = "all positions saved"
                else:
                    pos = list(nxt)
            elif ch == "n":
                pos = list(next_empty(done, pos) or pos)
            elif k == KEY_LEFT or ch == "a":
                pos[1] = max(0, pos[1] - 1)
            elif k == KEY_RIGHT or ch == "d":
                pos[1] = min(a.cols - 1, pos[1] + 1)
            elif k == KEY_UP or ch == "w":
                pos[0] = max(0, pos[0] - 1)
            elif k == KEY_DOWN or ch == "s":
                pos[0] = min(a.rows - 1, pos[0] + 1)
            elif ch == "c":
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cap.release()
        log_f.close()
        cv2.destroyAllWindows()
    if not done.all():
        print(f"stopped with {done.sum()}/{done.size} saved; run again with the same settings to resume")


if __name__ == "__main__":
    main()
