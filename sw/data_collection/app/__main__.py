"""
    This program is a top level program that collects data
    from our sensor and cameras. It will output that collected data to
    relevant files in `../sensor_data`.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial.serialutil

try:
    from . import grid_capture as cam  # python -m app
    from . import mcu as esp
except ImportError:
    import grid_capture as cam  # python app/
    import mcu as esp

SETTINGS_FILE = Path.cwd() / ".grid_capture.json" 
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440 # cv2.waitKeyEx


def placeholder(camera, other):
    """Black frame with a 'disconnected' notice, sized to match the other camera's frame
    (hconcat needs equal heights) or this camera's last good frame."""
    shape = other.shape if other is not None else camera.shape
    img = np.zeros(shape, np.uint8)
    scale = shape[1] / 1000
    cv2.putText(img, f"{camera.label.upper()} disconnected, attempting to reconnect", (int(40 * scale), shape[0] // 2 + int(60 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 0, 255), max(1, int(3 * scale)))
    return img

def dedupe_log(path):
    """Keep only the newest log row per file, preserving order."""
    try:
        with open(path, newline="", encoding="utf8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header, body = rows[0], rows[1:]
        last = {r[0]: i for i, r in enumerate(body) if r}
        keep = [r for i, r in enumerate(body) if r and last[r[0]] == i]
        if len(keep) != len(body):
            with open(path, "w", newline="", encoding="utf8") as f:
                csv.writer(f).writerows([header] + keep)
    except OSError:
        pass


def main():
    # Arguments
    ap = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", dest="mcu", type=str,
        help="MCU serial port, e.g. COM3 or /dev/ttyACM0 (default: last used)")
    ap.add_argument("--length", type=float,
        help="Duration of sensor data when a snapshot is started (default: 5)")
    ap.add_argument("--rows", type=int, help="# of rows in grid (default: 5)")
    ap.add_argument("--cols", type=int, help="# of columns in grid (default: 5)")
    ap.add_argument("--cam0", type=int, help="OpenCV camera for the left camera (default: None)")
    ap.add_argument("--cam1", type=int, help="OpenCV camera for the right camera (default: None)")
    ap.add_argument("--resume", type=str, metavar="FOLDER",
        help="Reopen an existing experiment folder (uses its config.json) and keep saving into it")
    ap.add_argument("--out", type=str, help="Output directory (default: ../data/experimental)")
    ap.add_argument("--prefix", type=str, help="Image format (default: img)")
    ap.add_argument("--width", type=int, help="# of horizontal pixels (default: 1920px)")
    ap.add_argument("--height", type=int, help="# of vertical pixels (default: 1080px)")
    ap.add_argument("--fps", type=int, help="Camera FPS (default: 30)")
    ap.add_argument("--fourcc", help="(default: MJPG)")
    ap.add_argument("--no-gui", action="store_true", help="Don't show the setup window")
    a = ap.parse_args()

    # defaults < last used settings < command line; the setup window shows the result
    cfg = {"rows": 5, "cols": 5, "cam0": 0, "cam1": 1, "cam0_name": "", "cam1_name": "", "out": "../data/experimental", "prefix": "img",
        "width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG", "mcu": "COM3" if sys.platform == "win32" else "/dev/ttyACM0", "length": 5}
    try:
        cfg.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
    resume = a.resume
    if not a.no_gui and not resume and (a.rows is None or a.cols is None): # 'Innomaker-U20CAM-1080p-S1: Inno'
        cfg = cam.setup_dialog(cfg)
        if cfg is None:
            return
        resume = cfg.pop("resume", "")
    elif a.cam0 is None:
        cams = cam.list_cams()
        cfg["cam0"], cfg["cam0_name"] = cam.pick_cam(cams, cfg["cam0"], cfg["cam0_name"])
        cfg["cam1"], cfg["cam1_name"] = cam.pick_cam(cams, cfg["cam1"], cfg["cam1_name"])

    if resume:
        out = Path(resume)
        try:
            saved = cam.load_config(out)
        except (OSError, ValueError):
            print(f"ERROR: no readable {cam.CONFIG_NAME} in {out}")
            return
        # the experiment's grid/camera settings win over the command line, or filenames wouldn't match
        cfg.update({k: v for k, v in saved.items() if k in cam.RESUME_KEYS})
        print(f"Resuming {out}")
    else:
        out = Path(cfg["out"]) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # no ':' so it's valid on Windows
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=1))
    a = argparse.Namespace(**cfg)

    out_cv = out / "grid_data"
    out_cv.mkdir(parents=True, exist_ok=True)
    if not resume:
        (out / cam.CONFIG_NAME).write_text(json.dumps(cfg, indent=1))
    log_path = out_cv / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf8")
    log = csv.writer(log_f)
    mcu_name = a.mcu
    if new_log:
        log.writerow(["file", "row", "col", "timestamp", "cam", "width", "height", "sensor_file"])

    mcu = None
    try:
        mcu = esp.Mcu(mcu_name)
        print(f"Connected to {mcu_name}")
    except serial.serialutil.SerialException:
        print(f"ERROR: Couldn't find/open {mcu_name}. Check if the device is plugged in or if the correct name is used.")
        print("Continuing without sensor data.")

    # A position is done when both images exist (and its sensor csv, if the MCU is connected)
    done = cam.saved_grid(out, cfg, sensors=mcu is not None)
    pos = [0, 0] if not done[0, 0] else list(cam.next_empty(done, (0, 0)) or (0, 0))
    print(f"{done.sum()} of {done.size} positions already saved in {out_cv}")

    cap0 = cam.Camera(a.cam0, a.cam0_name, a.width, a.height, a.fps, a.fourcc, label="left cam")
    if a.cam1 == a.cam0:  # one physical camera (testing): the right view mirrors the left
        print("Both cameras use the same device: mirroring it for the right camera")
        cap1 = cam.MirrorCamera(cap0, label="right cam")
    else:
        cap1 = cam.Camera(a.cam1, a.cam1_name, a.width, a.height, a.fps, a.fourcc, label="right cam")
    win = "Grid Capture"
    msg, msg_t = "", 0.0
    was_collecting = False

    out_sensor = out / "sensor_data"
    # Measurement mode: extra snapshots ("trials") of chosen positions, for repeatability and
    # drift analysis. Saved under <out>/repeat/ so they never count as, or overwrite, grid data.
    rep_cv, rep_sensor = out / cam.REPEAT_DIR / "grid_data", out / cam.REPEAT_DIR / "sensor_data"
    rep_log_f = rep_log = None
    measuring = False
    trials = cam.trials_saved(rep_cv, a.prefix)  # {(row, col): set of trial numbers} already on disk
    counts = np.zeros((a.rows, a.cols), int)
    for (r, c), ts in trials.items():
        if r < a.rows and c < a.cols:
            counts[r, c] = len(ts)

    def snapshot(frames, r, c, trial=None):
        """Save both camera frames and start the sensor recording for position (r, c).
        With a `trial` number it's a measurement-mode trial (..._t001) in the repeat folder."""
        nonlocal rep_log_f, rep_log
        name = Path(cam.fname(a.prefix, r, c, a.rows, a.cols))
        if trial is None:
            img_dir, sensor_dir, lg, lf, cols = out_cv, out_sensor, log, log_f, [r, c]
        else:
            if rep_log is None:
                rep_cv.mkdir(parents=True, exist_ok=True)
                p = rep_cv / "log.csv"
                new = not p.exists()
                rep_log_f = open(p, "a", newline="", encoding="utf8")  # noqa: SIM115
                rep_log = csv.writer(rep_log_f)
                if new:
                    rep_log.writerow(["file", "row", "col", "trial", "timestamp", "cam", "width", "height", "sensor_file"])
            name = name.with_stem(f"{name.stem}_t{trial:03d}")
            img_dir, sensor_dir, lg, lf, cols = rep_cv, rep_sensor, rep_log, rep_log_f, [r, c, trial]
        # The sensor csv shares the image's name, e.g. img_r00_c01_l.png -> img_r00_c01.csv
        sensor_name = ""
        if mcu:
            sensor_name = name.with_suffix(".csv").name
            mcu.start_snapshot(sensor_dir / sensor_name, a.length)
        for side, frame, cap in zip(("_l", "_r"), frames, (cap0, cap1)):
            n = name.with_stem(name.stem + side)
            cv2.imwrite(str(img_dir / n), frame)
            lg.writerow([n, *cols, datetime.now().isoformat(timespec="milliseconds"),
                         cap.idx, frame.shape[1], frame.shape[0], sensor_name])
        lf.flush()
        return name

    try:
        while True:
            # Frames come from the reader threads, so a dead camera can't block this loop
            frame0, frame1 = cap0.latest(), cap1.latest()
            cams_ok = frame0 is not None and frame1 is not None
            if frame0 is None:
                frame0 = placeholder(cap0, frame1)
            if frame1 is None:
                frame1 = placeholder(cap1, frame0)
            if frame0.shape[0] != frame1.shape[0]:  # hconcat needs equal heights (e.g. IR cam at another resolution)
                frame1 = cv2.resize(frame1, (round(frame1.shape[1] * frame0.shape[0] / frame1.shape[0]), frame0.shape[0]))
            scaley = min(1.0, 1200 / (frame0.shape[0] + frame1.shape[0]))
            scalex = min(1.0, 1400 / (frame0.shape[1] + frame1.shape[1]))
            combined = cv2.resize(cv2.hconcat([frame0,frame1]), None, fx=scalex, fy=scaley)
            collecting = bool(mcu and mcu.collecting)
            if was_collecting and not collecting:  # snapshot just finished
                msg, msg_t = f"snapshot done ({mcu.last_rows} rows) - ready", time.time() + 2  # shown ~4s
            was_collecting = collecting
            now = time.time()
            if collecting:  # persistent while recording
                shown = f"RECORDING {mcu.remaining:.1f}s - wait"
            else:
                shown = msg if now - msg_t < 2 else ""
            banner = ""
            if measuring:
                n = counts[pos[0], pos[1]]
                banner = (f"MEASURE  {n} extra snapshot{'' if n == 1 else 's'} here | SPACE snapshot | WASD move"
                          f" | m back to grid | q quit")
            im = cam.draw(combined, done, pos, shown, banner, counts if measuring else None)
            if mcu:
                cam.draw_photodiodes(im, mcu.latest, time.monotonic() - mcu.latest_t)
            cv2.imshow(win, im)
            k = cv2.waitKeyEx(15)  # always reached, so q / X work even with a camera down
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1: # window closed with X
                break
            if k == -1:
                continue
            ch = chr(k & 0xFF).lower() if k < 256 else ""
            move = {KEY_LEFT: (0, -1), KEY_RIGHT: (0, 1), KEY_UP: (-1, 0), KEY_DOWN: (1, 0)}.get(k) \
                or {"a": (0, -1), "d": (0, 1), "w": (-1, 0), "s": (1, 0)}.get(ch)
            if ch == "q" or k == 27:
                break
            elif ch == " ":
                if collecting:
                    continue  # countdown is already on screen
                if not cams_ok:  # never save half a pair
                    msg, msg_t = "camera disconnected - not saved", time.time()
                    continue
                if measuring:
                    ts = trials.setdefault(tuple(pos), set())
                    t = max(ts, default=0) + 1  # never reuse a number, even if a trial file was deleted
                    name = snapshot((frame0, frame1), *pos, trial=t)
                    ts.add(t)
                    counts[pos[0], pos[1]] = len(ts)
                    print(f"Saved {name.stem}")
                    msg, msg_t = f"saved {name.stem}", time.time()
                    continue
                name = snapshot((frame0, frame1), *pos)
                done[pos[0], pos[1]] = True
                print("Saved position")
                msg, msg_t = f"saved {name.stem}", time.time()
                nxt = cam.next_empty(done, pos)
                if nxt is None:
                    print("All positions saved")
                    msg = "All positions saved"
                else:
                    pos = list(nxt)
            elif ch == "m":
                if collecting:
                    msg, msg_t = "wait for the snapshot to finish", time.time()
                    continue
                measuring = not measuring
                msg, msg_t = ("measurement mode" if measuring else "grid mode"), time.time()
            elif ch == "n":
                pos = list(cam.next_empty(done, pos) or pos)
            elif move:
                pos = [min(max(pos[0] + move[0], 0), a.rows - 1), min(max(pos[1] + move[1], 0), a.cols - 1)]
            elif (ch == "p" or ch == "o") and mcu:
                mcu.send_cmd(ch)
            # elif ch == "c":
                # cap0.open_settings()
                # cap1.open_settings()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap0.release()
        cap1.release()
        if mcu:
            mcu.close()
        log_f.close()
        if rep_log_f:
            rep_log_f.close()
        dedupe_log(log_path)  # retaken positions leave duplicate rows
        cv2.destroyAllWindows()
    if not done.all():
        print(f"Stopped with {done.sum()}/{done.size} saved; run again with the same settings to resume")

if __name__ == '__main__':
    main()