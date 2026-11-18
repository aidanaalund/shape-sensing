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

def main():
    # Arguments
    ap = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", dest="mcu", type=str,
        help="MCU serial port, e.g. COM3 or /dev/ttyACM0 (default: last used)")
    ap.add_argument("--length", type=float,
        help="Duration of sensor data when a snapshot is started (default: 10)")
    ap.add_argument("--rows", type=int, help="# of rows in grid (default: 5)")
    ap.add_argument("--cols", type=int, help="# of columns in grid (default: 5)")
    ap.add_argument("--cam0", type=int, help="OpenCV camera for the left camera (default: None)")
    ap.add_argument("--cam1", type=int, help="OpenCV camera for the right camera (default: None)")
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
        "width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG", "mcu": "COM3" if sys.platform == "win32" else "/dev/ttyACM0", "length": 10}
    try:
        cfg.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
    if not a.no_gui and (a.rows is None or a.cols is None): # 'Innomaker-U20CAM-1080p-S1: Inno'
        cfg = cam.setup_dialog(cfg)
        if cfg is None:
            return
    elif a.cam0 is None:
        cfg["cam0"], cfg["cam0_name"] = cam.pick_cam(cam.list_cams(), cfg["cam0"])
        cfg["cam1"], cfg["cam1_name"] = cam.pick_cam(cam.list_cams(), cfg["cam1"])
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=1))
    a = argparse.Namespace(**cfg)

    out = Path(a.out) / f"{datetime.now().isoformat(timespec="hours")}"
    out_cv = out / "grid_data"
    out_cv.mkdir(parents=True, exist_ok=True)
    log_path = out_cv / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf8")
    log = csv.writer(log_f)
    mcu_name = a.mcu
    if new_log:
        log.writerow(["file", "row", "col", "timestamp", "cam", "width", "height", "sensor_file"])

    done = np.array([[(out_cv / cam.fname(a.prefix, r, c, a.rows, a.cols)).exists() for c in range(a.cols)]
                     for r in range(a.rows)])
    pos = [0, 0] if not done[0, 0] else list(cam.next_empty(done, (0, 0)) or (0, 0))
    print(f"{done.sum()} of {done.size} positions already saved in {out_cv}")

    cap0 = cam.Camera(a.cam0, a.cam0_name, a.width, a.height, a.fps, a.fourcc, label="left cam")
    cap1 = cam.Camera(a.cam1, a.cam1_name, a.width, a.height, a.fps, a.fourcc, label="right cam")
    win = "Grid Capture"
    msg, msg_t = "", 0.0
    was_collecting = False

    mcu = None
    out_sensor = out / "sensor_data"
    try:
        try:
            mcu = esp.Mcu(mcu_name)
            print(f"Connected to {mcu_name}")
        except serial.serialutil.SerialException:
            print(f"ERROR: Couldn't find/open {mcu_name}. Check if the device is plugged in or if the correct name is used.")
            print("Continuing without sensor data.")

        while True:
            # Frames come from the reader threads, so a dead camera can't block this loop
            frame0, frame1 = cap0.latest(), cap1.latest()
            cams_ok = frame0 is not None and frame1 is not None
            if frame0 is None:
                frame0 = placeholder(cap0, frame1)
            if frame1 is None:
                frame1 = placeholder(cap1, frame0)
            scaley = min(1.0, 1200 / (frame0.shape[0] + frame1.shape[0]))
            scalex = min(1.0, 1400 / (frame0.shape[1] + frame1.shape[1]))
            combined = cv2.resize(cv2.hconcat([frame0,frame1]), None, fx=scalex, fy=scaley)
            collecting = bool(mcu and mcu.collecting)
            if was_collecting and not collecting:  # snapshot just finished
                msg, msg_t = f"snapshot done ({mcu.last_rows} rows) - ready", time.time() + 2  # shown ~4s
            was_collecting = collecting
            if collecting:  # persistent while recording
                shown = f"RECORDING {mcu.remaining:.1f}s - wait"
            else:
                shown = msg if time.time() - msg_t < 2 else ""
            im = cam.draw(combined, done, pos, shown)
            cv2.imshow(win, im)
            k = cv2.waitKeyEx(15)  # always reached, so q / X work even with a camera down
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1: # window closed with X
                break
            if k == -1:
                continue
            ch = chr(k & 0xFF).lower() if k < 256 else ""
            if ch == "q" or k == 27:
                break
            elif ch == " ":
                if mcu and mcu.collecting:
                    continue  # countdown is already on screen
                if not cams_ok:  # never save half a pair
                    msg, msg_t = "camera disconnected - not saved", time.time()
                    continue
                name = Path(cam.fname(a.prefix, pos[0], pos[1], a.rows, a.cols))
                name_l = name.with_stem(name.stem+"_l")
                name_r = name.with_stem(name.stem+"_r")
                # The sensor csv shares the image's name, e.g. img_0_1.jpg -> img_0_1.csv
                sensor_name = ""
                if mcu:
                    sensor_name = name.with_suffix(".csv").name
                    mcu.start_snapshot(out_sensor / sensor_name, a.length)
                cv2.imwrite(str(out_cv / name_l), frame0)
                log.writerow([name_l, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              cap0.idx, frame0.shape[1], frame0.shape[0], sensor_name])
                cv2.imwrite(str(out_cv / name_r), frame1)
                log.writerow([name_r, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              cap1.idx, frame1.shape[1], frame1.shape[0], sensor_name])
                log_f.flush()
                done[pos[0], pos[1]] = True
                print("Saved position")
                msg, msg_t = f"saved {sensor_name}", time.time()
                nxt = cam.next_empty(done, pos)
                if nxt is None:
                    print("All positions saved")
                    msg = "All positions saved"
                else:
                    pos = list(nxt)
            elif ch == "n":
                pos = list(cam.next_empty(done, pos) or pos)
            elif k == KEY_LEFT or ch == "a":
                pos[1] = max(0, pos[1] - 1)
            elif k == KEY_RIGHT or ch == "d":
                pos[1] = min(a.cols - 1, pos[1] + 1)
            elif k == KEY_UP or ch == "w":
                pos[0] = max(0, pos[0] - 1)
            elif k == KEY_DOWN or ch == "s":
                pos[0] = min(a.rows - 1, pos[0] + 1)
            elif (ch == "p" or ch == "o") and mcu:
                mcu.send_cmd(ch)
            elif ch == "c":
                cap0.open_settings()
                cap1.open_settings()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap0.release()
        cap1.release()
        if mcu:
            mcu.close()
        log_f.close()
        cv2.destroyAllWindows()
    if not done.all():
        print(f"Stopped with {done.sum()}/{done.size} saved; run again with the same settings to resume")

if __name__ == '__main__':
    main()