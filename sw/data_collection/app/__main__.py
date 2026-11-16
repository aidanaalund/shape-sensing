"""
    This program is a top level program that collects data
    from our sensor and cameras. It will output that collected data to
    relevant files in `../sensor_data`.
"""

import argparse
import csv
import json
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

def main():
    # Arguments
    ap = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=str,
        help="MCU name/path. Windows users: check Device Manager. (default: /dev/ttyACM0)")
    ap.add_argument("--length", type=float,
        help="Duration of sensor data when a snapshot is started (default: 10)")
    ap.add_argument("--rows", type=int, help="# of rows in grid (default: 5)")
    ap.add_argument("--cols", type=int, help="# of columns in grid (default: 5)")
    ap.add_argument("--cam0", type=int, help="OpenCV camera for camera 0 (default: None)")
    ap.add_argument("--cam1", type=int, help="OpenCV camera for camera 1 (default: None)")
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
        "width": 1920, "height": 1080, "fps": 30, "fourcc": "MJPG", "mcu": "/dev/ttyACM0", "length": 10}
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

    out = Path(a.out) / f"{datetime.now().isoformat(timespec="seconds")}"
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

    cap0 = cam.open_cam(a.cam0, a.width, a.height, a.fps, a.fourcc)
    cap1 = cam.open_cam(a.cam0, a.width, a.height, a.fps, a.fourcc)
    win0 = "Grid Capture (0)"
    win1 = "Grid Capture (1)"
    msg, msg_t = "", 0.0

    mcu = None
    out_sensor = out / "sensor_data"
    try:
        try:
            mcu = esp.Mcu(mcu_name)
            print(f"Connected to {mcu_name}!")
        except serial.serialutil.SerialException:
            print(f"ERROR: Couldn't find/open {mcu_name}. Check if the device is plugged in or if the correct name is used.")
            print("Continuing without sensor data.")

        while True:
            ok0, frame0 = cap0.read()
            ok1, frame1 = cap1.read()
            if not ok0 or not ok1:
                print("Dropped frame")
                continue
            vis0 = cam.draw(frame0, done, pos, msg if time.time() - msg_t < 2 else "")
            vis1 = cam.draw(frame1, done, pos, msg if time.time() - msg_t < 2 else "")
            scale0 = min(1.0, 1600 / vis0.shape[1])
            scale1 = min(1.0, 1600 / vis1.shape[1])
            cv2.imshow(win0, cv2.resize(vis0, None, fx=scale0, fy=scale0))
            cv2.imshow(win1, cv2.resize(vis1, None, fx=scale1, fy=scale1))
            k = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(win0, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(win1, cv2.WND_PROP_VISIBLE) < 1:
                break   # window closed with X
            if k == -1:
                continue
            ch = chr(k & 0xFF).lower() if k < 256 else ""
            if ch == "q" or k == 27:
                break
            elif ch == " ":
                if mcu and mcu.collecting:
                    msg, msg_t = "snapshot still running", time.time()
                    continue
                name = cam.fname(a.prefix, pos[0], pos[1], a.rows, a.cols)
                # The sensor csv shares the image's name, e.g. img_0_1.jpg -> img_0_1.csv
                sensor_name = ""
                if mcu:
                    sensor_name = Path(name).with_suffix(".csv").name
                    mcu.start_snapshot(out_sensor / sensor_name, a.length)
                cv2.imwrite(str(out_cv / name), frame0)
                log.writerow([name, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              a.cam, frame0.shape[1], frame0.shape[0], sensor_name])
                cv2.imwrite(str(out_cv / name), frame1)
                log.writerow([name, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              a.cam, frame1.shape[1], frame1.shape[0], sensor_name])
                log_f.flush()
                done[pos[0], pos[1]] = True
                print("Saved", name)
                msg, msg_t = f"saved {name}", time.time()
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
                cap0.set(cv2.CAP_PROP_SETTINGS, 1)
                cap1.set(cv2.CAP_PROP_SETTINGS, 1)
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