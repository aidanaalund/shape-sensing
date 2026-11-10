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
        help="Device name/path. Windows users, check Device Manager. (default: /dev/ttyACM0)")
    ap.add_argument("--length", type=str,
        help="Duration of sensor data when a snapshot is started (default: 5)")
    ap.add_argument("--rows", type=int, help="# of rows in grid (default: 5)")
    ap.add_argument("--cols", type=int, help="# of columns in grid (default: 5)")
    ap.add_argument("--cam", type=int, help="OpenCV camera index (default: 0)")
    ap.add_argument("--out", type=str, help="Output directory (default: ../../data/experimental)")
    ap.add_argument("--prefix", type=str, help="Image format (default: img)")
    ap.add_argument("--width", type=int, help="# of horizontal pixels (default: 1280)")
    ap.add_argument("--height", type=int, help="# of vertical pixels (default: 720)")
    ap.add_argument("--fps", type=int, help="Camera FPS (default: 30)")
    ap.add_argument("--fourcc", help="(default: MJPG)")
    ap.add_argument("--no-gui", action="store_true", help="Don't show the setup window")
    a = ap.parse_args()

    # defaults < last used settings < command line; the setup window shows the result
    cfg = {"rows": 5, "cols": 5, "cam": 0, "cam_name": "", "out": "../../data/experimental", "prefix": "img",
        "width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG", "device": "/dev/ttyACM0", "length": 5}
    try:
        cfg.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    cfg.update({k: v for k, v in vars(a).items() if v is not None and k in cfg})
    if not a.no_gui and (a.rows is None or a.cols is None):
        cfg = cam.setup_dialog(cfg)
        if cfg is None:
            return
    elif a.cam is None:
        cfg["cam"], cfg["cam_name"] = cam.pick_cam(cam.list_cams(), cfg)
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=1))
    a = argparse.Namespace(**cfg)

    out = Path(a.out)
    out_cv = out / "grid_data"
    out_cv.mkdir(parents=True, exist_ok=True)
    log_path = out_cv / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf8")
    log = csv.writer(log_f)
    device = a.device
    if new_log:
        log.writerow(["file", "row", "col", "timestamp", "cam", "width", "height"])

    done = np.array([[(out_cv / cam.fname(a.prefix, r, c, a.rows, a.cols)).exists() for c in range(a.cols)]
                     for r in range(a.rows)])
    pos = [0, 0] if not done[0, 0] else list(cam.next_empty(done, (0, 0)) or (0, 0))
    print(f"{done.sum()} of {done.size} positions already saved in {out_cv}")

    cap = cam.open_cam(a.cam, a.width, a.height, a.fps, a.fourcc)
    win = "grid capture"
    msg, msg_t = "", 0.0

    # TODO: start taking keyboard input and parsing it
    # TODO: allow for 'snapshots'
    try:      
        device = a.device
        try:
            esp.begin_communication(device, out / "sensor_data")
        except serial.serialutil.SerialException:
            print(f"ERROR: Couldn't find/open {device}. Check if the device is plugged in or if the correct name is used.")
            print(f"Connected to {device}!")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("dropped frame")
                continue
            vis = cam.draw(frame, done, pos, msg if time.time() - msg_t < 2 else "")
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
                name = cam.fname(a.prefix, pos[0], pos[1], a.rows, a.cols)
                cv2.imwrite(str(out_cv / name), frame)
                log.writerow([name, pos[0], pos[1], datetime.now().isoformat(timespec="milliseconds"),
                              a.cam, frame.shape[1], frame.shape[0]])
                log_f.flush()
                done[pos[0], pos[1]] = True
                print("saved", name)
                msg, msg_t = f"saved {name}", time.time()
                nxt = cam.next_empty(done, pos)
                if nxt is None:
                    print("all positions saved")
                    msg = "all positions saved"
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
            elif ch == "c":
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        log_f.close()
        cv2.destroyAllWindows()
    if not done.all():
        print(f"stopped with {done.sum()}/{done.size} saved; run again with the same settings to resume")

if __name__ == '__main__':
    main()