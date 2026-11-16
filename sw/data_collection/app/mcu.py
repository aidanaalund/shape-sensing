"""
    Serial link to the MCU, which streams csv-formatted sensor readings.

    A background thread reads the port continuously. Readings are only written
    to disk while a snapshot is active; each snapshot gets its own csv file.
"""

import csv
import threading
import time
from pathlib import Path

import serial

FIELDNAMES = ['t_ms', 'vref_uv', 'vo1_uv', 'vo2_uv', 'vo3_uv',
              'vo4_uv', 'vo5_uv', 'vo6_uv', 'vo7_uv', 'i1_pa', 'i2_pa',
              'i3_pa', 'i4_pa', 'i5_pa', 'i6_pa', 'i7_pa']


class Mcu:
    def __init__(self, device_name: str, baud: int = 115200):
        # Raises serial.serialutil.SerialException if the port can't be opened
        self.ser = serial.Serial(device_name, baud, timeout=0.1)
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._deadline = 0.0
        self._rows = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send_cmd(self, cmd: str):
        self.ser.write(cmd.encode())

    @property
    def collecting(self) -> bool:
        return self._file is not None

    def start_snapshot(self, path, duration: float) -> bool:
        """Begin writing readings to `path` for `duration` seconds.
        Returns False if a snapshot is already running."""
        with self._lock:
            if self._file is not None:
                return False
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, 'w', newline='', encoding='utf8')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
            self._writer.writeheader()
            self._rows = 0
            self._deadline = time.monotonic() + duration
            self.ser.reset_input_buffer()  # drop stale readings from before the snapshot
            return True

    def _end_snapshot(self):  # call with lock held
        self._file.close()
        print(f"snapshot done ({self._rows} rows)")
        self._file = self._writer = None

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode(errors="replace").strip()
            except (serial.SerialException, OSError):
                print("ERROR: lost connection to MCU")
                break
            with self._lock:
                if line.startswith("#"):  # Status/log
                    print(line)
                elif line and self._writer:  # Data
                    self._writer.writerow(dict(zip(FIELDNAMES, line.split(','))))
                    self._rows += 1
                if self._file and time.monotonic() >= self._deadline:
                    self._end_snapshot()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        with self._lock:
            if self._file:
                self._end_snapshot()
        self.ser.close()
