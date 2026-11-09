"""
    This program is a top level program that collects data
    from our sensor and cameras. It will output that collected data to
    relevant files in `../sensor_data`.
"""

import mcu

import serial.serialutil

DEVICE_NAME: str = '/dev/ttyACM0' # NOTE: Windows users, check device manager for appropriate name (e.g. 'COM4').
SNAPSHOT_LENGTH = 5 # number of seconds we take data from to average at 2 samples / second

# TODO: handle camera stuff and syncing

try:
    mcu.begin_communication(DEVICE_NAME)
except serial.serialutil.SerialException:
    print(f"ERROR: Couldn't find/open {DEVICE_NAME}. Check if the device is plugged in or if the correct name is used.")
    print("\n--- Ending program ---")

print(f"Connected to {DEVICE_NAME}!")
print(f"\nGuide:")
print(" *   0..359 <Enter>  move to that absolute angle (degrees)")
print(" *   ] / [           increment / decrement by MOVE_SMALL (1 deg)")
print(" *   } / {           increment / decrement by MOVE_MED (10 deg)")
print(" *   > / <           increment / decrement by MOVE_LARGE (90 deg)")
print(" *   p               pause / resume CSV streaming")
print(" *   d               debug print (raw readings, for calibrating the photodiodes")
print(" *   ?               help")

while True:
    ...
    # TODO: start taking keyboard input and parsing it
    # TODO: allow for 'snapshots'