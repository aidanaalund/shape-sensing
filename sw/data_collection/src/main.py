"""
    This program is a top level program that collects data
    from our sensor and cameras. It will output that collected data to
    relevant files in `../sensor_data`.
"""

from plotter import begin_communication

import serial.serialutil

DEVICE_NAME: str = '/dev/ttyACM0' # NOTE: Windows users, check device manager for appropriate name (e.g. 'COM4').

try:
    begin_communication(DEVICE_NAME)
except serial.serialutil.SerialException:
    print(f"ERROR: Couldn't find/open {DEVICE_NAME}. Check if the device is plugged in or if the correct name is used.")
    print("\n--- Ending program ---")

# TODO: start taking keyboard input and parsing it

# TODO: handle camera stuff and syncing