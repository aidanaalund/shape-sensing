"""
    This program is a top level program that collects data
    from our sensor and cameras. It will output that collected data to
    relevant files in `../sensor_data`.
"""

from plotter import read_input

# TODO: handle user input to specify angle
# TODO: handle camera stuff and syncing
read_input('/dev/ttyACM0') # NOTE: Windows users, check device manager for appropriate name (e.g. 'COM4').