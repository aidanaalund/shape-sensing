# Data Collection Application

## Requirements

- Python 3.12>=

## How to use

In your terminal of choice, create a virtual environment in the `data_collection` folder (optional) and run `pip install -r requirements.txt` to install dependencies.

Once everything installs, run `python app` in your shell from this folder.

Before using the application, ensure the following:

- You are plugged in to the cameras
- You are plugged into the MCU
- Connections to photodiodes, LEDs, and other equipment are secure
- Config:
  - 20x20 grid
  - 5 second snapshot length
  - 1920x1080 images @ 30FPS
  - If you are resuming from an in-progress experiment, use the folder browser button to select that folder

Once you click start, check that:

- The left camera feed matches the camera marked 'L', and the right camera feed matches the camera marked 'R'
  - If not, close the app and start again, changing which camera maps to what

As you collect snapshots, periodically check that data is being saved properly. Cameras may momentarily disconnect if the connectors are touched. The app will attempt to recover if this happens.

Quit the app by pressing 'q' when you're done!
