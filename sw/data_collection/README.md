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

## Measurement mode (extra snapshots for repeatability / drift)

Press `m` to take extra snapshots of positions you pick, instead of stepping through the grid. The preview gets an orange border while it's on.

- Move to a position with `WASD`/arrows, then press `SPACE` for each extra snapshot. The position doesn't advance.
- The mini map shows each position's count of extra snapshots in an orange cell, and the top line shows the count for the current position.
- `m` again returns to normal grid capture.

Extra snapshots are saved apart from the grid, so they never count as (or overwrite) grid positions:

```
<experiment>/repeat/grid_data/img_r02_c03_t001_l.png, ..._r.png, log.csv (includes a trial column)
<experiment>/repeat/sensor_data/img_r02_c03_t001.csv
```

Snapshot numbers (`t001`, `t002`, ...) continue from what's already on disk, so resuming an experiment keeps counting.
