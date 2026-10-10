# System Description

The system has the following inputs:

- 4 buttons
- 2 ADC readings

The system has the following outputs:

- Terminal output

## What does the system do?

    - SW1 (button0): return to 0 degrees
    - SW2 (button1): tighten to TIGHTEN_DEG degrees
    - SW3 (button2): debug print (readings for calibrating the photodiode)
    - SW4 (button3): disable/reenable printing

## How does it do it?

Samples with ADC at `y` hz
Tracks stepper motor state with a number
