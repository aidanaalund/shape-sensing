
import serial
import signal
import datetime
import csv
import time

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
interrupted = False

ready = True

def send_cmd(ser, cmd: str):
    ser.write(cmd.encode())

def goto(ser, deg: int):
    ser.write(f"{deg}\r".encode())   # absolute angle needs Enter

def begin_communication(device_name: String):
    ser = serial.Serial(device_name)

    with open(f'../sensor_data/prototype-{datetime.datetime.now()}.csv', 'w', newline='') as csvfile:
        fieldnames = ['t_ms','vref_uv','vo1_uv','vo2_uv','vo3_uv','vo4_uv','vo5_uv','vo6_uv','vo7_uv',
            'i1_pa','i2_pa','i3_pa','i4_pa','i5_pa','i6_pa','i7_pa','pos_steps','pos_mdeg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            if ser.in_waiting > 0:
                if line.startswith("#"): # Status/log
                    print(line)
                    ready = True
                elif line: # Data
                    line = ser.readline().decode(errors="replace").strip()
                    writer.writerow(dict(zip(fieldnames, line.split(','))))

            if interrupted:
                break

            time.sleep(0.001)   
        