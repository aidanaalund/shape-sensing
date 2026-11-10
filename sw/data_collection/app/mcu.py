import csv
import datetime
import time

import serial


def send_cmd(ser, cmd: str):
    ser.write(cmd.encode())

def goto(ser, deg: int):
    ser.write(f"{deg}\r".encode())   # absolute angle needs Enter

def begin_communication(device_name: str, output_dir: str):
    ser = serial.Serial(device_name)

    with open(f'{output_dir}/sensor_data/prototype-{datetime.datetime.now()}.csv', 'w', newline='',
                                                                        encoding="utf8") as csvfile:
        fieldnames = ['t_ms','vref_uv','vo1_uv','vo2_uv','vo3_uv',
            'i1_pa','i2_pa','i3_pa','pos_steps','pos_mdeg']
        # fieldnames = ['t_ms','vref_uv','vo1_uv','vo2_uv','vo3_uv',
        #       'vo4_uv','vo5_uv','vo6_uv','vo7_uv','i1_pa','i2_pa',
        #       'i3_pa','i4_pa','i5_pa','i6_pa','i7_pa','pos_steps','pos_mdeg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode(errors="replace").strip()
                    if line.startswith("#"): # Status/log
                        print(line)
                    elif line: # Data
                        writer.writerow(dict(zip(fieldnames, line.split(','))))
            except KeyboardInterrupt:
                break

            time.sleep(0.001)
