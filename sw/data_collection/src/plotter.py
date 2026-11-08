
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

def read_input(device_name: String):
    ser = serial.Serial(device_name)  # open serial port (try ttyUSB0 or ttyUSB1 if not working). 
    with open(f'../sensor_data/prototype-{datetime.datetime.now()}.csv', 'w', newline='') as csvfile:
        fieldnames = ['t_ms','vo_uv','vref_uv','i_pa','pos_steps','pos_mdeg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                writer.writerow(dict(zip(fieldnames, line.split(','))))

            if interrupted:
                break

            time.sleep(0.001)   
        