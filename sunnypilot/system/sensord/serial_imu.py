#!/usr/bin/env python3
"""
Serial IMU daemon for LSM6DSOX connected via CP2104 USB-UART bridge.

Reads ASCII CSV lines (ax,ay,az,gx,gy,gz) from /dev/IMU at 460800 baud
and publishes cereal `accelerometer` + `gyroscope` messages at 104 Hz.

Values from the Arduino/Feather are already in m/s² (accel) and rad/s (gyro).
"""
import time
import serial

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

SERIAL_PORT = "/dev/IMU"
BAUD_RATE = 460800
SENSOR_SOURCE = log.SensorEventData.SensorSource.lsm6ds3


def create_accel_event(ax: float, ay: float, az: float, ts: int) -> log.SensorEventData:
  event = log.SensorEventData.new_message()
  event.timestamp = ts
  event.version = 1
  event.sensor = 1   # SENSOR_ACCELEROMETER
  event.type = 1     # SENSOR_TYPE_ACCELEROMETER
  event.source = SENSOR_SOURCE
  a = event.init('acceleration')
  a.v = [ax, ay, az]
  a.status = 1
  return event


def create_gyro_event(gx: float, gy: float, gz: float, ts: int) -> log.SensorEventData:
  event = log.SensorEventData.new_message()
  event.timestamp = ts
  event.version = 2
  event.sensor = 5   # SENSOR_GYRO_UNCALIBRATED
  event.type = 16    # SENSOR_TYPE_GYROSCOPE_UNCALIBRATED
  event.source = SENSOR_SOURCE
  g = event.init('gyroUncalibrated')
  g.v = [gx, gy, gz]
  g.status = 1
  return event


def parse_line(line: str) -> tuple[float, float, float, float, float, float] | None:
  """Parse 'ax,ay,az,gx,gy,gz' CSV line. Returns None on parse failure."""
  try:
    parts = line.strip().split(',')
    if len(parts) != 6:
      return None
    values = [float(p) for p in parts]
    return (values[0], values[1], values[2], values[3], values[4], values[5])
  except (ValueError, IndexError):
    return None


def main():
  pm = messaging.PubMaster(['accelerometer', 'gyroscope'])
  rk = Ratekeeper(SERVICE_LIST['accelerometer'].frequency, print_delay_threshold=None)

  cloudlog.info(f"serial_imu: opening {SERIAL_PORT} at {BAUD_RATE} baud")

  ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
  ser.reset_input_buffer()

  # Skip header line if present
  first_line = ser.readline().decode('ascii', errors='replace').strip()
  if not first_line or first_line.startswith('ax'):
    cloudlog.info(f"serial_imu: skipped header: {first_line}")
  else:
    # First line was data, try to parse and use it
    parsed = parse_line(first_line)
    if parsed:
      cloudlog.info("serial_imu: first line was data, processing")

  cloudlog.info("serial_imu: streaming started")
  sample_count = 0

  while True:
    # Drain serial buffer to get freshest data
    latest_line = None
    while ser.in_waiting:
      raw = ser.readline()
      try:
        latest_line = raw.decode('ascii', errors='replace').strip()
      except Exception:
        continue

    # If nothing was buffered, do a blocking read
    if latest_line is None:
      try:
        raw = ser.readline()
        latest_line = raw.decode('ascii', errors='replace').strip()
      except Exception:
        cloudlog.error("serial_imu: read error")
        rk.keep_time()
        continue

    parsed = parse_line(latest_line)
    if parsed is None:
      rk.keep_time()
      continue

    ax, ay, az, gx, gy, gz = parsed
    ts = time.monotonic_ns()

    # Publish accelerometer
    accel_evt = create_accel_event(ax, ay, az, ts)
    msg_accel = messaging.new_message('accelerometer', valid=True)
    msg_accel.accelerometer = accel_evt
    pm.send('accelerometer', msg_accel)

    # Publish gyroscope
    gyro_evt = create_gyro_event(gx, gy, gz, ts)
    msg_gyro = messaging.new_message('gyroscope', valid=True)
    msg_gyro.gyroscope = gyro_evt
    pm.send('gyroscope', msg_gyro)

    sample_count += 1
    if sample_count % 1040 == 0:  # ~every 10 seconds at 104 Hz
      cloudlog.info(f"serial_imu: {sample_count} samples published, "
                    f"accel=[{ax:.2f},{ay:.2f},{az:.2f}] gyro=[{gx:.4f},{gy:.4f},{gz:.4f}]")

    rk.keep_time()


if __name__ == "__main__":
  main()
