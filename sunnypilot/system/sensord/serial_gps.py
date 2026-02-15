#!/usr/bin/env python3
"""
Serial GPS daemon for Adafruit Ultimate GPS FeatherWing via CH9102 USB-UART.

Reads ASCII CSV lines (lat,lon,alt) from /dev/GPS at 460800 baud
and publishes cereal `gpsLocationExternal` messages at 10 Hz.

Speed and bearing are computed from consecutive GPS fixes since
the hardware only outputs position + altitude.
"""
import json
import math
import os
import time

import serial

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

SERIAL_PORT = "/dev/GPS"
SERIAL_PORT_FALLBACK = "/dev/ttyACM0"
BAUD_RATE = 460800

EARTH_RADIUS = 6371007.2  # meters


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Distance in meters between two lat/lon points."""
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = (math.sin(dlat / 2) ** 2 +
       math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
       math.sin(dlon / 2) ** 2)
  return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Bearing in degrees (0-360) from point 1 to point 2."""
  dlon = math.radians(lon2 - lon1)
  lat1_r = math.radians(lat1)
  lat2_r = math.radians(lat2)
  x = math.sin(dlon) * math.cos(lat2_r)
  y = (math.cos(lat1_r) * math.sin(lat2_r) -
       math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon))
  bearing = math.degrees(math.atan2(x, y))
  return bearing % 360.0


def parse_line(line: str) -> tuple[float, float, float] | None:
  """Parse 'lat,lon,alt' CSV line. Returns None on parse failure."""
  try:
    parts = line.strip().split(',')
    if len(parts) != 3:
      return None
    lat = float(parts[0])
    lon = float(parts[1])
    alt = float(parts[2])
    # Sanity check: valid GPS coordinates
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
      return None
    return (lat, lon, alt)
  except (ValueError, IndexError):
    return None


def main():
  pm = messaging.PubMaster(['gpsLocationExternal'])
  rk = Ratekeeper(SERVICE_LIST['gpsLocationExternal'].frequency, print_delay_threshold=None)
  mem_params = Params("/dev/shm/params")

  port = SERIAL_PORT if os.path.exists(SERIAL_PORT) else SERIAL_PORT_FALLBACK
  cloudlog.info(f"serial_gps: opening {port} at {BAUD_RATE} baud")

  ser = serial.Serial(port, BAUD_RATE, timeout=1.0)
  ser.reset_input_buffer()

  # Skip header line if present
  first_line = ser.readline().decode('ascii', errors='replace').strip()
  if not first_line or first_line.startswith('lat'):
    cloudlog.info(f"serial_gps: skipped header: {first_line}")
  else:
    parsed = parse_line(first_line)
    if parsed:
      cloudlog.info("serial_gps: first line was data, processing")

  cloudlog.info("serial_gps: streaming started")
  sample_count = 0

  # State for computing speed and bearing
  prev_lat = 0.0
  prev_lon = 0.0
  prev_time = 0.0
  speed = 0.0
  bearing = 0.0

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
        cloudlog.error("serial_gps: read error")
        rk.keep_time()
        continue

    parsed = parse_line(latest_line)
    if parsed is None:
      rk.keep_time()
      continue

    lat, lon, alt = parsed
    now = time.monotonic()
    has_fix = (lat != 0.0 or lon != 0.0)

    # Compute speed and bearing from consecutive fixes
    if prev_time > 0 and has_fix:
      dt = now - prev_time
      if dt > 0.01:  # Avoid division by near-zero
        dist = haversine(prev_lat, prev_lon, lat, lon)
        speed = dist / dt
        # Only update bearing if we've moved enough (>0.5m) to avoid noise
        if dist > 0.5:
          bearing = compute_bearing(prev_lat, prev_lon, lat, lon)

    if has_fix:
      prev_lat = lat
      prev_lon = lon
      prev_time = now

    # Publish gpsLocationExternal
    msg = messaging.new_message('gpsLocationExternal', valid=has_fix)
    gps = msg.gpsLocationExternal
    gps.latitude = lat
    gps.longitude = lon
    gps.altitude = alt
    gps.speed = speed
    gps.bearingDeg = bearing
    gps.horizontalAccuracy = 2.5  # Typical GPS accuracy ~2.5m
    gps.hasFix = has_fix
    gps.satelliteCount = 0  # Not available from this hardware
    gps.source = log.GpsLocationData.SensorSource.external
    gps.vNED = [0.0, 0.0, 0.0]
    gps.unixTimestampMillis = int(time.time() * 1000)
    gps.flags = 1 if has_fix else 0
    pm.send('gpsLocationExternal', msg)

    # Update LastGPSPosition for mapd and other consumers
    if has_fix:
      mem_params.put("LastGPSPosition", json.dumps({
        "latitude": lat,
        "longitude": lon,
        "bearing": bearing,
        "speed": speed,
        "timestamp": time.time(),
      }))

    sample_count += 1
    if sample_count % 100 == 0:  # every 10 seconds at 10Hz
      cloudlog.info(f"serial_gps: {sample_count} fixes, "
                    f"lat={lat:.6f} lon={lon:.6f} alt={alt:.1f} "
                    f"speed={speed:.1f}m/s bearing={bearing:.0f}° fix={has_fix}")

    rk.keep_time()


if __name__ == "__main__":
  main()
