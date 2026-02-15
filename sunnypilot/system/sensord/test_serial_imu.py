#!/usr/bin/env python3
"""
E2E test for serial_imu.py — verifies cereal accelerometer + gyroscope messages.

Runs serial_imu in a background thread, subscribes to messages for ~5 seconds,
and validates data sanity (accel norm ~9.8 m/s² at rest, gyro near zero).
"""
import math
import threading
import time

import cereal.messaging as messaging


def main():
  # Start serial_imu in a background thread
  from openpilot.system.sensord.serial_imu import main as imu_main
  imu_thread = threading.Thread(target=imu_main, daemon=True)
  imu_thread.start()
  print("Started serial_imu in background thread")

  # Give it a moment to connect and start publishing
  time.sleep(1.0)

  # Subscribe to both services
  sm = messaging.SubMaster(['accelerometer', 'gyroscope'])

  accel_samples = []
  gyro_samples = []
  start = time.monotonic()
  duration = 5.0

  print(f"Collecting data for {duration:.0f} seconds...")
  while time.monotonic() - start < duration:
    sm.update(100)  # 100ms timeout

    if sm.updated['accelerometer']:
      a = sm['accelerometer']
      v = list(a.acceleration.v)
      accel_samples.append(v)
      if len(accel_samples) <= 5 or len(accel_samples) % 100 == 0:
        norm = math.sqrt(sum(x**2 for x in v))
        print(f"  accel #{len(accel_samples):4d}: [{v[0]:7.3f}, {v[1]:7.3f}, {v[2]:7.3f}] norm={norm:.3f}")

    if sm.updated['gyroscope']:
      g = sm['gyroscope']
      v = list(g.gyroUncalibrated.v)
      gyro_samples.append(v)
      if len(gyro_samples) <= 5 or len(gyro_samples) % 100 == 0:
        norm = math.sqrt(sum(x**2 for x in v))
        print(f"  gyro  #{len(gyro_samples):4d}: [{v[0]:7.4f}, {v[1]:7.4f}, {v[2]:7.4f}] norm={norm:.4f}")

  # Validation
  print(f"\n{'='*60}")
  print(f"Results: {len(accel_samples)} accel, {len(gyro_samples)} gyro samples in {duration:.0f}s")

  passed = True

  # Check sample counts (expect ~104 Hz * 5s = ~520 samples, allow wide margin)
  for name, samples in [("accel", accel_samples), ("gyro", gyro_samples)]:
    if len(samples) < 100:
      print(f"FAIL: only {len(samples)} {name} samples (expected ~520)")
      passed = False
    else:
      rate = len(samples) / duration
      print(f"OK: {len(samples)} {name} samples ({rate:.1f} Hz)")

  # Check accel norm at rest (~9.81 m/s²)
  if accel_samples:
    norms = [math.sqrt(sum(x**2 for x in s)) for s in accel_samples]
    avg_norm = sum(norms) / len(norms)
    if 7.0 < avg_norm < 13.0:
      print(f"OK: accel norm avg = {avg_norm:.3f} m/s² (expected ~9.81)")
    else:
      print(f"FAIL: accel norm avg = {avg_norm:.3f} m/s² (expected ~9.81)")
      passed = False

  # Check gyro near zero at rest
  if gyro_samples:
    gyro_norms = [math.sqrt(sum(x**2 for x in s)) for s in gyro_samples]
    avg_gyro_norm = sum(gyro_norms) / len(gyro_norms)
    if avg_gyro_norm < 1.0:  # < 1 rad/s at rest is reasonable
      print(f"OK: gyro norm avg = {avg_gyro_norm:.4f} rad/s (expected ~0)")
    else:
      print(f"FAIL: gyro norm avg = {avg_gyro_norm:.4f} rad/s (expected ~0)")
      passed = False

  # Check for rejected values (locationd sanity: accel > 100, gyro > 10)
  if accel_samples:
    max_accel = max(abs(x) for s in accel_samples for x in s)
    if max_accel > 100:
      print(f"FAIL: max accel component = {max_accel:.1f} m/s² (locationd rejects > 100)")
      passed = False

  if gyro_samples:
    max_gyro = max(abs(x) for s in gyro_samples for x in s)
    if max_gyro > 10:
      print(f"FAIL: max gyro component = {max_gyro:.4f} rad/s (locationd rejects > 10)")
      passed = False

  print(f"\n{'PASS' if passed else 'FAIL'}")
  return 0 if passed else 1


if __name__ == "__main__":
  exit(main())
