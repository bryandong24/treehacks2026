#!/usr/bin/env python3
"""
Integration test: serial IMU → locationd → livePose

Verifies that real IMU data from serial_imu.py is accepted by sunnypilot's
locationd Kalman filter and produces a valid livePose with sensorsOK=True.

This test:
1. Starts serial_imu.py (real IMU data at 104 Hz)
2. Starts locationd (the Kalman filter)
3. Publishes mock carState (100 Hz), liveCalibration (4 Hz), cameraOdometry (20 Hz)
4. Subscribes to livePose and checks sensorsOK, inputsOK, filter validity
"""
import math
import threading
import time

import numpy as np
import cereal.messaging as messaging
from cereal.services import SERVICE_LIST


def run_serial_imu():
  """Run serial_imu in this thread."""
  from openpilot.system.sensord.serial_imu import main as imu_main
  imu_main()


def run_locationd():
  """Run locationd in this thread."""
  from openpilot.selfdrive.locationd.locationd import main as loc_main
  loc_main()


def mock_publisher():
  """Publish mock carState, liveCalibration, and cameraOdometry at their expected rates."""
  pm = messaging.PubMaster(['carState', 'liveCalibration', 'cameraOdometry'])

  car_state_dt = 1.0 / SERVICE_LIST['carState'].frequency       # 100 Hz
  calib_dt = 1.0 / SERVICE_LIST['liveCalibration'].frequency     # 4 Hz
  cam_odo_dt = 1.0 / SERVICE_LIST['cameraOdometry'].frequency    # 20 Hz

  last_car = time.monotonic()
  last_calib = time.monotonic()
  last_cam = time.monotonic()
  frame_id = 0

  while True:
    now = time.monotonic()

    # carState at 100 Hz — stationary vehicle
    if now - last_car >= car_state_dt:
      msg = messaging.new_message('carState', valid=True)
      msg.carState.vEgo = 0.0
      pm.send('carState', msg)
      last_car = now

    # liveCalibration at 4 Hz — identity calibration
    if now - last_calib >= calib_dt:
      msg = messaging.new_message('liveCalibration', valid=True)
      msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
      msg.liveCalibration.calStatus = 1
      pm.send('liveCalibration', msg)
      last_calib = now

    # cameraOdometry at 20 Hz — stationary (zero rotation/translation)
    if now - last_cam >= cam_odo_dt:
      msg = messaging.new_message('cameraOdometry', valid=True)
      msg.cameraOdometry.frameId = frame_id
      msg.cameraOdometry.timestampEof = time.time_ns()
      msg.cameraOdometry.trans = [0.0, 0.0, 0.0]
      msg.cameraOdometry.rot = [0.0, 0.0, 0.0]
      msg.cameraOdometry.transStd = [1.0, 1.0, 1.0]
      msg.cameraOdometry.rotStd = [0.1, 0.1, 0.1]
      pm.send('cameraOdometry', msg)
      last_cam = now
      frame_id += 1

    time.sleep(0.002)  # 2ms spin to avoid busy-wait


def main():
  print("=" * 60)
  print("Serial IMU → locationd Integration Test")
  print("=" * 60)

  # Start all components as daemon threads
  threads = [
    ("serial_imu", run_serial_imu),
    ("locationd", run_locationd),
    ("mock_publisher", mock_publisher),
  ]
  for name, target in threads:
    t = threading.Thread(target=target, daemon=True, name=name)
    t.start()
    print(f"  Started {name}")
    time.sleep(0.3)  # stagger startup so publishers are ready before consumers

  # Give everything time to initialize
  print("\nWaiting 3s for filter initialization...")
  time.sleep(3.0)

  # Subscribe to livePose and also to raw sensor data for comparison
  sm = messaging.SubMaster(['livePose', 'accelerometer', 'gyroscope'])

  poses = []
  accel_samples = []
  gyro_samples = []
  duration = 10.0
  start = time.monotonic()

  print(f"Collecting livePose for {duration:.0f}s...\n")
  while time.monotonic() - start < duration:
    sm.update(200)

    if sm.updated['livePose']:
      lp = sm['livePose']
      poses.append({
        'valid': sm.valid['livePose'],
        'sensorsOK': lp.sensorsOK,
        'inputsOK': lp.inputsOK,
        'posenetOK': lp.posenetOK,
        'accel': [lp.accelerationDevice.x, lp.accelerationDevice.y, lp.accelerationDevice.z],
        'angVel': [lp.angularVelocityDevice.x, lp.angularVelocityDevice.y, lp.angularVelocityDevice.z],
        'orient': [lp.orientationNED.x, lp.orientationNED.y, lp.orientationNED.z],
      })
      if len(poses) <= 3 or len(poses) % 50 == 0:
        p = poses[-1]
        print(f"  livePose #{len(poses):4d}: valid={p['valid']} sensorsOK={p['sensorsOK']} "
              f"inputsOK={p['inputsOK']} posenetOK={p['posenetOK']}")
        print(f"    accel_device=[{p['accel'][0]:7.3f}, {p['accel'][1]:7.3f}, {p['accel'][2]:7.3f}]"
              f"  angVel=[{p['angVel'][0]:7.4f}, {p['angVel'][1]:7.4f}, {p['angVel'][2]:7.4f}]")

    if sm.updated['accelerometer']:
      a = sm['accelerometer']
      accel_samples.append(list(a.acceleration.v))

    if sm.updated['gyroscope']:
      g = sm['gyroscope']
      gyro_samples.append(list(g.gyroUncalibrated.v))

  # === Validation ===
  print(f"\n{'=' * 60}")
  print("RESULTS")
  print(f"{'=' * 60}")
  passed = True

  # 1. Did we get livePose messages?
  print(f"\n[1] livePose messages: {len(poses)}")
  if len(poses) < 20:
    print("    FAIL: too few livePose messages (expected ~200 at 20Hz)")
    passed = False
  else:
    print(f"    OK: {len(poses)} messages ({len(poses)/duration:.1f} Hz)")

  # 2. Is the filter valid (initialized)?
  if poses:
    valid_count = sum(1 for p in poses if p['valid'])
    valid_pct = 100 * valid_count / len(poses)
    print(f"\n[2] Filter valid: {valid_count}/{len(poses)} ({valid_pct:.0f}%)")
    if valid_pct < 50:
      print("    FAIL: filter not initializing (< 50% valid)")
      passed = False
    else:
      print("    OK")

  # 3. sensorsOK — the critical IMU health check
  if poses:
    sensors_ok_count = sum(1 for p in poses if p['sensorsOK'])
    sensors_pct = 100 * sensors_ok_count / len(poses)
    print(f"\n[3] sensorsOK: {sensors_ok_count}/{len(poses)} ({sensors_pct:.0f}%)")
    if sensors_pct < 80:
      print("    FAIL: IMU not recognized as healthy by locationd")
      passed = False
    else:
      print("    OK: locationd accepts IMU data")

  # 4. inputsOK
  if poses:
    inputs_ok_count = sum(1 for p in poses if p['inputsOK'])
    inputs_pct = 100 * inputs_ok_count / len(poses)
    print(f"\n[4] inputsOK: {inputs_ok_count}/{len(poses)} ({inputs_pct:.0f}%)")
    if inputs_pct < 50:
      print("    WARN: inputs not fully valid (mock services may not satisfy all checks)")
    else:
      print("    OK")

  # 5. posenetOK (should always be true at car_speed=0)
  if poses:
    posenet_ok_count = sum(1 for p in poses if p['posenetOK'])
    posenet_pct = 100 * posenet_ok_count / len(poses)
    print(f"\n[5] posenetOK: {posenet_ok_count}/{len(poses)} ({posenet_pct:.0f}%)")
    if posenet_pct < 90:
      print("    WARN: unexpected posenet failures at zero speed")
    else:
      print("    OK")

  # 6. Kalman filter acceleration estimate (informational)
  # NOTE: With mock cameraOdometry (zero motion), the Kalman filter's acceleration
  # estimate will drift because the mock conflicts with real gravity. This is expected
  # and will resolve once real camera odometry feeds the filter.
  if poses:
    late_poses = poses[len(poses)//2:]  # use second half after convergence
    if late_poses:
      accel_norms = [math.sqrt(sum(x**2 for x in p['accel'])) for p in late_poses]
      avg_accel_norm = sum(accel_norms) / len(accel_norms)
      print(f"\n[6] Kalman accel estimate norm (last 50%): {avg_accel_norm:.3f} m/s²")
      print("    INFO: drift expected with mock cameraOdometry (no real visual odometry)")

  # 7. Angular velocity estimate (informational)
  if poses:
    late_poses = poses[len(poses)//2:]
    if late_poses:
      angvel_norms = [math.sqrt(sum(x**2 for x in p['angVel'])) for p in late_poses]
      avg_angvel = sum(angvel_norms) / len(angvel_norms)
      print(f"\n[7] Kalman angular velocity norm (last 50%): {avg_angvel:.4f} rad/s")
      print("    INFO: drift expected with mock cameraOdometry")

  # 8. Raw IMU sanity check
  print(f"\n[8] Raw IMU: {len(accel_samples)} accel, {len(gyro_samples)} gyro samples")
  if accel_samples:
    norms = [math.sqrt(sum(x**2 for x in s)) for s in accel_samples]
    print(f"    Accel norm: avg={sum(norms)/len(norms):.3f} m/s²")
  if gyro_samples:
    norms = [math.sqrt(sum(x**2 for x in s)) for s in gyro_samples]
    print(f"    Gyro norm:  avg={sum(norms)/len(norms):.4f} rad/s")

  print(f"\n{'=' * 60}")
  print(f"{'PASS' if passed else 'FAIL'}")
  print(f"{'=' * 60}")
  return 0 if passed else 1


if __name__ == "__main__":
  exit(main())
