#!/usr/bin/env python3
"""
Full system E2E test — runs the entire sunnypilot stack on Jetson Thor
with real cameras, IMU, GPS, and navigation. Mocks only the panda/car layer.

Prerequisites:
  1. holoscan_frame_publisher.py running inside Docker (real cameras)
  2. IMU plugged in (/dev/IMU or /dev/ttyUSB0)
  3. GPS plugged in (/dev/GPS or /dev/ttyACM0)
  4. /JETSON marker file exists (sudo touch /JETSON)
  5. Valhalla tiles built (data/valhalla/tiles/)

Usage:
  cd /home/subha/treehacks2026/sunnypilot
  LD_LIBRARY_PATH=.venv/lib:/usr/local/cuda/lib64 \
    .venv/bin/python scripts/test_full_system.py [--duration 60]
"""
import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

VENV_PYTHON = os.path.join(ROOT, '.venv', 'bin', 'python')
LD_PATH = f"{ROOT}/.venv/lib:/usr/local/cuda/lib64"

# Services to monitor and their expected frequencies
MONITOR_SERVICES = {
  # Hardware layer (mocked)
  'pandaStates': 10,
  'peripheralState': 2,
  'can': 100,
  'carState': 100,
  'carOutput': 100,
  # Sensor layer (real hardware)
  'accelerometer': 104,
  'gyroscope': 104,
  'gpsLocationExternal': 10,
  # Compute layer (real inference)
  'modelV2': 20,
  'driverMonitoringState': 20,
  # Localization
  'livePose': 20,
  'liveCalibration': 4,
  # Navigation
  'navInstruction': 1,
  # Control layer
  'deviceState': 2,
  'selfdriveState': 100,
  'carControl': 100,
  'controlsState': 100,
  'longitudinalPlan': 20,
}


def setup_params():
  """Pre-set params required for startup."""
  from openpilot.common.params import Params
  from openpilot.system.version import terms_version, training_version, terms_version_sp

  params = Params()

  # Accept terms and training
  params.put("HasAcceptedTerms", terms_version)
  params.put("HasAcceptedTermsSP", terms_version_sp)
  params.put("CompletedTrainingVersion", training_version)

  # Disable things that block startup
  params.put_bool("DisableUpdates", True)
  params.put_bool("OffroadMode", False)
  params.put_bool("OpenpilotEnabledToggle", True)
  params.put_bool("IsDriverViewEnabled", False)
  params.put_bool("IsTakingSnapshot", False)
  params.put_bool("DisableLogging", True)
  params.put_bool("RecordFront", False)

  # Set a dongle ID so registration doesn't block
  params.put("DongleId", "jetson_test_0000000000")
  params.put("HardwareSerial", "jetson_thor_test")

  print("  Params configured for startup")
  return params


def start_process(name, module_path, env=None):
  """Start a Python process as a subprocess."""
  proc_env = os.environ.copy()
  proc_env['LD_LIBRARY_PATH'] = LD_PATH
  proc_env['PYTHONPATH'] = ROOT
  proc_env['NOBOARD'] = '1'
  if env:
    proc_env.update(env)

  is_script = module_path.endswith('.py')
  cmd = [VENV_PYTHON, module_path] if is_script else [VENV_PYTHON, '-m', module_path]
  proc = subprocess.Popen(
    cmd,
    cwd=ROOT,
    env=proc_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
  )
  return proc


def launch_all():
  """Launch all daemons in the correct order and return process dict."""
  procs = {}

  print("\n--- Phase 1: Mock Panda/Car ---")
  procs['mock_car'] = start_process('mock_car', 'scripts/mock_car.py')
  print(f"  mock_car started (pid={procs['mock_car'].pid})")
  time.sleep(2)  # Let CarParams get written

  print("\n--- Phase 2: Hardware Daemons (real sensors) ---")
  procs['hardwared'] = start_process('hardwared', 'system.hardware.hardwared')
  print(f"  hardwared started (pid={procs['hardwared'].pid})")
  time.sleep(2)  # hardwared needs to publish deviceState with started=True

  procs['serial_imu'] = start_process('serial_imu', 'system.sensord.serial_imu')
  procs['serial_gps'] = start_process('serial_gps', 'system.sensord.serial_gps')
  print(f"  serial_imu started (pid={procs['serial_imu'].pid})")
  print(f"  serial_gps started (pid={procs['serial_gps'].pid})")

  procs['jetson_camerad'] = start_process('jetson_camerad', 'selfdrive.camerad.jetson_camerad')
  print(f"  jetson_camerad started (pid={procs['jetson_camerad'].pid})")
  time.sleep(3)  # Cameras need time to initialize VisionIPC

  print("\n--- Phase 3: Compute Daemons ---")
  procs['modeld'] = start_process('modeld', 'selfdrive.modeld.modeld')
  procs['mock_dmonitoringd'] = start_process('mock_dmonitoringd', 'selfdrive.monitoring.mock_dmonitoringd')
  print(f"  modeld started (pid={procs['modeld'].pid})")
  print(f"  mock_dmonitoringd started (pid={procs['mock_dmonitoringd'].pid})")
  time.sleep(3)  # ONNX models need warmup

  print("\n--- Phase 4: Localization + Navigation ---")
  procs['locationd'] = start_process('locationd', 'selfdrive.locationd.locationd')
  procs['calibrationd'] = start_process('calibrationd', 'selfdrive.locationd.calibrationd')
  procs['paramsd'] = start_process('paramsd', 'selfdrive.locationd.paramsd')
  procs['navd'] = start_process('navd', 'sunnypilot.navd.navd')
  print(f"  locationd started (pid={procs['locationd'].pid})")
  print(f"  calibrationd started (pid={procs['calibrationd'].pid})")
  print(f"  paramsd started (pid={procs['paramsd'].pid})")
  print(f"  navd started (pid={procs['navd'].pid})")

  print("\n--- Phase 5: Control Daemons ---")
  procs['plannerd'] = start_process('plannerd', 'selfdrive.controls.plannerd')
  procs['radard'] = start_process('radard', 'selfdrive.controls.radard')
  procs['selfdrived'] = start_process('selfdrived', 'selfdrive.selfdrived.selfdrived')
  print(f"  plannerd started (pid={procs['plannerd'].pid})")
  print(f"  radard started (pid={procs['radard'].pid})")
  print(f"  selfdrived started (pid={procs['selfdrived'].pid})")
  time.sleep(1)

  procs['controlsd'] = start_process('controlsd', 'selfdrive.controls.controlsd')
  print(f"  controlsd started (pid={procs['controlsd'].pid})")

  return procs


def check_alive(procs):
  """Check which processes are still alive."""
  alive = {}
  dead = {}
  for name, proc in procs.items():
    if proc.poll() is None:
      alive[name] = proc
    else:
      dead[name] = proc
  return alive, dead


def get_dead_output(proc, max_lines=20):
  """Get last N lines of output from a dead process."""
  try:
    out = proc.stdout.read().decode(errors='replace')
    lines = out.strip().split('\n')
    return '\n'.join(lines[-max_lines:])
  except Exception:
    return "(no output)"


def monitor(procs, duration):
  """Monitor cereal messages for the given duration."""
  import cereal.messaging as messaging
  from openpilot.common.params import Params

  print(f"\n{'='*70}")
  print(f"  MONITORING ({duration}s)")
  print(f"{'='*70}")

  service_names = list(MONITOR_SERVICES.keys())
  sm = messaging.SubMaster(service_names)
  mem_params = Params("/dev/shm/params")

  msg_counts = {s: 0 for s in service_names}
  first_seen = {s: None for s in service_names}
  start_time = time.monotonic()
  last_report = start_time

  while time.monotonic() - start_time < duration:
    sm.update(200)

    for s in service_names:
      if sm.updated[s]:
        msg_counts[s] += 1
        if first_seen[s] is None:
          first_seen[s] = time.monotonic() - start_time

    elapsed = time.monotonic() - start_time

    # Print status every 10 seconds
    if elapsed - (last_report - start_time) >= 10:
      last_report = time.monotonic()
      alive, dead = check_alive(procs)

      print(f"\n  --- {elapsed:.0f}s elapsed ---")
      print(f"  Processes: {len(alive)} alive, {len(dead)} dead")
      if dead:
        for name in dead:
          print(f"    DEAD: {name} (exit={dead[name].returncode})")

      # Print message counts
      print(f"  {'Service':<30s} {'Count':>7s} {'Hz':>7s} {'Expected':>9s} {'Status':>8s}")
      for s in service_names:
        count = msg_counts[s]
        hz = count / elapsed if elapsed > 0 else 0
        expected_hz = MONITOR_SERVICES[s]
        status = "OK" if hz > expected_hz * 0.5 else ("LOW" if hz > 0 else "NONE")
        print(f"  {s:<30s} {count:>7d} {hz:>7.1f} {expected_hz:>7d}Hz {'['+status+']':>8s}")

      # Print nav desire if available
      try:
        raw = mem_params.get("NavDesire")
        if raw:
          import json
          nd = json.loads(raw)
          desire_names = {0: "none", 1: "turnLeft", 2: "turnRight", 5: "keepLeft", 6: "keepRight"}
          d = desire_names.get(nd.get("desire", 0), "?")
          print(f"\n  NavDesire: {d} (dist={nd.get('distance', 0):.0f}m)")
      except Exception:
        pass

      # Print GPS position
      try:
        raw = mem_params.get("LastGPSPosition")
        if raw:
          import json
          pos = json.loads(raw)
          print(f"  GPS: lat={pos.get('latitude', 0):.6f} lon={pos.get('longitude', 0):.6f}")
      except Exception:
        pass

      # Print model output summary if we have modelV2
      if sm.updated.get('modelV2') and msg_counts.get('modelV2', 0) > 0:
        try:
          mv = sm['modelV2']
          if len(mv.velocity.x) > 0:
            print(f"  Model: vel={mv.velocity.x[0]:.1f}m/s, "
                  f"lanes={len(mv.laneLines)}, leads={len(mv.leads)}")
        except Exception:
          pass

  return msg_counts


def print_summary(msg_counts, duration, procs):
  """Print final summary."""
  print(f"\n{'='*70}")
  print(f"  FINAL SUMMARY")
  print(f"{'='*70}")

  alive, dead = check_alive(procs)

  # Process status
  print(f"\n  Process Status ({len(alive)} alive / {len(dead)} dead):")
  for name in sorted(procs.keys()):
    proc = procs[name]
    if proc.poll() is None:
      print(f"    {name:<25s} ALIVE (pid={proc.pid})")
    else:
      print(f"    {name:<25s} DEAD  (exit={proc.returncode})")

  # Dead process diagnostics
  if dead:
    print(f"\n  Dead Process Output:")
    for name, proc in dead.items():
      output = get_dead_output(proc, max_lines=5)
      print(f"    --- {name} ---")
      for line in output.split('\n'):
        print(f"      {line}")

  # Service status
  print(f"\n  Service Status (over {duration}s):")
  print(f"  {'Service':<30s} {'Count':>7s} {'Actual':>8s} {'Expected':>9s} {'Result':>8s}")

  passed = 0
  failed = 0
  for s, expected_hz in MONITOR_SERVICES.items():
    count = msg_counts.get(s, 0)
    hz = count / duration if duration > 0 else 0
    # Pass if we got at least 30% of expected rate (generous for startup transients)
    ok = hz > expected_hz * 0.3
    status = "PASS" if ok else "FAIL"
    if ok:
      passed += 1
    else:
      failed += 1
    print(f"  {s:<30s} {count:>7d} {hz:>7.1f}Hz {expected_hz:>7d}Hz  [{status}]")

  # Overall
  total = passed + failed
  print(f"\n  Result: {passed}/{total} services active")

  # Key pipeline checks
  print(f"\n  Pipeline Checks:")
  checks = [
    ("Cameras → modeld",    msg_counts.get('modelV2', 0) > 0),
    ("IMU → locationd",     msg_counts.get('livePose', 0) > 0),
    ("GPS → serial_gps",    msg_counts.get('gpsLocationExternal', 0) > 0),
    ("Navigation → navd",   msg_counts.get('navInstruction', 0) > 0),
    ("Controls → carControl", msg_counts.get('carControl', 0) > 0),
    ("Selfdrived running",  msg_counts.get('selfdriveState', 0) > 0),
  ]
  for desc, ok in checks:
    print(f"    {'PASS' if ok else 'FAIL'}: {desc}")

  all_critical = all(ok for _, ok in checks)
  print(f"\n  {'ALL CRITICAL CHECKS PASSED' if all_critical else 'SOME CHECKS FAILED'}")

  return 0 if all_critical else 1


def cleanup(procs):
  """Stop all subprocesses."""
  print(f"\n--- Cleanup ---")
  for name, proc in procs.items():
    if proc.poll() is None:
      proc.send_signal(signal.SIGINT)
  time.sleep(2)
  for name, proc in procs.items():
    if proc.poll() is None:
      print(f"  Force-killing {name}")
      proc.kill()
  for proc in procs.values():
    try:
      proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
      pass
  print("  All processes stopped")


def main():
  parser = argparse.ArgumentParser(description="Full system E2E test")
  parser.add_argument("--duration", type=int, default=60, help="Monitor duration in seconds")
  args = parser.parse_args()

  print("=" * 70)
  print("  SUNNYPILOT FULL SYSTEM E2E TEST")
  print("  Real: cameras, IMU, GPS, modeld, locationd, navd, controlsd")
  print("  Mock: panda, car (Honda Civic Bosch)")
  print("=" * 70)

  # Preflight checks
  print("\n--- Preflight Checks ---")
  checks_ok = True

  if not os.path.exists('/JETSON'):
    print("  WARN: /JETSON marker not found (sudo touch /JETSON)")

  if not os.path.exists('/dev/ttyUSB0') and not os.path.exists('/dev/IMU'):
    print("  WARN: IMU not found (/dev/IMU or /dev/ttyUSB0)")
    checks_ok = False

  if not os.path.exists('/dev/ttyACM0') and not os.path.exists('/dev/GPS'):
    print("  WARN: GPS not found (/dev/GPS or /dev/ttyACM0)")
    checks_ok = False

  valhalla_config = os.path.join(ROOT, 'data', 'valhalla', 'valhalla.json')
  if not os.path.exists(valhalla_config):
    print("  WARN: Valhalla config not found (data/valhalla/valhalla.json)")

  # Check cameras
  from selfdrive.camerad.shm_buffer import SHM_PATH_ROAD
  if not os.path.exists(SHM_PATH_ROAD):
    print(f"  WARN: Camera shm not found ({SHM_PATH_ROAD})")
    print("        Is holoscan_frame_publisher.py running in Docker?")

  print(f"  Preflight: {'OK' if checks_ok else 'warnings above'}")

  # Setup params
  print("\n--- Setup Params ---")
  params = setup_params()

  # Launch all processes
  procs = launch_all()

  print(f"\n  Total: {len(procs)} processes launched")
  print(f"  Waiting 5s for initialization...")
  time.sleep(5)

  # Check which are still alive
  alive, dead = check_alive(procs)
  print(f"  After init: {len(alive)} alive, {len(dead)} dead")
  for name in dead:
    print(f"    DEAD: {name}")
    output = get_dead_output(dead[name], max_lines=3)
    for line in output.split('\n'):
      print(f"      {line}")

  try:
    # Monitor
    msg_counts = monitor(procs, args.duration)

    # Summary
    result = print_summary(msg_counts, args.duration, procs)

  except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    result = 1

  finally:
    cleanup(procs)

  return result


if __name__ == '__main__':
  sys.exit(main())
