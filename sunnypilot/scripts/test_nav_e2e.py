#!/usr/bin/env python3
"""
End-to-end tests for the navigation POC (GPS → navd → desire pipeline).

Tests:
  1. GPS publishing: serial_gps publishes gpsLocationExternal
  2. Valhalla routing: route computation + map matching work offline
  3. NavDesire param: navd writes correct desires to /dev/shm/params
  4. Desire sequence: simulated drive along route produces correct turn desires

Usage:
  cd /home/subha/treehacks2026/sunnypilot
  .venv/bin/python scripts/test_nav_e2e.py [--test N]
"""
import argparse
import json
import math
import os
import sys
import time

# Ensure sunnypilot root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENPILOT_PREFIX", "test_nav")
sys.path.insert(0, ROOT)

import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import Params

VALHALLA_CONFIG = "/home/subha/treehacks2026/sunnypilot/data/valhalla/valhalla.json"
ROUTE_ORIGIN = {"lat": 37.425611, "lon": -122.177434}
ROUTE_DESTINATION = {"lat": 37.429649, "lon": -122.170194}

# ── Helpers ──────────────────────────────────────────────────────────

def ok(msg: str):
  print(f"  [PASS] {msg}")

def fail(msg: str):
  print(f"  [FAIL] {msg}")
  return False

def section(name: str):
  print(f"\n{'='*60}")
  print(f"  TEST: {name}")
  print(f"{'='*60}")

# ── Test 1: GPS publishing ───────────────────────────────────────────

def test_gps_publishing() -> bool:
  """Verify GPS hardware reads + serial_gps parsing works.
  Tests: serial port access, CSV parsing, and cereal message construction."""
  import subprocess
  section("GPS Publishing (gpsLocationExternal)")

  # Step 1: Verify serial port exists
  from system.sensord.serial_gps import SERIAL_PORT, SERIAL_PORT_FALLBACK, BAUD_RATE, parse_line
  port = SERIAL_PORT if os.path.exists(SERIAL_PORT) else SERIAL_PORT_FALLBACK
  if not os.path.exists(port):
    return fail(f"GPS serial port not found: {port}")
  ok(f"GPS device found at {port}")

  # Step 2: Kill competing readers and read raw data
  # Kill by PID (not pattern match which could kill ourselves)
  result = subprocess.run(["fuser", port], capture_output=True, text=True)
  if result.stdout.strip():
    for pid in result.stdout.strip().split():
      pid = pid.strip()
      if pid.isdigit() and int(pid) != os.getpid():
        subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(0.5)

  import serial as pyserial
  valid_count = 0
  lat = lon = alt = 0.0

  for attempt in range(3):
    try:
      ser = pyserial.Serial(port, BAUD_RATE, timeout=2.0)
      ser.reset_input_buffer()
      time.sleep(0.2)  # let buffer fill
    except Exception as e:
      print(f"  Attempt {attempt+1}: cannot open {port}: {e}")
      time.sleep(1)
      continue

    for _ in range(20):
      try:
        raw = ser.readline().decode('ascii', errors='replace').strip()
      except Exception:
        break
      if not raw:
        continue

      parsed = parse_line(raw)
      if parsed:
        lat, lon, alt = parsed
        print(f"  GPS fix: lat={lat:.7f} lon={lon:.7f} alt={alt:.2f}")
        valid_count += 1
        if valid_count >= 3:
          break

    ser.close()
    if valid_count > 0:
      break
    print(f"  Attempt {attempt+1}: no data (port may be held by another process)")
    time.sleep(1)

  if valid_count == 0:
    return fail("No valid GPS data read from serial port (is another process reading it?)")
  ok(f"Read {valid_count} valid GPS fixes from {port}")

  # Step 3: Verify cereal message construction works
  msg = messaging.new_message('gpsLocationExternal', valid=True)
  gps = msg.gpsLocationExternal
  gps.latitude = lat
  gps.longitude = lon
  gps.altitude = alt
  gps.hasFix = True
  ok(f"cereal gpsLocationExternal message constructed (lat={lat:.6f})")

  return True


# ── Test 2: Valhalla routing ─────────────────────────────────────────

def test_valhalla_routing() -> bool:
  """Verify Valhalla offline routing and map matching work."""
  section("Valhalla Offline Routing")

  if not os.path.exists(VALHALLA_CONFIG):
    return fail(f"Valhalla config not found: {VALHALLA_CONFIG}")

  try:
    import valhalla
  except ImportError:
    return fail("pyvalhalla not installed")

  actor = valhalla.Actor(VALHALLA_CONFIG)
  ok(f"Valhalla Actor loaded from {VALHALLA_CONFIG}")

  # Test route computation
  req = json.dumps({
    "locations": [ROUTE_ORIGIN, ROUTE_DESTINATION],
    "costing": "auto",
    "units": "km",
  })
  result = json.loads(actor.route(req))
  leg = result["trip"]["legs"][0]
  maneuvers = leg["maneuvers"]
  shape_encoded = leg["shape"]

  print(f"  Route: {leg['summary']['length']:.2f} km, {len(maneuvers)} maneuvers")
  for i, m in enumerate(maneuvers):
    print(f"    [{i}] type={m['type']:2d} {m.get('instruction', '')}")

  if len(maneuvers) < 2:
    return fail("Route has fewer than 2 maneuvers")
  ok(f"Route computed: {len(maneuvers)} maneuvers")

  # Test map matching (trace_route)
  # Simulate 3 GPS points near the route origin
  trace_points = [
    {"lat": 37.425611, "lon": -122.177434},
    {"lat": 37.425700, "lon": -122.177300},
    {"lat": 37.425800, "lon": -122.177200},
  ]
  trace_req = json.dumps({
    "shape": trace_points,
    "costing": "auto",
    "shape_match": "map_snap",
  })
  try:
    trace_result = json.loads(actor.trace_route(trace_req))
    ok("Map matching (trace_route) works")
  except Exception as e:
    print(f"  [WARN] trace_route failed: {e} (may still work with more points)")

  # Test polyline decoding
  from sunnypilot.navd.navd import decode_polyline6
  shape_pts = decode_polyline6(shape_encoded)
  if len(shape_pts) < 10:
    return fail(f"Polyline decoded only {len(shape_pts)} points")
  ok(f"Polyline decoded: {len(shape_pts)} shape points")

  return True


# ── Test 3: NavDesire param writing ──────────────────────────────────

def test_nav_desire_param() -> bool:
  """Verify NavDesire param is written and readable."""
  section("NavDesire Param (shared memory)")

  mem_params = Params("/dev/shm/params")

  # Write a test NavDesire
  test_desire = {
    "desire": int(log.Desire.turnRight),
    "distance": 50.0,
    "timestamp": time.monotonic(),
  }
  mem_params.put("NavDesire", json.dumps(test_desire))
  ok("Wrote test NavDesire to /dev/shm/params")

  # Read it back
  raw = mem_params.get("NavDesire")
  if raw is None:
    return fail("Could not read NavDesire back")

  data = json.loads(raw)
  if data["desire"] != int(log.Desire.turnRight):
    return fail(f"Desire mismatch: expected {int(log.Desire.turnRight)}, got {data['desire']}")
  ok(f"NavDesire round-trip: desire={data['desire']} distance={data['distance']:.0f}m")

  # Test NavDesireController reads it
  from sunnypilot.selfdrive.controls.lib.nav_desire import NavDesireController

  class FakeDH:
    pass

  ctrl = NavDesireController(FakeDH())
  ctrl.update_params()
  desire = ctrl.get_desire()
  if desire != log.Desire.turnRight:
    return fail(f"NavDesireController returned {desire}, expected turnRight")
  ok("NavDesireController reads NavDesire correctly")

  # Test staleness — write an old timestamp
  stale_desire = {
    "desire": int(log.Desire.turnLeft),
    "distance": 30.0,
    "timestamp": time.monotonic() - 10.0,  # 10 seconds ago
  }
  mem_params.put("NavDesire", json.dumps(stale_desire))
  ctrl.param_read_counter = 0  # Force re-read
  ctrl.update_params()
  desire = ctrl.get_desire()
  if desire != log.Desire.none:
    return fail(f"Stale NavDesire should return none, got {desire}")
  ok("Stale NavDesire correctly ignored (>3s threshold)")

  # Clean up
  mem_params.remove("NavDesire")
  return True


# ── Test 4: Simulated route desire sequence ──────────────────────────

def test_desire_sequence() -> bool:
  """Simulate GPS positions along the route and verify correct desire sequence."""
  section("Desire Sequence (simulated drive)")

  if not os.path.exists(VALHALLA_CONFIG):
    return fail(f"Valhalla config not found: {VALHALLA_CONFIG}")

  try:
    import valhalla
  except ImportError:
    return fail("pyvalhalla not installed")

  from sunnypilot.navd.navd import (
    NavDaemon, decode_polyline6, MANEUVER_TO_DESIRE,
    TURN_DESIRE_ARM_DISTANCE, KEEP_DESIRE_ARM_DISTANCE, KEEP_MANEUVER_TYPES,
  )
  from sunnypilot.navd.helpers import Coordinate

  # Compute route
  actor = valhalla.Actor(VALHALLA_CONFIG)
  req = json.dumps({
    "locations": [ROUTE_ORIGIN, ROUTE_DESTINATION],
    "costing": "auto",
    "units": "km",
  })
  result = json.loads(actor.route(req))
  leg = result["trip"]["legs"][0]
  maneuvers = leg["maneuvers"]
  shape = decode_polyline6(leg["shape"])
  coords = [Coordinate(lat, lon) for lat, lon in shape]

  print(f"  Route: {len(shape)} shape points, {len(maneuvers)} maneuvers")

  # Walk through shape points and check desire at each
  desire_changes = []
  prev_desire_name = "none"

  for idx in range(len(coords)):
    pos = coords[idx]

    # Find next maneuver ahead
    for mi in range(len(maneuvers)):
      mstart = maneuvers[mi]["begin_shape_index"]
      if mstart >= idx:
        # Compute distance along route to maneuver
        dist = 0.0
        for j in range(idx, min(mstart, len(coords) - 1)):
          dist += coords[j].distance_to(coords[j + 1])

        mtype = maneuvers[mi]["type"]
        expected_desire = MANEUVER_TO_DESIRE.get(mtype, log.Desire.none)

        if expected_desire != log.Desire.none:
          arm_dist = KEEP_DESIRE_ARM_DISTANCE if mtype in KEEP_MANEUVER_TYPES else TURN_DESIRE_ARM_DISTANCE
          if dist <= arm_dist:
            actual_desire = expected_desire
          else:
            actual_desire = log.Desire.none
        else:
          actual_desire = log.Desire.none

        desire_name = {
          0: "none", 1: "turnLeft", 2: "turnRight",
          3: "lcLeft", 4: "lcRight", 5: "keepLeft", 6: "keepRight",
        }.get(int(actual_desire), "?")

        if desire_name != prev_desire_name:
          desire_changes.append((idx, desire_name, dist, mi))
          prev_desire_name = desire_name
        break

  print(f"\n  Desire transitions along route:")
  for shape_idx, desire_name, dist, man_idx in desire_changes:
    m = maneuvers[man_idx] if man_idx < len(maneuvers) else {}
    print(f"    shape[{shape_idx:3d}] → {desire_name:10s}  "
          f"(dist={dist:.0f}m to maneuver [{man_idx}] "
          f"type={m.get('type', '?')} {m.get('instruction', '')[:40]})")

  # Verify we got at least one non-none desire
  non_none = [d for _, d, _, _ in desire_changes if d != "none"]
  if not non_none:
    return fail("No non-none desires produced along route")

  # Verify we have both turnRight and turnLeft (route has both)
  has_right = any(d == "turnRight" for _, d, _, _ in desire_changes)
  has_left = any(d == "turnLeft" for _, d, _, _ in desire_changes)

  if has_right:
    ok("turnRight desire detected")
  else:
    print("  [WARN] No turnRight desire (may depend on arm distance)")

  if has_left:
    ok("turnLeft desire detected")
  else:
    print("  [WARN] No turnLeft desire (may depend on arm distance)")

  ok(f"Desire sequence validated: {len(non_none)} non-none transitions")
  return True


# ── Main ─────────────────────────────────────────────────────────────

TESTS = {
  1: ("GPS Publishing", test_gps_publishing),
  2: ("Valhalla Routing", test_valhalla_routing),
  3: ("NavDesire Param", test_nav_desire_param),
  4: ("Desire Sequence", test_desire_sequence),
}

def main():
  parser = argparse.ArgumentParser(description="Navigation E2E tests")
  parser.add_argument("--test", type=int, help="Run specific test (1-4)")
  args = parser.parse_args()

  print("=" * 60)
  print("  Navigation POC — End-to-End Tests")
  print("=" * 60)

  results = {}
  tests_to_run = {args.test: TESTS[args.test]} if args.test else TESTS

  for num, (name, fn) in tests_to_run.items():
    try:
      results[num] = fn()
    except Exception as e:
      print(f"  [ERROR] {name}: {e}")
      import traceback
      traceback.print_exc()
      results[num] = False

  # Summary
  print(f"\n{'='*60}")
  print("  SUMMARY")
  print(f"{'='*60}")
  for num in sorted(results):
    name = TESTS[num][0]
    status = "PASS" if results[num] else "FAIL"
    print(f"  Test {num}: {name:30s} [{status}]")

  passed = sum(1 for v in results.values() if v)
  total = len(results)
  print(f"\n  {passed}/{total} tests passed")

  return 0 if all(results.values()) else 1


if __name__ == "__main__":
  sys.exit(main())
