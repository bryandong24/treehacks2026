#!/usr/bin/env python3
"""
Navigation daemon for Jetson Thor.

Uses Valhalla for offline A-to-B routing on Stanford-area OSM data.
Subscribes to gpsLocationExternal, publishes navInstruction,
and writes NavDesire param for desire_helper.

Runs at 1 Hz.
"""
import json
import math
import time

import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.navd.helpers import Coordinate

VALHALLA_CONFIG = "/home/subha/treehacks2026/sunnypilot/data/valhalla/valhalla.json"

# Hardcoded Stanford demo route
ROUTE_ORIGIN = {"lat": 37.425611, "lon": -122.177434}
ROUTE_DESTINATION = {"lat": 37.429649, "lon": -122.170194}

# Distance thresholds (meters)
TURN_DESIRE_ARM_DISTANCE = 100.0
KEEP_DESIRE_ARM_DISTANCE = 150.0
REROUTE_DISTANCE = 200.0
GPS_HISTORY_SIZE = 5

# Valhalla maneuver type constants
# See https://valhalla.github.io/valhalla/api/turn-by-turn/api-reference/
MANEUVER_TO_DESIRE = {
  # Right turns
  9: log.Desire.turnRight,    # kSlightRight
  10: log.Desire.turnRight,   # kRight
  11: log.Desire.turnRight,   # kSharpRight
  12: log.Desire.turnRight,   # kUturnRight
  # Left turns
  13: log.Desire.turnLeft,    # kUturnLeft
  14: log.Desire.turnLeft,    # kSharpLeft
  15: log.Desire.turnLeft,    # kLeft
  16: log.Desire.turnLeft,    # kSlightLeft
  # Keep/exit right
  18: log.Desire.keepRight,   # kRampRight
  20: log.Desire.keepRight,   # kExitRight
  23: log.Desire.keepRight,   # kStayRight
  37: log.Desire.keepRight,   # kMergeRight
  # Keep/exit left
  19: log.Desire.keepLeft,    # kRampLeft
  21: log.Desire.keepLeft,    # kExitLeft
  24: log.Desire.keepLeft,    # kStayLeft
  38: log.Desire.keepLeft,    # kMergeLeft
}

KEEP_MANEUVER_TYPES = {18, 19, 20, 21, 23, 24, 37, 38}

# Roundabout types — no desire needed, model follows the road
# 26 = kRoundaboutEnter, 27 = kRoundaboutExit
# 1 = kStart, 2 = kStartRight, 3 = kStartLeft, 4 = kDestination
# 5 = kDestinationRight, 6 = kDestinationLeft, 7 = kBecomes
# 8 = kContinue, 17 = kRampStraight, 25 = kStayStraight
# These all map to Desire.none (default)


def decode_polyline6(encoded: str) -> list[tuple[float, float]]:
  """Decode Valhalla encoded polyline (precision 6) to list of (lat, lon)."""
  result = []
  index = 0
  lat = 0
  lng = 0
  while index < len(encoded):
    # Decode latitude
    shift = 0
    res = 0
    while True:
      b = ord(encoded[index]) - 63
      index += 1
      res |= (b & 0x1F) << shift
      shift += 5
      if b < 0x20:
        break
    lat += (~(res >> 1)) if (res & 1) else (res >> 1)

    # Decode longitude
    shift = 0
    res = 0
    while True:
      b = ord(encoded[index]) - 63
      index += 1
      res |= (b & 0x1F) << shift
      shift += 5
      if b < 0x20:
        break
    lng += (~(res >> 1)) if (res & 1) else (res >> 1)

    result.append((lat / 1e6, lng / 1e6))
  return result


class NavDaemon:
  def __init__(self):
    self.params = Params()
    self.mem_params = Params("/dev/shm/params")

    self.sm = messaging.SubMaster(['gpsLocationExternal'])
    self.pm = messaging.PubMaster(['navInstruction', 'navRoute'])

    # Valhalla — Actor takes a file path (str) to the config JSON
    import valhalla
    self.actor = valhalla.Actor(VALHALLA_CONFIG)

    # Route state
    self.maneuvers: list[dict] = []
    self.route_shape: list[tuple[float, float]] = []
    self.route_coordinates: list[Coordinate] = []
    self.current_maneuver_idx = 0
    self.route_valid = False

    # GPS state
    self.gps_history: list[dict] = []
    self.current_desire = log.Desire.none

    # Compute initial route
    self._compute_route(ROUTE_ORIGIN, ROUTE_DESTINATION)

  def _compute_route(self, origin: dict, destination: dict):
    """Compute route via Valhalla."""
    req = json.dumps({
      "locations": [origin, destination],
      "costing": "auto",
      "units": "km",
    })
    try:
      result = json.loads(self.actor.route(req))
      leg = result["trip"]["legs"][0]
      self.maneuvers = leg["maneuvers"]
      self.route_shape = decode_polyline6(leg["shape"])
      self.route_coordinates = [Coordinate(lat, lon) for lat, lon in self.route_shape]
      self.current_maneuver_idx = 0
      self.route_valid = True

      # Publish navRoute
      self._publish_nav_route()

      cloudlog.info(f"navd: route computed: {len(self.maneuvers)} maneuvers, "
                    f"{len(self.route_shape)} shape points")
      for i, m in enumerate(self.maneuvers):
        desire = MANEUVER_TO_DESIRE.get(m["type"], log.Desire.none)
        cloudlog.info(f"  [{i}] type={m['type']} desire={desire} "
                      f"len={m.get('length', 0):.3f}km {m.get('instruction', '')}")
    except Exception as e:
      cloudlog.error(f"navd: route computation failed: {e}")
      self.route_valid = False

  def _publish_nav_route(self):
    """Publish navRoute with route coordinate list."""
    msg = messaging.new_message('navRoute')
    coords = msg.navRoute.init('coordinates', len(self.route_shape))
    for i, (lat, lon) in enumerate(self.route_shape):
      coords[i].latitude = lat
      coords[i].longitude = lon
    self.pm.send('navRoute', msg)

  def _map_match(self, lat: float, lon: float) -> Coordinate:
    """Snap GPS position to road using Valhalla trace_route."""
    self.gps_history.append({"lat": lat, "lon": lon})
    if len(self.gps_history) > GPS_HISTORY_SIZE:
      self.gps_history.pop(0)

    if len(self.gps_history) < 2:
      return Coordinate(lat, lon)

    try:
      req = json.dumps({
        "shape": self.gps_history,
        "costing": "auto",
        "shape_match": "map_snap",
      })
      result = json.loads(self.actor.trace_route(req))
      # trace_route returns matched points in the trip shape
      matched = result.get("matched_points", [])
      if matched:
        last = matched[-1]
        if last.get("type") == "matched":
          return Coordinate(last["lat"], last["lon"])
    except Exception as e:
      cloudlog.debug(f"navd: map matching failed, using raw GPS: {e}")

    return Coordinate(lat, lon)

  def _find_current_maneuver(self, pos: Coordinate) -> tuple[int, float]:
    """Find the next maneuver ahead and distance to it along the route.

    Returns (maneuver_index, distance_to_maneuver_start_meters).
    """
    if not self.route_valid or not self.maneuvers:
      return 0, float('inf')

    # Find closest point on route shape
    min_dist = float('inf')
    closest_idx = 0
    for i, coord in enumerate(self.route_coordinates):
      d = pos.distance_to(coord)
      if d < min_dist:
        min_dist = d
        closest_idx = i

    # Check if we need to reroute (too far from route)
    if min_dist > REROUTE_DISTANCE:
      cloudlog.warning(f"navd: {min_dist:.0f}m off route, rerouting")
      self._compute_route(
        {"lat": pos.latitude, "lon": pos.longitude},
        ROUTE_DESTINATION,
      )
      return 0, float('inf')

    # Find next maneuver that is ahead of current position
    for i in range(self.current_maneuver_idx, len(self.maneuvers)):
      maneuver_start_idx = self.maneuvers[i]["begin_shape_index"]

      if maneuver_start_idx >= closest_idx:
        # Sum distance along route segments from current to maneuver start
        dist = 0.0
        for j in range(closest_idx, min(maneuver_start_idx, len(self.route_coordinates) - 1)):
          dist += self.route_coordinates[j].distance_to(self.route_coordinates[j + 1])

        self.current_maneuver_idx = i
        return i, dist

    # Past all maneuvers - at destination
    return len(self.maneuvers) - 1, 0.0

  def _compute_desire(self, maneuver_idx: int, distance: float) -> log.Desire:
    """Determine desire based on upcoming maneuver and distance to it."""
    if maneuver_idx >= len(self.maneuvers):
      return log.Desire.none

    mtype = self.maneuvers[maneuver_idx]["type"]
    desire = MANEUVER_TO_DESIRE.get(mtype, log.Desire.none)

    if desire == log.Desire.none:
      return log.Desire.none

    # Check distance threshold
    arm_dist = KEEP_DESIRE_ARM_DISTANCE if mtype in KEEP_MANEUVER_TYPES else TURN_DESIRE_ARM_DISTANCE
    if distance <= arm_dist:
      return desire

    return log.Desire.none

  def _publish_nav_instruction(self, maneuver_idx: int, distance: float,
                                total_remaining: float):
    """Publish navInstruction cereal message."""
    msg = messaging.new_message('navInstruction', valid=self.route_valid)
    inst = msg.navInstruction

    if maneuver_idx < len(self.maneuvers):
      m = self.maneuvers[maneuver_idx]
      inst.maneuverPrimaryText = m.get("instruction", "")
      inst.maneuverSecondaryText = ", ".join(m.get("street_names", []))
      inst.maneuverDistance = distance
      inst.maneuverType = str(m["type"])
      inst.distanceRemaining = total_remaining
      inst.timeRemaining = 0.0
      inst.showFull = distance < 300.0

      # Populate allManeuvers for lookahead
      remaining = self.maneuvers[maneuver_idx:]
      all_m = inst.init('allManeuvers', len(remaining))
      for i, rm in enumerate(remaining):
        all_m[i].distance = rm.get("length", 0.0) * 1000.0  # km -> m
        all_m[i].type = str(rm["type"])
        all_m[i].modifier = ""

    self.pm.send('navInstruction', msg)

  def _write_desire_param(self, desire: int, distance: float):
    """Write NavDesire to /dev/shm/params for desire_helper to read."""
    self.mem_params.put("NavDesire", json.dumps({
      "desire": int(desire),
      "distance": distance,
      "timestamp": time.monotonic(),
    }))

  def tick(self):
    """Main 1Hz tick: read GPS, match to route, compute desire."""
    self.sm.update(0)

    if not self.sm.updated['gpsLocationExternal']:
      return

    gps = self.sm['gpsLocationExternal']
    if not gps.hasFix:
      self._write_desire_param(log.Desire.none, 0.0)
      return

    lat = gps.latitude
    lon = gps.longitude

    # Map-match GPS to road network
    matched = self._map_match(lat, lon)

    # Find current position on route and next maneuver
    maneuver_idx, distance = self._find_current_maneuver(matched)

    # Compute desire from maneuver type + distance
    desire = self._compute_desire(maneuver_idx, distance)
    self.current_desire = desire

    # Calculate total remaining distance
    total_remaining = 0.0
    if maneuver_idx < len(self.maneuvers):
      for m in self.maneuvers[maneuver_idx:]:
        total_remaining += m.get("length", 0.0) * 1000.0

    # Publish and write params
    self._publish_nav_instruction(maneuver_idx, distance, total_remaining)
    self._write_desire_param(desire, distance)

    desire_name = {0: "none", 1: "turnLeft", 2: "turnRight",
                   3: "lcLeft", 4: "lcRight", 5: "keepLeft", 6: "keepRight"}.get(int(desire), "?")
    cloudlog.info(f"navd: ({matched.latitude:.6f},{matched.longitude:.6f}) "
                  f"maneuver {maneuver_idx}/{len(self.maneuvers)} "
                  f"dist={distance:.0f}m desire={desire_name}")


def main():
  cloudlog.info("navd: starting")
  daemon = NavDaemon()
  rk = Ratekeeper(1.0, print_delay_threshold=None)

  cloudlog.info("navd: running")
  while True:
    daemon.tick()
    rk.keep_time()


if __name__ == "__main__":
  main()
