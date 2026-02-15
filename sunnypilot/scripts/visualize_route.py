#!/usr/bin/env python3
"""
Visualize the hardcoded navigation route on an interactive map.

Queries Valhalla for the route between ROUTE_ORIGIN and ROUTE_DESTINATION,
then renders the route polyline, maneuver markers, and turn-by-turn
instructions onto a Leaflet map via folium.

Usage:
  cd /home/subha/treehacks2026/sunnypilot
  .venv/bin/python scripts/visualize_route.py

Output:
  scripts/route_map.html  (open in a browser)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import folium
from folium import plugins

VALHALLA_CONFIG = "/home/subha/treehacks2026/sunnypilot/data/valhalla/valhalla.json"
ROUTE_ORIGIN = {"lat": 37.425611, "lon": -122.177434}
ROUTE_DESTINATION = {"lat": 37.429649, "lon": -122.170194}

# Maneuver type names from Valhalla API reference
MANEUVER_TYPE_NAMES = {
    0: "None", 1: "Start", 2: "StartRight", 3: "StartLeft",
    4: "Destination", 5: "DestinationRight", 6: "DestinationLeft",
    7: "Becomes", 8: "Continue", 9: "SlightRight", 10: "Right",
    11: "SharpRight", 12: "UturnRight", 13: "UturnLeft", 14: "SharpLeft",
    15: "Left", 16: "SlightLeft", 17: "RampStraight", 18: "RampRight",
    19: "RampLeft", 20: "ExitRight", 21: "ExitLeft", 22: "StayStraight",
    23: "StayRight", 24: "StayLeft", 25: "Merge", 26: "RoundaboutEnter",
    27: "RoundaboutExit", 28: "FerryEnter", 29: "FerryExit",
    37: "MergeRight", 38: "MergeLeft",
}

# Desire mapping (same as navd.py)
DESIRE_NAMES = {
    9: "turnRight", 10: "turnRight", 11: "turnRight", 12: "turnRight",
    13: "turnLeft", 14: "turnLeft", 15: "turnLeft", 16: "turnLeft",
    18: "keepRight", 20: "keepRight", 23: "keepRight", 37: "keepRight",
    19: "keepLeft", 21: "keepLeft", 24: "keepLeft", 38: "keepLeft",
}

# Colors for desire types
DESIRE_COLORS = {
    "turnRight": "red",
    "turnLeft": "blue",
    "keepRight": "orange",
    "keepLeft": "purple",
}


def decode_polyline6(encoded: str) -> list[tuple[float, float]]:
    """Decode Valhalla encoded polyline (precision 6) to list of (lat, lon)."""
    result = []
    index = 0
    lat = 0
    lng = 0
    while index < len(encoded):
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


def main():
    import valhalla

    actor = valhalla.Actor(VALHALLA_CONFIG)

    # Compute route
    req = json.dumps({
        "locations": [ROUTE_ORIGIN, ROUTE_DESTINATION],
        "costing": "auto",
        "units": "km",
    })
    result = json.loads(actor.route(req))
    leg = result["trip"]["legs"][0]
    maneuvers = leg["maneuvers"]
    shape = decode_polyline6(leg["shape"])
    summary = leg["summary"]

    print(f"Route: {summary['length']:.2f} km, ~{summary['time']:.0f}s")
    print(f"Shape: {len(shape)} points, {len(maneuvers)} maneuvers\n")

    # Print maneuvers
    for i, m in enumerate(maneuvers):
        mtype = m["type"]
        name = MANEUVER_TYPE_NAMES.get(mtype, f"Unknown({mtype})")
        desire = DESIRE_NAMES.get(mtype, "none")
        streets = ", ".join(m.get("street_names", []))
        length_m = m.get("length", 0) * 1000
        instruction = m.get("instruction", "")
        begin_idx = m.get("begin_shape_index", 0)
        end_idx = m.get("end_shape_index", 0)
        print(f"  [{i}] {name:15s} desire={desire:10s} "
              f"len={length_m:6.0f}m shape[{begin_idx}-{end_idx}] "
              f"| {instruction}")
        if streets:
            print(f"       streets: {streets}")

    # ── Build map ──
    center_lat = (ROUTE_ORIGIN["lat"] + ROUTE_DESTINATION["lat"]) / 2
    center_lon = (ROUTE_ORIGIN["lon"] + ROUTE_DESTINATION["lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=16,
                   tiles="OpenStreetMap")

    # Route polyline
    route_coords = [(lat, lon) for lat, lon in shape]
    folium.PolyLine(
        route_coords, weight=5, color="#2196F3", opacity=0.8,
        tooltip=f"Route: {summary['length']:.2f} km",
    ).add_to(m)

    # Origin marker
    folium.Marker(
        [ROUTE_ORIGIN["lat"], ROUTE_ORIGIN["lon"]],
        popup=folium.Popup(
            f"<b>ORIGIN</b><br>"
            f"lat: {ROUTE_ORIGIN['lat']:.6f}<br>"
            f"lon: {ROUTE_ORIGIN['lon']:.6f}",
            max_width=200,
        ),
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip="Origin",
    ).add_to(m)

    # Destination marker
    folium.Marker(
        [ROUTE_DESTINATION["lat"], ROUTE_DESTINATION["lon"]],
        popup=folium.Popup(
            f"<b>DESTINATION</b><br>"
            f"lat: {ROUTE_DESTINATION['lat']:.6f}<br>"
            f"lon: {ROUTE_DESTINATION['lon']:.6f}",
            max_width=200,
        ),
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        tooltip="Destination",
    ).add_to(m)

    # Maneuver markers (skip start/destination which are types 1-6)
    for i, man in enumerate(maneuvers):
        mtype = man["type"]
        if mtype in (1, 2, 3, 4, 5, 6):
            continue

        begin_idx = man.get("begin_shape_index", 0)
        if begin_idx >= len(shape):
            continue

        lat, lon = shape[begin_idx]
        name = MANEUVER_TYPE_NAMES.get(mtype, f"Unknown({mtype})")
        desire = DESIRE_NAMES.get(mtype, "none")
        instruction = man.get("instruction", "")
        length_m = man.get("length", 0) * 1000
        streets = ", ".join(man.get("street_names", []))

        color = DESIRE_COLORS.get(desire, "gray")

        # Icon based on desire
        if "Right" in desire or "Right" in name:
            icon_name = "arrow-right"
        elif "Left" in desire or "Left" in name:
            icon_name = "arrow-left"
        else:
            icon_name = "arrow-up"

        popup_html = (
            f"<b>Maneuver [{i}]</b><br>"
            f"<b>Type:</b> {name} ({mtype})<br>"
            f"<b>Desire:</b> {desire}<br>"
            f"<b>Instruction:</b> {instruction}<br>"
            f"<b>Length:</b> {length_m:.0f}m<br>"
        )
        if streets:
            popup_html += f"<b>Streets:</b> {streets}<br>"

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
            tooltip=f"[{i}] {name}: {instruction[:30]}",
        ).add_to(m)

    # Add shape points as small circles (every Nth for clarity)
    step = max(1, len(shape) // 50)
    for idx in range(0, len(shape), step):
        lat, lon = shape[idx]
        folium.CircleMarker(
            [lat, lon], radius=3, color="#666", fill=True,
            fill_opacity=0.5, weight=1,
            tooltip=f"shape[{idx}] ({lat:.6f}, {lon:.6f})",
        ).add_to(m)

    # Desire zones: highlight route segments where a desire would be active
    for i, man in enumerate(maneuvers):
        mtype = man["type"]
        desire = DESIRE_NAMES.get(mtype, "none")
        if desire == "none":
            continue

        begin_idx = man.get("begin_shape_index", 0)
        arm_dist = 150.0 if mtype in (18, 19, 20, 21, 23, 24, 37, 38) else 100.0

        # Walk backwards from maneuver start to find the arm point
        total_dist = 0.0
        arm_start_idx = begin_idx
        for j in range(begin_idx - 1, -1, -1):
            lat1, lon1 = shape[j]
            lat2, lon2 = shape[j + 1]
            d = _haversine(lat1, lon1, lat2, lon2)
            total_dist += d
            if total_dist >= arm_dist:
                arm_start_idx = j
                break

        # Draw the desire zone
        zone_coords = [(shape[k][0], shape[k][1])
                       for k in range(arm_start_idx, min(begin_idx + 1, len(shape)))]
        if len(zone_coords) >= 2:
            color = DESIRE_COLORS.get(desire, "gray")
            folium.PolyLine(
                zone_coords, weight=10, color=color, opacity=0.4,
                tooltip=f"Desire zone: {desire} ({arm_dist:.0f}m arm)",
                dash_array="10 5",
            ).add_to(m)

    # Fit bounds
    m.fit_bounds([(min(p[0] for p in shape), min(p[1] for p in shape)),
                  (max(p[0] for p in shape), max(p[1] for p in shape))])

    out_path = os.path.join(ROOT, "scripts", "route_map.html")
    m.save(out_path)
    print(f"\nMap saved to: {out_path}")
    print(f"Open in browser: file://{out_path}")


def _haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    import math
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


if __name__ == "__main__":
    main()
