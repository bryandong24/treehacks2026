#!/bin/bash
set -e

VENV=/home/subha/treehacks2026/sunnypilot/.venv
VALHALLA_DIR=/home/subha/treehacks2026/sunnypilot/data/valhalla
VALHALLA_BIN=$VENV/lib/python3.12/site-packages/valhalla/bin

# Stanford area bounding box (SW: 37.400, -122.230 to NE: 37.450, -122.120)
BBOX_OVERPASS="37.400,-122.230,37.450,-122.120"

mkdir -p "$VALHALLA_DIR" "$VALHALLA_DIR/tiles"

echo "=== Step 1: Download Stanford area OSM via Overpass API ==="
STANFORD_OSM="$VALHALLA_DIR/stanford.osm"
if [ ! -f "$STANFORD_OSM" ]; then
    echo "Downloading Stanford area roads from Overpass API..."
    curl -s -o "$STANFORD_OSM" --max-time 120 \
        "https://overpass-api.de/api/interpreter" \
        --data-urlencode "data=[out:xml][timeout:120][bbox:${BBOX_OVERPASS}];(way[\"highway\"];node(w);relation[\"type\"=\"restriction\"](bw););out body;>;out skel qt;"
    echo "Downloaded: $(du -h "$STANFORD_OSM" | cut -f1)"
else
    echo "Stanford OSM already cached: $(du -h "$STANFORD_OSM" | cut -f1)"
fi

echo "=== Step 2: Sort OSM data (required for PBF conversion) ==="
STANFORD_SORTED_PBF="$VALHALLA_DIR/stanford_sorted.osm.pbf"
$VENV/bin/python3 << 'PYEOF'
import xml.etree.ElementTree as ET
import os

VDIR = os.environ.get("VALHALLA_DIR", "/home/subha/treehacks2026/sunnypilot/data/valhalla")
tree = ET.parse(f"{VDIR}/stanford.osm")
root = tree.getroot()

nodes, ways, relations, other = [], [], [], []
for elem in root:
    {"node": nodes, "way": ways, "relation": relations}.get(elem.tag, other).append(elem)

print(f"  Nodes: {len(nodes)}, Ways: {len(ways)}, Relations: {len(relations)}")
nodes.sort(key=lambda e: int(e.get("id", 0)))
ways.sort(key=lambda e: int(e.get("id", 0)))
relations.sort(key=lambda e: int(e.get("id", 0)))

root.clear()
root.set("version", "0.6")
root.set("generator", "stanford_sort")
for lst in [other, nodes, ways, relations]:
    for e in lst:
        root.append(e)

sorted_path = f"{VDIR}/stanford_sorted.osm"
tree.write(sorted_path, encoding="unicode", xml_declaration=True)
print(f"  Sorted: {os.path.getsize(sorted_path)/1e6:.1f} MB")
PYEOF

echo "=== Step 3: Convert to PBF ==="
# Compile osmconvert if needed
if ! command -v "$VENV/bin/osmconvert" &>/dev/null; then
    echo "Compiling osmconvert..."
    cd /tmp
    wget -q "https://gitlab.com/osm-c-tools/osmctools/-/raw/master/src/osmconvert.c" -O osmconvert.c
    gcc -O2 -o osmconvert osmconvert.c -lz
    cp osmconvert "$VENV/bin/"
    cd -
fi

"$VENV/bin/osmconvert" "$VALHALLA_DIR/stanford_sorted.osm" \
    --out-pbf -o="$STANFORD_SORTED_PBF" 2>&1 || true
echo "PBF: $(du -h "$STANFORD_SORTED_PBF" | cut -f1)"

echo "=== Step 4: Generate Valhalla config ==="
$VENV/bin/python3 << 'PYEOF'
import valhalla, json, os
VDIR = os.environ.get("VALHALLA_DIR", "/home/subha/treehacks2026/sunnypilot/data/valhalla")
config = valhalla.get_config(tile_dir=f"{VDIR}/tiles", tile_extract="", verbose=True)
with open(f"{VDIR}/valhalla.json", "w") as f:
    json.dump(config, f)
print(f"Config: {VDIR}/valhalla.json")
PYEOF

echo "=== Step 5: Build Valhalla tiles ==="
"$VALHALLA_BIN/valhalla_build_tiles" -c "$VALHALLA_DIR/valhalla.json" -- "$STANFORD_SORTED_PBF" 2>&1 | tail -5

echo "=== Step 6: Verify routing ==="
$VENV/bin/python3 << 'PYEOF'
import valhalla, json, os
VDIR = os.environ.get("VALHALLA_DIR", "/home/subha/treehacks2026/sunnypilot/data/valhalla")
actor = valhalla.Actor(f"{VDIR}/valhalla.json")

result = json.loads(actor.route(json.dumps({
    "locations": [
        {"lat": 37.425611, "lon": -122.177434},
        {"lat": 37.429649, "lon": -122.170194},
    ],
    "costing": "auto", "units": "km",
})))

leg = result["trip"]["legs"][0]
print(f"Route: {leg['summary']['length']:.2f} km, {len(leg['maneuvers'])} maneuvers")
for j, m in enumerate(leg["maneuvers"]):
    print(f"  [{j}] type={m['type']:2d} {m.get('instruction', '')}")
print("\nValhalla setup complete!")
PYEOF

echo "=== Done ==="
