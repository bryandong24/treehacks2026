import paho.mqtt.client as mqtt
import threading
import json
import time
import os

BROKER = "34.134.81.0"
PORT = 1883
SUBSCRIBE_TOPIC = "from-phone/command-car"
GPS_TOPIC = "from-car/gps-info"
NAV_STATUS_TOPIC = "from-car/nav-status"

NAV_DESTINATION_PATH = "/dev/shm/nav_destination.json"
NAV_PROGRESS_PATH = "/dev/shm/nav_progress.json"
LAST_GPS_POSITION_PATH = "/dev/shm/params/d/LastGPSPosition"
GPS_POLL_RATE = 1.0       # seconds
PROGRESS_POLL_RATE = 1.0  # seconds — match navd 1Hz rate


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to MQTT broker at {BROKER}:{PORT}")
        client.subscribe(SUBSCRIBE_TOPIC)
        print(f"Subscribed to topic: {SUBSCRIBE_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    print(f"[{msg.topic}] {payload}")

    try:
        data = json.loads(payload)
        dest = data.get("destination")
        if dest and "latitude" in dest and "longitude" in dest:
            nav_dest = json.dumps({
                "latitude": dest["latitude"],
                "longitude": dest["longitude"],
                "place_name": dest.get("name", ""),
                "timestamp": time.time(),
            })
            # Atomic write: write to temp file then rename (safe for concurrent readers)
            tmp_path = NAV_DESTINATION_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(nav_dest)
            os.rename(tmp_path, NAV_DESTINATION_PATH)
            print(f"[NAV] Set destination: {nav_dest}")
    except Exception as e:
        print(f"[NAV] Error parsing command: {e}")


def gps_reader(client):
    """Poll LastGPSPosition written by sunnypilot's serial_gps, publish to MQTT."""
    last_timestamp = 0.0

    while True:
        try:
            with open(LAST_GPS_POSITION_PATH, "r") as f:
                raw = f.read().strip()
            if not raw or raw == "{}":
                time.sleep(GPS_POLL_RATE)
                continue
            data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(GPS_POLL_RATE)
            continue

        ts = data.get("timestamp", 0.0)
        if ts == last_timestamp:
            time.sleep(GPS_POLL_RATE)
            continue
        last_timestamp = ts

        gps_payload = json.dumps({
            "latitude": data["latitude"],
            "longitude": data["longitude"],
            "bearing": data.get("bearing", 0.0),
            "speed": data.get("speed", 0.0),
            "timestamp": ts,
        })
        client.publish(GPS_TOPIC, gps_payload)
        print(f"[GPS] {gps_payload}")

        time.sleep(GPS_POLL_RATE)


def progress_reader(client):
    """Poll nav_progress.json written by navd, publish nav status to MQTT."""
    last_timestamp = 0.0

    while True:
        try:
            with open(NAV_PROGRESS_PATH, "r") as f:
                data = json.loads(f.read())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(PROGRESS_POLL_RATE)
            continue

        ts = data.get("timestamp", 0.0)
        if ts == last_timestamp:
            time.sleep(PROGRESS_POLL_RATE)
            continue
        last_timestamp = ts

        # Publish navigation status
        nav = data.get("navigation", {})
        nav_payload = json.dumps({
            "route_valid": nav.get("route_valid", False),
            "arrived": nav.get("arrived", False),
            "destination": nav.get("destination"),
            "destination_name": nav.get("destination_name", ""),
            "maneuver_index": nav.get("maneuver_index", 0),
            "total_maneuvers": nav.get("total_maneuvers", 0),
            "distance_to_next_maneuver_m": nav.get("distance_to_next_maneuver_m", 0),
            "distance_remaining_m": nav.get("distance_remaining_m", 0),
            "current_instruction": nav.get("current_instruction", ""),
            "timestamp": ts,
        })
        client.publish(NAV_STATUS_TOPIC, nav_payload)
        print(f"[NAV] {nav_payload}")

        time.sleep(PROGRESS_POLL_RATE)
# s
# s
def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)

    # Start GPS reader — reads LastGPSPosition from sunnypilot's serial_gps via /dev/shm
    # (only one process can read /dev/ttyACM0, so we read the param file instead)
    gps_thread = threading.Thread(target=gps_reader, args=(client,), daemon=True)
    gps_thread.start()
    print(f"Started GPS reader, polling {LAST_GPS_POSITION_PATH}")
    print(f"Publishing GPS to {GPS_TOPIC}")

    # Start progress reader — polls navd output, publishes nav status to MQTT
    progress_thread = threading.Thread(target=progress_reader, args=(client,), daemon=True)
    progress_thread.start()
    print(f"Started progress reader, polling {NAV_PROGRESS_PATH}")
    print(f"Publishing nav status to {NAV_STATUS_TOPIC}")

    client.loop_forever()


if __name__ == "__main__":
    main()
