import paho.mqtt.client as mqtt
import subprocess
import threading
import json
import time

BROKER = "34.134.81.0"
PORT = 1883
SUBSCRIBE_TOPIC = "from-phone/command-car"
PUBLISH_TOPIC = "from-car/gps-info"

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = "460800"


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to MQTT broker at {BROKER}:{PORT}")
        client.subscribe(SUBSCRIBE_TOPIC)
        print(f"Subscribed to topic: {SUBSCRIBE_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")


def gps_reader(client):
    """Run listen.sh approach: stty + cat, parse output, publish to MQTT."""
    while True:
        try:
            # Configure the serial port just like listen.sh
            subprocess.run(
                ["stty", "-F", SERIAL_PORT, SERIAL_BAUD, "raw", "-echo"],
                check=True,
            )
            print(f"Configured {SERIAL_PORT} at {SERIAL_BAUD} baud (raw)")

            # Stream cat output line by line
            proc = subprocess.Popen(
                ["cat", SERIAL_PORT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f"Reading from {SERIAL_PORT}...")

            for raw_line in proc.stdout:
                decoded = raw_line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    continue
                parts = decoded.split(",")
                if len(parts) >= 2:
                    try:
                        payload = json.dumps({
                            "latitude": float(parts[0]),
                            "longitude": float(parts[1]),
                            "timestamp": time.time(),
                        })
                        print(f"[GPS] {payload}")
                        client.publish(PUBLISH_TOPIC, payload)
                    except ValueError:
                        print(f"[GPS] skipping bad line: {decoded}")

            proc.wait()
            print(f"cat process exited with code {proc.returncode}, restarting...")
        except Exception as e:
            print(f"GPS reader error: {e} — retrying in 2s...")
            time.sleep(2)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)

    # Start GPS reader in a background thread
    gps_thread = threading.Thread(target=gps_reader, args=(client,), daemon=True)
    gps_thread.start()
    print(f"Started GPS reader ({SERIAL_PORT}), publishing to {PUBLISH_TOPIC}")

    client.loop_forever()


if __name__ == "__main__":
    main()
