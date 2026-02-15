import paho.mqtt.client as mqtt
import serial
import threading

BROKER = "34.134.81.0"
PORT = 1883
SUBSCRIBE_TOPIC = "from-phone/command-car"
PUBLISH_TOPIC = "from-car/gps-info"

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 460800


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to MQTT broker at {BROKER}:{PORT}")
        client.subscribe(SUBSCRIBE_TOPIC)
        print(f"Subscribed to topic: {SUBSCRIBE_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")


def serial_reader(client):
    """Read lines from /dev/ttyACM0 and publish to MQTT."""
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            print(f"Opened serial port {SERIAL_PORT} at {SERIAL_BAUD} baud")
            while True:
                line = ser.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        print(f"[GPS] {decoded}")
                        client.publish(PUBLISH_TOPIC, decoded)
        except serial.SerialException as e:
            print(f"Serial error: {e} — retrying in 2s...")
            import time
            time.sleep(2)
        except Exception as e:
            print(f"Unexpected error reading serial: {e} — retrying in 2s...")
            import time
            time.sleep(2)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)

    # Start serial reader in a background thread
    gps_thread = threading.Thread(target=serial_reader, args=(client,), daemon=True)
    gps_thread.start()
    print(f"Started GPS serial reader on {SERIAL_PORT}, publishing to {PUBLISH_TOPIC}")

    client.loop_forever()


if __name__ == "__main__":
    main()
