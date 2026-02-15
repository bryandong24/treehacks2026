"""Paho MQTT publish/subscribe wrapper.

Uses the existing from-X/description topic naming convention.
Publishes to: from-server/coc-reasoning, from-server/trajectory, from-server/hail-response
Subscribes to: from-phone/hail-request, from-phone/command-car, from-car/gps-info
"""

import json
import logging

import paho.mqtt.client as mqtt

from . import config

logger = logging.getLogger(__name__)

# Topics we publish to
TOPIC_COC_REASONING = "from-server/coc-reasoning"
TOPIC_TRAJECTORY = "from-server/trajectory"
TOPIC_HAIL_RESPONSE = "from-server/hail-response"
TOPIC_COMMAND_CAR = "from-server/command-car"

# Topics we subscribe to
TOPIC_HAIL_REQUEST = "from-phone/hail-request"
TOPIC_PHONE_COMMAND = "from-phone/command-car"
TOPIC_GPS_INFO = "from-car/gps-info"


class MQTTClient:
    """Wrapper around paho MQTT for server publish/subscribe."""

    def __init__(self):
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._latest_gps: dict | None = None

    def connect(self):
        """Connect to the Mosquitto broker."""
        logger.info("Connecting to MQTT broker at %s:%d", config.MQTT_HOST, config.MQTT_PORT)
        self._client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=60)
        self._client.loop_start()

    def disconnect(self):
        """Disconnect from broker."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT disconnected.")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("MQTT connected (rc=%s)", rc)
        client.subscribe(TOPIC_HAIL_REQUEST)
        client.subscribe(TOPIC_PHONE_COMMAND)
        client.subscribe(TOPIC_GPS_INFO)

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Invalid JSON on topic %s", topic)
            return

        if topic == TOPIC_HAIL_REQUEST:
            self._handle_hail_request(payload)
        elif topic == TOPIC_GPS_INFO:
            self._latest_gps = payload
        elif topic == TOPIC_PHONE_COMMAND:
            self._handle_phone_command(payload)

    def _handle_hail_request(self, payload: dict):
        """Auto-accept hail requests and respond."""
        user_id = payload.get("user_id", "unknown")
        logger.info("Hail request from user %s — auto-accepting", user_id)
        response = {
            "user_id": user_id,
            "status": "accepted",
            "message": "Car is on its way!",
        }
        if self._latest_gps:
            response["car_gps"] = self._latest_gps
        self.publish(TOPIC_HAIL_RESPONSE, response)

    def _handle_phone_command(self, payload: dict):
        """Forward destination/command from phone to car."""
        logger.info("Phone command received, forwarding to car: %s", payload)
        self.publish(TOPIC_COMMAND_CAR, payload)

    def publish(self, topic: str, payload: dict):
        """Publish a JSON message to a topic."""
        self._client.publish(topic, json.dumps(payload))
        logger.debug("Published to %s", topic)

    def publish_coc(self, coc: str, trajectory_summary: str):
        """Publish Chain-of-Causation reasoning result."""
        self.publish(TOPIC_COC_REASONING, {
            "coc": coc,
            "trajectory_summary": trajectory_summary,
        })

    def publish_trajectory(self, pred_xyz: list):
        """Publish predicted trajectory waypoints."""
        self.publish(TOPIC_TRAJECTORY, {
            "pred_xyz": pred_xyz,
        })

    @property
    def latest_gps(self) -> dict | None:
        return self._latest_gps
