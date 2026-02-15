# API Reference — Alpamayo Inference Server

H100 server address: `<H100_IP>:8000`
MQTT broker: `<H100_IP>:1883`

---

## Thor (AGX Thor / Sunnypilot) Integration

### WebSocket: `ws://<H100_IP>:8000/ws/thor`

The Thor sends camera frames + ego-motion data over a WebSocket at ~10Hz using **msgpack binary** encoding.

#### Message Format

```python
{
    "frame_jpeg": bytes,       # JPEG-encoded road camera frame
    "ego": {
        "timestamp": float,    # Unix timestamp in seconds
        "orientation_ned": [float, float, float, float],  # Quaternion [w, x, y, z]
        "velocity_device": [float, float, float],         # Velocity in device frame [vx, vy, vz] m/s
    }
}
```

#### Field Mapping from cereal `livePose`

| cereal livePose field         | Our ego field         | Notes                           |
|-------------------------------|-----------------------|---------------------------------|
| `livePose.orientationNED`     | `orientation_ned`     | Quaternion [w, x, y, z]        |
| `livePose.velocityDevice`     | `velocity_device`     | [vx, vy, vz] in m/s           |
| `logMonoTime` (converted)     | `timestamp`           | Convert ns to seconds           |

#### Reference Implementation (Thor Side)

```python
#!/usr/bin/env python3
"""Thor-side bridge: streams camera frames + ego data to the H100 server."""

import asyncio
import time

import cv2
import msgpack
import websockets
from cereal import messaging

H100_SERVER = "ws://<H100_IP>:8000/ws/thor"


async def stream_to_server():
    sm = messaging.SubMaster(["livePose"])

    # VisionIPC for road camera
    from cereal.visionipc import VisionIpcClient, VisionStreamType
    vipc = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    while not vipc.connect(False):
        await asyncio.sleep(0.1)

    async with websockets.connect(H100_SERVER, max_size=4 * 1024 * 1024) as ws:
        while True:
            buf = vipc.recv()
            if buf is None:
                await asyncio.sleep(0.01)
                continue

            # Encode frame as JPEG
            frame = buf.data.reshape(buf.height, buf.width, 3)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Read latest livePose
            sm.update(0)
            lp = sm["livePose"]

            msg = {
                "frame_jpeg": jpeg.tobytes(),
                "ego": {
                    "timestamp": time.time(),
                    "orientation_ned": [
                        lp.orientationNED.w,
                        lp.orientationNED.x,
                        lp.orientationNED.y,
                        lp.orientationNED.z,
                    ],
                    "velocity_device": [
                        lp.velocityDevice.x,
                        lp.velocityDevice.y,
                        lp.velocityDevice.z,
                    ],
                },
            }

            await ws.send(msgpack.packb(msg, use_bin_type=True))
            await asyncio.sleep(0.1)  # ~10 Hz


if __name__ == "__main__":
    asyncio.run(stream_to_server())
```

### MQTT Topics (Thor)

The Thor should also publish car status and subscribe to commands:

| Direction | Topic                   | Payload (JSON)                        |
|-----------|-------------------------|---------------------------------------|
| Publish   | `from-car/gps-info`     | `{"lat": float, "lng": float, "speed_mph": float, "heading": float}` |
| Subscribe | `from-server/command-car`| `{"destination": {"lat": float, "lng": float}, "action": str}` |

---

## Mobile App Integration

### MQTT Topics

Connect to the MQTT broker at `<H100_IP>:1883`.

#### Subscribe To

| Topic                         | Payload (JSON)                                | Description                    |
|-------------------------------|-----------------------------------------------|--------------------------------|
| `from-server/coc-reasoning`  | `{"coc": str, "trajectory_summary": str}`     | Chain-of-Causation reasoning from Alpamayo (every ~5s) |
| `from-server/trajectory`     | `{"pred_xyz": [[x,y,z], ...]}`                | 64 predicted waypoints (6.4s horizon, 0.1s spacing) |
| `from-server/hail-response`  | `{"user_id": str, "status": str, "message": str, "car_gps": {...}}` | Response to hail request |
| `from-car/gps-info`          | `{"lat": float, "lng": float, "speed_mph": float, "heading": float}` | Live car GPS (existing topic) |

#### Publish To

| Topic                         | Payload (JSON)                                | Description                    |
|-------------------------------|-----------------------------------------------|--------------------------------|
| `from-phone/hail-request`   | `{"user_id": str}`                            | Request the car to come to you |
| `from-phone/command-car`     | `{"destination": {"lat": float, "lng": float}, "action": str}` | Send destination/command (existing topic) |

#### Reference: Connect & Subscribe (JavaScript)

```javascript
import mqtt from "mqtt";

const client = mqtt.connect("mqtt://<H100_IP>:1883");

client.on("connect", () => {
  console.log("Connected to MQTT broker");
  client.subscribe("from-server/coc-reasoning");
  client.subscribe("from-server/trajectory");
  client.subscribe("from-server/hail-response");
  client.subscribe("from-car/gps-info");
});

client.on("message", (topic, message) => {
  const data = JSON.parse(message.toString());

  switch (topic) {
    case "from-server/coc-reasoning":
      // Display Chain-of-Causation reasoning
      console.log("CoC:", data.coc);
      console.log("Trajectory summary:", data.trajectory_summary);
      break;
    case "from-server/trajectory":
      // Render predicted trajectory on map
      console.log("Waypoints:", data.pred_xyz.length);
      break;
    case "from-server/hail-response":
      console.log("Hail response:", data.status, data.message);
      break;
    case "from-car/gps-info":
      // Update car marker on map
      console.log("Car at:", data.lat, data.lng);
      break;
  }
});
```

#### Reference: Hail the Car

```javascript
function hailCar(userId) {
  client.publish(
    "from-phone/hail-request",
    JSON.stringify({ user_id: userId })
  );
}
```

#### Reference: Send Destination

```javascript
function setDestination(lat, lng) {
  client.publish(
    "from-phone/command-car",
    JSON.stringify({
      destination: { lat, lng },
      action: "navigate",
    })
  );
}
```

### WebSocket: Live Video Feed

Connect to `ws://<H100_IP>:8000/ws/mobile/video` to receive a live JPEG stream from the car at ~5 FPS.

Each message is raw JPEG bytes. Display example:

```javascript
const ws = new WebSocket("ws://<H100_IP>:8000/ws/mobile/video");
ws.binaryType = "arraybuffer";

ws.onmessage = (event) => {
  const blob = new Blob([event.data], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  document.getElementById("video-feed").src = url;
};
```

---

## Server Health Check

```
GET http://<H100_IP>:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "frames_buffered": 42,
  "ego_samples": 150
}
```

---

## Starting the Server

```bash
cd mqtt-server
PYTHONPATH=/path/to/alpamayo/src uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
# Terminal 1: Subscribe to CoC output
mosquitto_sub -t "from-server/coc-reasoning"

# Terminal 2: Run mock client
python mock_thor_client.py

# Terminal 3: Test hailing
mosquitto_pub -t "from-phone/hail-request" -m '{"user_id":"test"}'

# Check hail response
mosquitto_sub -t "from-server/hail-response"
```
