# treehacks2026

Autonomous vehicle platform running on the **NVIDIA Jetson AGX Thor** to drive a **Honda Bosch** vehicle. Built at TreeHacks 2026.

## Hardware

- **Compute**: NVIDIA Jetson AGX Thor (JetPack 7.1, CUDA 13.0, L4T R38.4.0)
- **Cameras**: Two IMX274 cameras via Holoscan Sensor Bridge — 90° FOV (road) and 120° FOV (wide)
- **IMU**: LSM6DSOX via Arduino → USB-UART at 104 Hz
- **GPS**: Adafruit Ultimate GPS FeatherWing via USB-UART at 10 Hz
- **Vehicle**: Honda Bosch platform via red panda OBD-II adapter

## Architecture

```
Cameras → CUDA warp/YUV (CuPy) → driving_vision.onnx → driving_policy.onnx → controlsd → pandad → CAN bus → Honda
                                                                                    ↑
IMU (104Hz) → locationd (Kalman) → livePose ────────────────────────────────────────┘
GPS (10Hz)  → navd (Valhalla offline routing) → NavDesire → desire_helper ──────────┘
```

## What We Built

### ONNX Runtime on CUDA
- Built a compatible onnxruntime wheel for Jetson Thor (aarch64, CUDA 13.0) using [jetson-containers](https://github.com/dusty-nv/jetson-containers)
- All three models run on GPU with `ORT_ENABLE_ALL` graph optimization + `EXHAUSTIVE` cuDNN algo search

| Model | Avg Latency | Output |
|---|---|---|
| `driving_vision.onnx` | 7.06 ms | [1,1576] fp16 |
| `driving_policy.onnx` | 0.93 ms | [1,1000] fp16 |
| `dmonitoring_model.onnx` | 3.83 ms | [1,551] fp16 |
| **Total driving pipeline** | **~8 ms** | Well within 50ms (20Hz) budget |

### CUDA Preprocessing
Replaced the entire OpenCL pipeline with CuPy CUDA kernels:
- `warpPerspective` CUDA RawKernel — bilinear interpolation perspective warp
- `loadyuv` — YUV420 channel packing via CuPy array slicing
- `DrivingModelFrame` / `MonitoringModelFrame` — full temporal buffer management

### ONNX-Based Model Runner
Replaced tinygrad/OpenCL model runner with ONNX Runtime + CUDA sessions for both the driving model and driver monitoring model.

### Camera Integration
- Two IMX274 cameras via NVIDIA Holoscan Sensor Bridge running in Docker
- Shared memory ring buffer (`/dev/shm`) for zero-copy frame transfer from Docker → host
- RGBA uint16 → NV12 uint8 conversion via CuPy CUDA kernel
- Publishes via VisionIPC at 20 Hz

### IMU Integration
- LSM6DSOX IMU → Arduino (I2C) → CP2104 USB-UART → `/dev/IMU`
- Publishes accelerometer + gyroscope at 104 Hz
- Fully integrated with locationd Kalman filter: `sensorsOK`, `inputsOK`, `valid` all 100%

### GPS Integration
- Adafruit Ultimate GPS via USB-UART → `/dev/GPS`
- Publishes GPS location at 10 Hz with speed and bearing

### Navigation (Valhalla + NavDesire)
- Offline turn-by-turn routing using `pyvalhalla` with Stanford-area OSM data
- GPS map matching → maneuver tracking → desire inputs (turnRight, turnLeft, keepLeft, keepRight)
- Pipeline: GPS → navd → NavDesire → desire_helper → desire_pulse → driving_policy.onnx

### Mobile App — Ride Hailing ([`mobileApp/`](mobileApp/))
A native iOS app (SwiftUI) that lets users hail the autonomous vehicle, similar to the Waymo rider app. The phone and car communicate over MQTT via a VPS broker.

**Ride flow:**
1. User sets a destination and pickup location (search, recommended spots, or map pin)
2. App sends a `from-phone/command-car` MQTT message with pickup/destination coordinates
3. Car drives to pickup — app tracks the car's live GPS position on a map
4. Car sends `from-car/car-arrived` — user taps "Start Driving"
5. App sends `from-phone/start-ride` — car drives to destination
6. Car sends `from-car/ride-finished` — ride complete

**Features:**
- Real-time car location tracking via MQTT GPS updates
- MapKit route visualization and address autocomplete (scoped to Stanford campus)
- Ride phase UI (approaching → arrived → driving → reached destination)
- Car diagnostics tab with live MQTT message log and connection status
- Auto-reconnecting MQTT client (CocoaMQTT)

### Cloud Inference with Alpamayo ([`alpamayo/`](alpamayo/), [`mqtt-server/`](mqtt-server/))
We run [NVIDIA Alpamayo R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) (10B parameter Vision-Language-Action model) on a remote H100 GPU to provide high-level scene reasoning alongside the on-device driving stack.

**How it works:**
1. The Jetson Thor streams camera frames + ego-motion data (orientation, velocity from `livePose`) to the H100 server over a WebSocket at ~10 Hz
2. The server buffers frames and runs Alpamayo inference every few seconds
3. Alpamayo produces two outputs:
   - **Chain-of-Causation (CoC) reasoning** — a natural language explanation of what the car sees, why it's making decisions, and causal relationships between scene elements (e.g. "The lead vehicle is braking because a pedestrian is crossing, so I should decelerate")
   - **Trajectory prediction** — 64 waypoints over a 6.4s horizon at 10 Hz, generated via flow-matching diffusion conditioned on the VLM's reasoning
4. Results are published over MQTT (`from-server/coc-reasoning`, `from-server/trajectory`)
5. The mobile app subscribes to these topics and displays the CoC reasoning in real time, so the rider can see *why* the car is doing what it's doing

**Pipeline:**
```
Thor cameras + livePose → WebSocket (msgpack) → H100 server → Alpamayo R1 → CoC + trajectory → MQTT → mobile app
```

The server also relays a live JPEG video feed from the car to the app via a second WebSocket endpoint.

### Platform Integration
- Hardware detection via `/JETSON` marker file
- C++ pandad binary compiled for aarch64 — connects to red panda over USB
- All Cython extensions compiled for aarch64

## End-to-End Latency

| Stage | Avg | p95 |
|---|---|---|
| CUDA preprocess (both cams) | 1.81 ms | 2.22 ms |
| Vision ONNX inference | 5.33 ms | 8.40 ms |
| Policy ONNX inference | 1.22 ms | 1.74 ms |
| **Total driving pipeline** | **8.66 ms** | **12.34 ms** |
| Driver monitoring (preprocess + inference) | 4.15 ms | 4.39 ms |
| **Budget (20 Hz)** | **50 ms** | **~39 ms headroom** |

## Remaining

- **Honda Bosch CAN interface**: Connect red panda, verify CAN read/write with Honda ECUs
- **Full system integration test**: All subsystems running simultaneously on Jetson with live CAN

## MQTT Topics

```
Topic: from-phone/command-car
Payload: {"timestamp":...,"pickup":{"longitude":...,"latitude":...},"destination":{"name":"...","longitude":...,"latitude":...}}

Topic: from-phone/start-ride
Payload: null

Topic: from-car/ride-finished
```

## Running

```bash
sudo PYTHONPATH=/home/subha/.local/lib/python3.12/site-packages python3 main.py
```
