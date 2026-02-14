# Sunnypilot on NVIDIA Jetson AGX Thor — Project Context

## Goal
Port sunnypilot to run on the **NVIDIA Jetson AGX Thor** to drive a **Honda Bosch** vehicle. The stock sunnypilot runs on comma.ai hardware (Qualcomm Snapdragon) using tinygrad + OpenCL. We are replacing that with **ONNX Runtime + CUDA** on the Jetson Thor.

## Hardware
- **Compute**: NVIDIA Jetson AGX Thor (JetPack 7.1, CUDA 13.0, Driver 580.00, L4T R38.4.0, Tegra 6.8.12 kernel)
- **Cameras**: Two cameras — 120-degree FOV and 90-degree FOV
  - Sunnypilot expects:
    - `fcam` (forward/road camera): ~52-degree horizontal FOV (OX03C10/OS04C10 on comma hardware) — primary camera for lane detection, lead vehicles, road edges
    - `ecam` (extra/wide camera): ~120-degree horizontal FOV (AR0231 on comma hardware) — wider scene context for low speeds and turns
  - Our camera FOVs differ from comma hardware; calibration/warp matrices will need adjustment
- **IMU**: External IMU being set up (required — sunnypilot will not operate without accelerometer + gyroscope data at ~104 Hz)
- **GPS**: External GPS being set up
- **Vehicle**: Honda Bosch platform

## Architecture — Model Pipeline
Sunnypilot uses a two-stage driving model:

1. **`driving_vision.onnx`** (45MB) — Perception
   - Inputs: `img` [1,12,128,256] uint8, `big_img` [1,12,128,256] uint8
   - 12 channels = 2 consecutive frames x 6 YUV420 channels
   - Output: [1,1576] float16 — visual features including 512-dim hidden_state

2. **`driving_policy.onnx`** (14MB) — Planning
   - Inputs: features_buffer (100x512 temporal context), desire_pulse, traffic_convention
   - Output: Driving plan (trajectory, curvature, acceleration)

3. **`dmonitoring_model.onnx`** (6.7MB) — Driver monitoring
   - Input: 1440x960 Y-channel luminance + calibration angles
   - Output: Face pose, eye tracking, attention state

Flow: `Camera → [vision model] → features → [policy model] → steering/accel`

Stock code runs via tinygrad on Qualcomm. We are porting to ONNX Runtime with CUDAExecutionProvider.

## Key Files
- `selfdrive/modeld/modeld.py` — Main model runner (orchestrates vision + policy pipeline)
- `selfdrive/modeld/models/` — ONNX models and commonmodel C++/Cython code
- `selfdrive/modeld/runners/` — tinygrad execution helpers (to be replaced with ONNX)
- `selfdrive/modeld/parse_model_outputs.py` — Output tensor parsing
- `selfdrive/modeld/fill_model_msg.py` — Fills cereal messages from model output
- `selfdrive/modeld/constants.py` — Model constants (frequencies, dimensions)
- `system/sensord/sensord.py` — IMU/sensor data acquisition
- `selfdrive/locationd/` — Pose estimation (Kalman filter, requires IMU)

## Venv
- Path: `/home/subha/treehacks2026/sunnypilot/.venv`
- Python 3.12
- ONNX Runtime installed from custom-built wheel for Jetson Thor (aarch64)

## CUDA/GPU Setup
- CUDA 13.0 toolkit installed (`cuda-toolkit-13-0`)
- cuBLAS 13.0 (`libcublas-13-0`)
- cuDNN 9 for CUDA 13 (`cudnn9-cuda-13-0`)
- CUDAExecutionProvider confirmed working in onnxruntime
- TensorrtExecutionProvider also available

## Important: No OpenCL on Jetson Thor
The Jetson Thor does NOT support OpenCL. All OpenCL code in sunnypilot must be rewritten:
- Image perspective warp (camera → model input transform)
- YUV420 channel packing/unpacking
- These must become CUDA kernels or use equivalent CUDA/cuDNN operations

## IMU Requirement
Sunnypilot **will not operate** without IMU data:
- Requires accelerometer + gyroscope at ~104 Hz
- `locationd` declares `["accelerometer", "gyroscope", "cameraOdometry"]` as critical services
- Without IMU: `livePose.sensorsOK=False` → `livePose.valid=False` → autopilot disengages
- Sanity checks: rejects accel > 100 m/s^2, rotation > 10 rad/s, data must be < 100ms fresh

## ONNX Runtime on Jetson
Do NOT use `pip install onnxruntime-gpu` — desktop wheels are not compatible with Jetson aarch64 + Tegra CUDA. Use jetson-containers to build a compatible wheel:
```bash
# Using https://github.com/dusty-nv/jetson-containers
jetson-containers build onnxruntime
jetson-containers run $(autotag onnxruntime)
# Copy .whl out and install into venv
```

## Progress Tracking
See `CLAUDE_PROGRESS.md` for detailed progress and next steps.
