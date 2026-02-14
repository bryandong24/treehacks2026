# Sunnypilot on Jetson AGX Thor — Progress Tracker

## Completed

### 1. ONNX Runtime with CUDA — DONE
- Built a compatible onnxruntime wheel for Jetson Thor (aarch64, CUDA 13.0) using jetson-containers
- Installed into sunnypilot venv at `/home/subha/treehacks2026/sunnypilot/.venv`
- Standard `pip install onnxruntime-gpu` does NOT work on Jetson — must build from jetson-containers

### 2. CUDA Toolkit Dependencies — DONE
- Installed `cuda-toolkit-13-0` (full toolkit including cuBLAS, cuFFT, cuSOLVER, cuRAND, etc.)
- Installed `cudnn9-cuda-13-0` (cuDNN 9 for CUDA 13)
- Installed `libcublas-13-0`
- All shared library dependencies for onnxruntime CUDAExecutionProvider resolved

### 3. CUDAExecutionProvider Verified — DONE
- Confirmed available providers: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

### 4. End-to-End Model Inference on GPU — DONE
- Ran `driving_vision.onnx` with CUDAExecutionProvider using dummy uint8 image inputs
- Results:
  - Model loads and runs successfully on Thor GPU
  - Input: `img` [1,12,128,256] uint8 + `big_img` [1,12,128,256] uint8
  - Output: [1,1576] float16
  - **Avg latency: 7.06 ms** (min 2.93 ms, max 8.74 ms) on CUDA
  - Compared to ~34 ms on CPU fallback (~5x speedup)

---

## TODO — Next Steps

### 5. Test `driving_policy.onnx` on CUDA
- Load and run the policy model with dummy inputs on CUDAExecutionProvider
- Verify input/output shapes match what modeld.py expects
- Benchmark latency

### 6. Test `dmonitoring_model.onnx` on CUDA
- Load and run driver monitoring model
- Verify latency is acceptable for real-time operation

### 7. Rewrite OpenCL Kernels as CUDA Kernels
**Critical — Jetson Thor does NOT support OpenCL.**

The stock sunnypilot uses OpenCL for image preprocessing before feeding the model. These must be rewritten as CUDA kernels:

- **Perspective warp**: Transforms raw camera image to model input space using a 3x3 warp matrix (from `get_warp_matrix()` in `common/transformations/model.py`). This corrects for camera mounting, calibration, and maps to the model's expected viewpoint.
- **YUV420 channel packing**: The camera provides YUV420 planar data. The model expects 6 channels per frame: 4 subsampled Y channels `[Y[::2,::2], Y[::2,1::2], Y[1::2,::2], Y[1::2,1::2]]` + U + V, packed into a `[6, 128, 256]` uint8 tensor.

Key files to study:
- `selfdrive/modeld/models/commonmodel.cc` — C++ image transform code
- `selfdrive/modeld/models/commonmodel_pyx.pyx` — Cython bindings for DrivingModelFrame
- `selfdrive/modeld/runners/` — tinygrad runner helpers

Options:
- Write CUDA kernels (`.cu`) using PyCUDA or CuPy
- Use NVIDIA NPP (NVIDIA Performance Primitives) for warp/resize
- Use custom CUDA C extensions compiled with nvcc

### 8. Build ONNX-Based Model Runner
- Replace the tinygrad-based `ModelState` in `modeld.py` with an ONNX Runtime session
- The current flow loads `.pkl` tinygrad models; we need to load `.onnx` files instead
- Wire up: camera frames → CUDA preprocessing → ONNX vision model → features → ONNX policy model → driving plan
- Maintain the same output format so downstream code (fill_model_msg, controls) works unchanged

### 9. Camera Integration
- Our cameras: 120-degree FOV + 90-degree FOV
- Sunnypilot expects:
  - `fcam`: ~52-degree FOV (comma OX03C10/OS04C10)
  - `ecam`: ~120-degree FOV (comma AR0231)
- Need to:
  - Map our cameras to fcam/ecam roles (90-degree → fcam, 120-degree → ecam)
  - Adjust camera intrinsics in `common/transformations/camera.py`
  - The warp matrix compensates for FOV differences, but intrinsics must be correct
  - Test with VisionIPC or write a custom camera feed adapter

### 10. IMU + GPS Integration
- Sunnypilot **requires** IMU (accelerometer + gyroscope) — hard dependency
- Stock uses LSM6DS3 at 104 Hz via I2C through `system/sensord/sensord.py`
- Need to:
  - Connect external IMU to Jetson Thor
  - Write or adapt a sensor driver that publishes `accelerometer` and `gyroscope` cereal messages
  - Ensure data freshness < 100ms and passes sanity checks (accel < 100 m/s^2, rotation < 10 rad/s)
- GPS feeds into `locationd` for position — needed for navigation but less critical than IMU for core driving

### 11. Honda Bosch CAN Interface
- Vehicle: Honda Bosch platform
- Need panda OBD-II adapter (or compatible CAN interface) connected to Jetson
- Verify opendbc Honda Bosch DBC definitions work
- Test CAN read (steering angle, wheel speeds, brake) and CAN write (steering torque, gas, brake)

### 12. VisionIPC / Camera Pipeline
- Stock sunnypilot uses `camerad` → VisionIPC shared memory → `modeld`
- Need to either:
  - Write a custom `camerad` for our camera hardware on Jetson
  - Or feed frames via V4L2 → VisionIPC adapter
- Must provide both road and wide camera streams simultaneously

### 13. Full System Integration Test
- Run full sunnypilot stack on Jetson Thor with:
  - Camera feeds (both streams)
  - ONNX models on CUDA
  - IMU + GPS data
  - CAN bus connected to Honda Bosch
- Validate end-to-end: camera → model → controls → CAN output
- Test latency budget: full pipeline must complete within 50ms (20 Hz model rate)

---

## Architecture Diagram

```
┌─────────────┐    ┌─────────────┐
│  Road Cam   │    │  Wide Cam   │
│  (90° FOV)  │    │ (120° FOV)  │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────────────────────────┐
│  CUDA Kernels (TODO)            │
│  • Perspective warp             │
│  • YUV420 channel packing       │
└──────┬──────────────────┬───────┘
       │ img              │ big_img
       ▼                  ▼
┌─────────────────────────────────┐
│  driving_vision.onnx (CUDA)  ✅ │
│  [1,12,128,256] uint8 → [1,1576]│
└──────────────┬──────────────────┘
               │ hidden_state (512-dim)
               ▼
┌─────────────────────────────────┐
│  driving_policy.onnx (CUDA)     │
│  features + desire + traffic    │
│  → driving plan                 │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Controls (steering, accel)     │
│  → CAN bus → Honda Bosch        │
└─────────────────────────────────┘

        ┌──────────┐  ┌──────────┐
        │   IMU    │  │   GPS    │
        │ (104 Hz) │  │          │
        └────┬─────┘  └────┬─────┘
             │             │
             ▼             ▼
        ┌─────────────────────┐
        │  locationd (Kalman) │
        │  → livePose         │
        └─────────────────────┘
```

---

## Notes
- The DRM warning (`/sys/class/drm/card3/device/vendor`) is harmless — Jetson sysfs quirk, does not affect CUDA inference
- TensorRT execution provider is also available — could provide further speedup after CUDA path is stable
- Model runs at 20 Hz (50ms budget per frame); vision model alone takes ~7ms on CUDA, leaving good headroom
