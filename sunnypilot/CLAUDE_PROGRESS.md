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

### 5. `driving_policy.onnx` on CUDA — DONE
- Loaded and ran with CUDAExecutionProvider using dummy inputs
- Results:
  - Input: `desire_pulse` [1,25,8] fp16, `traffic_convention` [1,2] fp16, `features_buffer` [1,25,512] fp16
  - Output: [1,1000] float16 — matches expected policy output size
  - **Avg latency: 0.93 ms** (min 0.88 ms, max 1.31 ms) on CUDA
  - No NaN/Inf in output, valid value range [-32, 208]

### 6. `dmonitoring_model.onnx` on CUDA — DONE
- Loaded and ran with CUDAExecutionProvider
- Results:
  - Input: `input_img` [1,1382400] uint8 (1440x960 Y-channel), `calib` [1,3] float32
  - Output: [1,551] float16 — matches expected dmonitoring output size
  - **Avg latency: 3.83 ms** (with `ORT_ENABLE_ALL` + `EXHAUSTIVE` cuDNN algo search)
  - No NaN/Inf in output, face/eye probabilities produce plausible raw logits
- **Critical finding**: Stock `onnxmodel.py` uses `ORT_DISABLE_ALL` (graph optimizations disabled), which causes Conv ops to fall back to unoptimized cuDNN paths (~93ms). Enabling `ORT_ENABLE_ALL` fuses the graph and brings latency down to ~4ms. **Must use `ORT_ENABLE_ALL` for Jetson Thor.**
- TensorRT provider failed due to missing `libnvinfer.so.10` — TensorRT libraries not installed. Not critical since CUDA+optimizations is already fast. Can install later for further gains.

### All Three Models — Latency Summary
| Model | Avg Latency | Output | Status |
|---|---|---|---|
| `driving_vision.onnx` | 7.06 ms | [1,1576] fp16 | PASS |
| `driving_policy.onnx` | 0.93 ms | [1,1000] fp16 | PASS |
| `dmonitoring_model.onnx` | 3.83 ms | [1,551] fp16 | PASS |
| **Total driving pipeline** | **~8 ms** | — | Well within 50ms budget |

---

### 7. Rewrite OpenCL Kernels as CUDA (CuPy) — DONE
- Replaced entire OpenCL pipeline with pure Python + CuPy CUDA:
  - `selfdrive/modeld/transforms/cuda_transforms.py` — CUDA warp kernel (RawKernel) + loadyuv + frame classes
  - `selfdrive/modeld/transforms/test_cuda_transforms.py` — comprehensive test suite
- **Components implemented**:
  - `warpPerspective` CUDA RawKernel — line-by-line translation of `transform.cl` with bilinear interpolation
  - `transform_scale_buffer()` — port of `common/mat.h` pixel-center scaling
  - `loadyuv()` — replaces `loadyuv.cl` loadys/loaduv kernels using CuPy array slicing
  - `DrivingModelFrame` — full temporal buffer management (warp Y/U/V → pack → shift → output oldest+newest)
  - `MonitoringModelFrame` — Y-plane warp only for driver monitoring
- **CuPy installed**: `cupy-cuda12x` v13.6.0 with `libnvrtc.so.12` symlink for CUDA 13 compatibility
- **All tests pass**: identity warp, translation warp, loadyuv sub-plane packing, temporal buffer, ONNX integration
- **ONNX integration verified**: preprocessed output → driving_vision.onnx → valid [1,1576] fp16 output, no NaN/Inf
- **Performance**:
  - DrivingModelFrame (1928×1208 → 512×256): **avg 1.09ms** (min 0.92ms)
  - MonitoringModelFrame (1928×1208 → 1440×960): **avg 1.16ms** (min 0.59ms)
  - Well within 5ms target

## TODO — Next Steps

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
│  driving_vision.onnx (CUDA)  ✅  │
│  [1,12,128,256] uint8 → [1,1576] │
│  ~7ms                             │
└──────────────┬────────────────────┘
               │ hidden_state (512-dim)
               ▼
┌───────────────────────────────────┐
│  driving_policy.onnx (CUDA)  ✅   │
│  features + desire + traffic      │
│  → driving plan  ~1ms             │
└──────────────┬────────────────────┘
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
- TensorRT provider listed as available but fails at runtime — needs `libnvinfer.so.10` (TensorRT libs). Install later for potential further speedup.
- **IMPORTANT**: Must use `ORT_ENABLE_ALL` graph optimization level on Jetson Thor. Stock code uses `ORT_DISABLE_ALL` which causes Conv fallback (~93ms for dmon). With `ORT_ENABLE_ALL` + `EXHAUSTIVE` cuDNN algo search, all models run in single-digit ms.
- Model runs at 20 Hz (50ms budget per frame); full driving pipeline (vision + policy) takes ~8ms on CUDA, leaving 42ms headroom
- Driver monitoring runs on a separate thread; at ~4ms per inference it can easily sustain 20+ Hz
