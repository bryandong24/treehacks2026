# Sunnypilot on Jetson AGX Thor — Progress Tracker

## Platform
- **Hardware**: NVIDIA Jetson AGX Thor
- **JetPack**: 7.1 (L4T R38.4.0)
- **CUDA**: 13.0
- **Kernel**: 6.8.12-tegra
- **OS**: Ubuntu 24.04.4 LTS (Noble)

## Completed

### 1. ONNX Runtime with CUDA — DONE
- Built a compatible onnxruntime wheel for Jetson Thor (aarch64, CUDA 13.0, JetPack 7.1) using jetson-containers
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

### 8. Build ONNX-Based Model Runner — DONE
- Replaced tinygrad/OpenCL `ModelState` in both `modeld.py` and `dmonitoringmodeld.py` with ONNX Runtime + CUDA
- **Changes to `selfdrive/modeld/modeld.py`**:
  - Removed all tinygrad, OpenCL, CLContext, TICI/USBGPU imports and code
  - `ModelState.__init__()`: creates two `ort.InferenceSession` (vision + policy) with `ORT_ENABLE_ALL` + `EXHAUSTIVE` cuDNN
  - `ModelState.run()`: CUDA preprocessing via `CUDADrivingModelFrame.prepare()` → ONNX vision inference → ONNX policy inference (fp16 inputs)
  - `main()`: no CLContext, VisionIpcClient without CL context arg
  - All downstream code (InputQueues, fill_model_msg, cereal publishing) unchanged
- **Changes to `selfdrive/modeld/dmonitoringmodeld.py`**:
  - Same pattern: removed tinygrad/OpenCL, replaced with ONNX RT session
  - `CUDAMonitoringModelFrame.prepare()` for Y-plane warp → ONNX inference (input_img uint8 + calib float32)
  - All output parsing (slice_outputs, parse_model_output, fill_driver_data) unchanged
- **Generated metadata pickle files** from ONNX model metadata:
  - `driving_vision_metadata.pkl`, `driving_policy_metadata.pkl`, `dmonitoring_model_metadata.pkl`
  - Generated via `sunnypilot/modeld/get_model_metadata.py` — keeps all metadata loading code unchanged

### 9. Cython Extensions Compiled — DONE
- Compiled all required Cython `.so` extensions using `scons --minimal`:
  - `msgq_repo/msgq/ipc_pyx.so` (955KB) — IPC messaging (Context, PubSocket, SubSocket, Poller)
  - `msgq_repo/msgq/visionipc/visionipc_pyx.so` (2.5MB) — VisionIPC (Server, Client, VisionBuf, VisionStreamType)
  - `common/params_pyx.so` — Params key-value store
  - `rednose/helpers/ekf_sym_pyx.so` — Kalman filter helpers
- Created `sunnypilot/modeld/runners/runmodel_pyx.py` — Python stub for the old tinygrad runner (not needed on Jetson)
- **All imports verified**: `cereal.messaging`, `msgq.visionipc`, `openpilot.common.params`, full modeld.py import chain works

### 10. End-to-End Pipeline Validation — DONE
- **Standalone pipeline test** (`selfdrive/modeld/test_pipeline_e2e.py`):
  - Tests CUDA preprocessing + vision + policy + dmon inference with synthetic NV12 frames
  - 100-frame sustained 20Hz simulation: **0 missed deadlines** out of 100 frames
  - All model outputs validated (plan, hidden_state, lane_lines_prob, lead_prob, meta, desire_state, pose, face/eye probabilities)
- **VisionIPC integration test** (`selfdrive/modeld/test_vipc_e2e.py`):
  - Creates VisionIPC server publishing fake camera frames
  - Connects VisionIPC clients, receives frames through shared memory
  - Full pipeline: VisionIPC recv → CUDA preprocess → ONNX vision → ONNX policy → output parsing
  - Frame data integrity verified: sent == received through VisionIPC round-trip

#### Verified Latency (steady-state, excluding warmup)

| Stage | Avg | Min | Max | p95 |
|---|---|---|---|---|
| CUDA preprocess (both cams) | 1.81 ms | 1.49 ms | 2.48 ms | 2.22 ms |
| Vision ONNX inference | 5.33 ms | 3.46 ms | 8.55 ms | 8.40 ms |
| Policy ONNX inference | 1.22 ms | 0.84 ms | 1.86 ms | 1.74 ms |
| **Total driving pipeline** | **8.66 ms** | 6.06 ms | 12.53 ms | 12.34 ms |
| DMonitoring preprocess | 0.50 ms | 0.47 ms | 0.59 ms | 0.56 ms |
| DMonitoring inference | 3.64 ms | 3.49 ms | 3.94 ms | 3.88 ms |
| **Total DMonitoring** | **4.15 ms** | 3.99 ms | 4.45 ms | 4.39 ms |
| **Total all (driving + dmon)** | **11.03 ms** | 8.85 ms | 22.06 ms | 20.18 ms |
| **Budget (20Hz)** | **50 ms** | — | — | — |
| **Headroom** | **~39 ms** | — | — | — |

---

## TODO — Next Steps

### 11. Camera Integration
- Our cameras: 120-degree FOV + 90-degree FOV
- Sunnypilot expects:
  - `fcam`: ~52-degree FOV (comma OX03C10/OS04C10)
  - `ecam`: ~120-degree FOV (comma AR0231)
- Need to:
  - Map our cameras to fcam/ecam roles (90-degree → fcam, 120-degree → ecam)
  - Adjust camera intrinsics in `common/transformations/camera.py`
  - The warp matrix compensates for FOV differences, but intrinsics must be correct
  - Test with VisionIPC or write a custom camera feed adapter

### 12. IMU + GPS Integration
- Sunnypilot **requires** IMU (accelerometer + gyroscope) — hard dependency
- Stock uses LSM6DS3 at 104 Hz via I2C through `system/sensord/sensord.py`
- Need to:
  - Connect external IMU to Jetson Thor
  - Write or adapt a sensor driver that publishes `accelerometer` and `gyroscope` cereal messages
  - Ensure data freshness < 100ms and passes sanity checks (accel < 100 m/s^2, rotation < 10 rad/s)
- GPS feeds into `locationd` for position — needed for navigation but less critical than IMU for core driving

### 13. Honda Bosch CAN Interface
- Vehicle: Honda Bosch platform
- Need panda OBD-II adapter (or compatible CAN interface) connected to Jetson
- Verify opendbc Honda Bosch DBC definitions work
- Test CAN read (steering angle, wheel speeds, brake) and CAN write (steering torque, gas, brake)

### 14. VisionIPC / Camera Pipeline
- Stock sunnypilot uses `camerad` → VisionIPC shared memory → `modeld`
- Need to either:
  - Write a custom `camerad` for our camera hardware on Jetson
  - Or feed frames via V4L2 → VisionIPC adapter
- Must provide both road and wide camera streams simultaneously

### 15. Full System Integration Test
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
│  CUDA Kernels (CuPy)  ✅        │
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
- **CuPy libnvrtc symlink**: CuPy (cupy-cuda12x) looks for `libnvrtc.so.12` but Jetson Thor has CUDA 13 (`libnvrtc.so.13`). Symlink created at `.venv/lib/libnvrtc.so.12 → /usr/local/cuda/lib64/libnvrtc.so.13`. Must set `LD_LIBRARY_PATH` to include `.venv/lib` when running, or create a system-level symlink.
- The DRM warning (`/sys/class/drm/card3/device/vendor`) is harmless — Jetson sysfs quirk, does not affect CUDA inference
- TensorRT provider listed as available but fails at runtime — needs `libnvinfer.so.10` (TensorRT libs). Install later for potential further speedup.
- **IMPORTANT**: Must use `ORT_ENABLE_ALL` graph optimization level on Jetson Thor. Stock code uses `ORT_DISABLE_ALL` which causes Conv fallback (~93ms for dmon). With `ORT_ENABLE_ALL` + `EXHAUSTIVE` cuDNN algo search, all models run in single-digit ms.
- Model runs at 20 Hz (50ms budget per frame); full driving pipeline (vision + policy) takes ~8ms on CUDA, leaving 42ms headroom
- Driver monitoring runs on a separate thread; at ~4ms per inference it can easily sustain 20+ Hz
