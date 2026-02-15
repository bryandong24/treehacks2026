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

### 11. USB Webcam → Driving Model Pipeline — DONE
- **Test file**: `selfdrive/modeld/test_webcam.py`
- Captures live frames from a USB webcam, converts BGR→NV12, and runs the full pipeline:
  - CUDA preprocessing (warp + YUV packing) → ONNX vision inference → ONNX policy inference
- Constructs a proper warp matrix from webcam intrinsics (estimated ~65° HFOV) using `calib_from_medmodel`
- Runs 30 frames with full model output parsing (plan, lane lines, lead prob, pose, desire state)
- **Verifies**: real sensor data flows through the entire CUDA + ONNX pipeline end-to-end
- Note: model outputs are not meaningful (webcam isn't mounted like a car camera), but this confirms the full pipeline works with real image data from a real sensor on CUDA

### 12. Serial IMU Integration (LSM6DSOX via USB-UART) — DONE
- **Hardware**: LSM6DSOX IMU → Arduino/Feather (I2C) → CP2104 USB-UART bridge → `/dev/ttyUSB0`
- **udev rule**: `/etc/udev/rules.d/13-imu.rules` creates persistent `/dev/IMU` symlink (matches CP2104 serial `02896079`, mode `0666`)
- **Daemon**: `system/sensord/serial_imu.py`
  - Opens `/dev/IMU` at 460800 baud, reads ASCII CSV lines (`ax,ay,az,gx,gy,gz`)
  - Values arrive pre-scaled in m/s² (accel) and rad/s (gyro) from the Arduino
  - Publishes cereal `accelerometer` + `gyroscope` messages at 104 Hz via `Ratekeeper`
  - Uses `SensorSource.lsm6ds3` (LSM6DSOX same family, accepted by locationd)
  - Drains serial buffer each cycle to always use the freshest sample
  - Uses `time.monotonic_ns()` for timestamps (must match `logMonoTime` clock domain — `time.time_ns()` differs by ~56 years on this system)
- **E2E sensor test**: `system/sensord/test_serial_imu.py`
  - Runs daemon in background thread, subscribes via `SubMaster` for 5 seconds
  - Results: **521 samples at 104.2 Hz**, accel norm 10.0 m/s², gyro norm 0.009 rad/s
  - All values within locationd sanity bounds (accel < 100 m/s², gyro < 10 rad/s)

### 13. IMU → locationd Integration — DONE
- **Integration test**: `system/sensord/test_serial_imu_locationd.py`
  - Runs `serial_imu.py` (real IMU) + `locationd.py` (Kalman filter) + mock publishers for `carState` (100Hz), `liveCalibration` (4Hz), `cameraOdometry` (20Hz)
  - Subscribes to `livePose` output for 10 seconds
- **Build step**: Compiled `selfdrive/locationd/models/generated/libpose.so` from generated `pose.cpp` (Kalman filter shared library required by `ekf_sym_pyx`)
- **Results** (197 livePose messages over 10s):
  - **`sensorsOK: 197/197 (100%)`** — locationd fully accepts serial IMU data
  - **`inputsOK: 197/197 (100%)`** — all critical service inputs pass validation
  - **`valid: 197/197 (100%)`** — Kalman filter initialized and producing state estimates
  - **`posenetOK: 197/197 (100%)`**
  - Raw IMU: 1040 accel + 1040 gyro samples at 104 Hz
- **Timestamp fix**: sensor `event.timestamp` must use `time.monotonic_ns()` (not `time.time_ns()`) because locationd compares it against `logMonoTime` which uses the monotonic clock. Wall clock differs from monotonic by ~56 years on this Jetson.
- **Note**: Kalman filter acceleration/angular velocity estimates drift with mock cameraOdometry (zero motion conflicts with real gravity vector). This is expected and will resolve with real visual odometry.

### 14. Platform Integration & Pandad for Jetson — DONE
- **Hardware detection**: Modified `system/hardware/__init__.py` to add `JETSON = os.path.isfile('/JETSON')` flag
  - `PC = not TICI and not JETSON` — Jetson is no longer treated as PC mode
  - Still uses `Pc()` hardware class (no custom JetsonThor class needed)
  - This unblocks all processes gated by `enabled=not PC` (sensord, dmonitoringmodeld, dmonitoringd, timed, etc.)
- **Marker file**: Requires `sudo touch /JETSON` on the Jetson device
- **Process manager**: Updated `system/manager/process_config.py`:
  - Imported `JETSON` flag from hardware module
  - Stock `sensord` (I2C-based) gated with `enabled=not PC and not JETSON` (won't run on Jetson)
  - Added `serial_imu` process (`system.sensord.serial_imu`) with `enabled=JETSON` — uses our USB-UART IMU daemon
- **C++ pandad binary**: Compiled for aarch64 Jetson Thor using scons
  - `scons --minimal -j$(nproc) selfdrive/pandad/pandad`
  - Built with clang++ 18.1.3, links against libusb-1.0, libzmq, libcapnp, common, messaging
  - Binary: `selfdrive/pandad/pandad` (4.7MB, ELF 64-bit ARM aarch64)
  - Smoke tested: starts, attempts USB connection (no panda attached), falls back to SPI (expected failure)
  - On Jetson (no `__TICI__` define), uses `HardwarePC` C++ class — skips RT priority/core affinity (safe)
  - Panda connection strategy: USB first → SPI fallback. External red panda via USB works.
- **Car interface code** (card.py, controlsd.py, Honda carcontroller/carstate/hondacan) is 100% hardware-agnostic — no changes needed
- **Data flow verified** (architecture review):
  - `modeld` → `modelV2` (cereal) → `controlsd` (100Hz) → `carControl` → `card.py` → `CarInterface.apply()` → Honda `CarController` → CAN frames → `sendcan` (cereal) → C++ `pandad` → USB → red panda → CAN bus → Honda ECUs
  - Reverse: Honda ECUs → CAN bus → red panda → USB → C++ `pandad` → `can` (cereal) → `card.py` → Honda `CarState` → `carState` (cereal) → `controlsd`

---

## TODO — Next Steps

### 15. Camera Integration
- Our cameras: 120-degree FOV + 90-degree FOV
- Sunnypilot expects:
  - `fcam`: ~52-degree FOV (comma OX03C10/OS04C10)
  - `ecam`: ~120-degree FOV (comma AR0231)
- Need to:
  - Map our cameras to fcam/ecam roles (90-degree → fcam, 120-degree → ecam)
  - Adjust camera intrinsics in `common/transformations/camera.py`
  - The warp matrix compensates for FOV differences, but intrinsics must be correct
  - Test with VisionIPC or write a custom camera feed adapter

### 16. GPS Integration
- GPS feeds into `locationd` for position — needed for navigation but less critical than IMU for core driving

### 17. Honda Bosch CAN Interface
- Vehicle: Honda Bosch platform
- Need red panda OBD-II adapter connected to Jetson via USB
- Verify opendbc Honda Bosch DBC definitions work
- Test CAN read (steering angle, wheel speeds, brake) and CAN write (steering torque, gas, brake)
- C++ pandad binary is compiled and ready — will connect to red panda over USB

### 18. VisionIPC / Camera Pipeline
- Stock sunnypilot uses `camerad` → VisionIPC shared memory → `modeld`
- Need to either:
  - Write a custom `camerad` for our camera hardware on Jetson
  - Or feed frames via V4L2 → VisionIPC adapter
- Must provide both road and wide camera streams simultaneously

### 19. Full System Integration Test
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
│  controlsd (100Hz)  ✅           │
│  → carControl → card.py         │
└──────────────┬──────────────────┘
               │ sendcan (cereal IPC)
               ▼
┌─────────────────────────────────┐
│  C++ pandad (USB relay)  ✅      │
│  → Red Panda → CAN bus          │
│  → Honda Bosch ECUs             │
└─────────────────────────────────┘

        ┌──────────┐  ┌──────────┐
        │   IMU  ✅ │  │   GPS    │
        │ (104 Hz) │  │          │
        └────┬─────┘  └────┬─────┘
             │             │
             ▼             ▼
        ┌─────────────────────┐
        │  locationd (Kalman)✅│
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
