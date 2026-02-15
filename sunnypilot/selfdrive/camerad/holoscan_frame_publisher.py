#!/usr/bin/env python3
"""Holoscan IMX274 stereo camera → /dev/shm NV12 frame publisher.

Runs INSIDE the Holoscan Docker container. Captures from two IMX274
cameras via the Holoscan Sensor Bridge pipeline, converts RGBA uint16
output to NV12 uint8 via CUDA, and writes to /dev/shm ring buffers
that jetson_camerad.py reads on the host.

Usage (inside Docker):
  python3 /path/to/holoscan_frame_publisher.py [--headless] [--frame-limit N]

Camera mapping:
  Camera 0 (right, 90° FOV)  → /dev/shm/sunnypilot_cam_road  (VISION_STREAM_ROAD)
  Camera 1 (left, 120° FOV)  → /dev/shm/sunnypilot_cam_wide  (VISION_STREAM_WIDE_ROAD)
"""

import argparse
import ctypes
import logging
import os
import sys
import time

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import cuda.bindings.driver as cuda
import holoscan
import hololink as hololink_module

# Add the sunnypilot root so we can import shm_buffer
# When running from Docker with sunnypilot mounted, adjust path as needed
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SUNNYPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _SUNNYPILOT_ROOT not in sys.path:
    sys.path.insert(0, _SUNNYPILOT_ROOT)

from selfdrive.camerad.shm_buffer import ShmRingBufferWriter, SHM_PATH_ROAD, SHM_PATH_WIDE

# Output resolution from IMX274 in 1920x1080 mode
CAM_W = 1920
CAM_H = 1080

# --------------------------------------------------------------------------
# CUDA kernel: RGBA uint16 → NV12 uint8
# --------------------------------------------------------------------------
if HAS_CUPY:
    _rgba16_to_nv12_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void rgba16_to_nv12(const unsigned short* __restrict__ rgba,
                        unsigned char* __restrict__ nv12,
                        const int width, const int height) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        const int rgba_idx = (y * width + x) * 4;
        const float r = (float)(rgba[rgba_idx] >> 8);
        const float g = (float)(rgba[rgba_idx + 1] >> 8);
        const float b = (float)(rgba[rgba_idx + 2] >> 8);

        // BT.601 RGB → Y
        nv12[y * width + x] = (unsigned char)fminf(fmaxf(
            0.299f * r + 0.587f * g + 0.114f * b, 0.0f), 255.0f);

        // UV plane: subsample 2x2, interleaved U/V
        if ((x & 1) == 0 && (y & 1) == 0) {
            const unsigned char U = (unsigned char)fminf(fmaxf(
                -0.169f * r - 0.331f * g + 0.500f * b + 128.0f, 0.0f), 255.0f);
            const unsigned char V = (unsigned char)fminf(fmaxf(
                 0.500f * r - 0.419f * g - 0.081f * b + 128.0f, 0.0f), 255.0f);
            const int uv_base = width * height + (y >> 1) * width + x;
            nv12[uv_base]     = U;
            nv12[uv_base + 1] = V;
        }
    }
    ''', 'rgba16_to_nv12')


def rgba16_to_nv12_gpu(rgba_gpu, width, height):
    """Convert RGBA uint16 CuPy array → NV12 uint8 CuPy array on GPU."""
    nv12_size = width * height * 3 // 2
    nv12_gpu = cp.empty(nv12_size, dtype=cp.uint8)
    block = (16, 16)
    grid = ((width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1])
    _rgba16_to_nv12_kernel(grid, block, (rgba_gpu.ravel(), nv12_gpu, width, height))
    return nv12_gpu


def rgba16_to_nv12_cpu(rgba_np, width, height):
    """Convert RGBA uint16 numpy array → NV12 uint8 numpy array on CPU."""
    # rgba_np shape: (height, width, 4) dtype uint16
    rgb8 = (rgba_np[:, :, :3] >> 8).astype(np.float32)
    r, g, b = rgb8[:, :, 0], rgb8[:, :, 1], rgb8[:, :, 2]

    # BT.601
    y_plane = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)

    # Subsample 2x2 for UV
    r_sub = r[::2, ::2]
    g_sub = g[::2, ::2]
    b_sub = b[::2, ::2]
    u_plane = np.clip(-0.169 * r_sub - 0.331 * g_sub + 0.500 * b_sub + 128, 0, 255).astype(np.uint8)
    v_plane = np.clip(0.500 * r_sub - 0.419 * g_sub - 0.081 * b_sub + 128, 0, 255).astype(np.uint8)

    # Interleave UV
    uv = np.empty((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u_plane
    uv[:, 1::2] = v_plane

    return np.concatenate([y_plane.ravel(), uv.ravel()])


# --------------------------------------------------------------------------
# Custom Holoscan operator: captures demosaiced frames → /dev/shm
# --------------------------------------------------------------------------
class FrameToShmOp(holoscan.core.Operator):
    """Receives RGBA uint16 tensor from BayerDemosaic, converts to NV12,
    writes to /dev/shm ring buffer."""

    def __init__(self, fragment, *args, shm_path, width, height,
                 tensor_name="", **kwargs):
        self._shm_path = shm_path
        self._cam_width = width
        self._cam_height = height
        self._tensor_name = tensor_name
        self._writer = None
        self._frame_count = 0
        self._t_last_log = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("receivers")

    def start(self):
        logging.info(f"FrameToShmOp[{self.name}]: creating shm writer at {self._shm_path} "
                     f"({self._cam_width}x{self._cam_height})")
        self._writer = ShmRingBufferWriter(self._shm_path,
                                           self._cam_width,
                                           self._cam_height)

    def compute(self, op_input, op_output, context):
        message = op_input.receive("receivers")
        if message is None:
            return

        ts_ns = int(time.monotonic_ns())

        try:
            # Get tensor from Holoscan message
            # Try common Holoscan SDK tensor access patterns
            tensor = None
            try:
                tensor = message.get(self._tensor_name)
            except Exception:
                try:
                    tensor = message.get("")
                except Exception:
                    pass

            if tensor is None:
                logging.warning(f"FrameToShmOp[{self.name}]: could not extract tensor")
                return

            # Convert to NV12
            if HAS_CUPY:
                try:
                    gpu_arr = cp.from_dlpack(tensor)
                    nv12_gpu = rgba16_to_nv12_gpu(gpu_arr,
                                                   self._cam_width,
                                                   self._cam_height)
                    nv12_cpu = cp.asnumpy(nv12_gpu)
                except Exception:
                    # Fallback: copy to CPU first
                    np_arr = np.asarray(tensor)
                    if np_arr.ndim == 2:
                        np_arr = np_arr.reshape(self._cam_height, self._cam_width, -1)
                    nv12_cpu = rgba16_to_nv12_cpu(np_arr, self._cam_width, self._cam_height)
            else:
                np_arr = np.asarray(tensor)
                if np_arr.ndim == 2:
                    np_arr = np_arr.reshape(self._cam_height, self._cam_width, -1)
                nv12_cpu = rgba16_to_nv12_cpu(np_arr, self._cam_width, self._cam_height)

            # Write to shared memory
            self._writer.write(nv12_cpu, ts_ns)
            self._frame_count += 1

            # Log periodically
            now = time.monotonic()
            if now - self._t_last_log > 5.0:
                fps = self._frame_count / max(now - self._t_last_log, 0.001) if self._t_last_log > 0 else 0
                logging.info(f"FrameToShmOp[{self.name}]: {self._frame_count} frames, "
                             f"{fps:.1f} fps → {self._shm_path}")
                self._frame_count = 0
                self._t_last_log = now

        except Exception as e:
            logging.error(f"FrameToShmOp[{self.name}]: error: {e}")

    def stop(self):
        if self._writer:
            self._writer.close()
            logging.info(f"FrameToShmOp[{self.name}]: closed shm writer")


# --------------------------------------------------------------------------
# Holoscan Application
# --------------------------------------------------------------------------
class SunnypilotCameraApp(holoscan.core.Application):
    def __init__(self, headless, cuda_context, cuda_device_ordinal,
                 hololink_channel_left, camera_left,
                 hololink_channel_right, camera_right,
                 camera_mode, frame_limit):
        logging.info("SunnypilotCameraApp.__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._camera_left = camera_left
        self._hololink_channel_right = hololink_channel_right
        self._camera_right = camera_right
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")

        # -- Conditions --
        if self._frame_limit:
            condition_left = holoscan.conditions.CountCondition(
                self, name="count_left", count=self._frame_limit)
            condition_right = holoscan.conditions.CountCondition(
                self, name="count_right", count=self._frame_limit)
        else:
            condition_left = holoscan.conditions.BooleanCondition(
                self, name="ok_left", enable_tick=True)
            condition_right = holoscan.conditions.BooleanCondition(
                self, name="ok_right", enable_tick=True)

        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        # -- CSI to Bayer --
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self, name="pool", storage_type=1,
            block_size=self._camera_left._width * ctypes.sizeof(ctypes.c_uint16) * self._camera_left._height,
            num_blocks=6)

        csi_to_bayer_left = hololink_module.operators.CsiToBayerOp(
            self, name="csi_to_bayer_left", allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="left")
        self._camera_left.configure_converter(csi_to_bayer_left)

        csi_to_bayer_right = hololink_module.operators.CsiToBayerOp(
            self, name="csi_to_bayer_right", allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="right")
        self._camera_right.configure_converter(csi_to_bayer_right)

        frame_size = csi_to_bayer_left.get_csi_length()
        assert frame_size == csi_to_bayer_right.get_csi_length()

        # -- Receivers --
        frame_context = self._cuda_context
        receiver_left = hololink_module.operators.LinuxReceiverOp(
            self, condition_left, name="receiver_left",
            frame_size=frame_size, frame_context=frame_context,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left)

        receiver_right = hololink_module.operators.LinuxReceiverOp(
            self, condition_right, name="receiver_right",
            frame_size=frame_size, frame_context=frame_context,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right)

        # -- Image processing --
        bayer_format = self._camera_left.bayer_format()
        assert bayer_format == self._camera_right.bayer_format()
        pixel_format = self._camera_left.pixel_format()
        assert pixel_format == self._camera_right.pixel_format()

        image_processor_left = hololink_module.operators.ImageProcessorOp(
            self, name="image_processor_left", optical_black=50,
            bayer_format=bayer_format.value, pixel_format=pixel_format.value)

        image_processor_right = hololink_module.operators.ImageProcessorOp(
            self, name="image_processor_right", optical_black=50,
            bayer_format=bayer_format.value, pixel_format=pixel_format.value)

        # -- Demosaic --
        rgba_components = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self, name="bayer_pool", storage_type=1,
            block_size=self._camera_left._width * rgba_components * ctypes.sizeof(ctypes.c_uint16) * self._camera_left._height,
            num_blocks=6)

        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self, name="demosaic_left", pool=bayer_pool,
            generate_alpha=True, alpha_value=65535,
            bayer_grid_pos=bayer_format.value, interpolation_mode=0,
            in_tensor_name="left", out_tensor_name="left")

        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self, name="demosaic_right", pool=bayer_pool,
            generate_alpha=True, alpha_value=65535,
            bayer_grid_pos=bayer_format.value, interpolation_mode=0,
            in_tensor_name="right", out_tensor_name="right")

        # -- Our custom frame-to-shm operators --
        # Camera 0 = right = 90° FOV → road camera
        frame_to_shm_road = FrameToShmOp(
            self, name="frame_to_shm_road",
            shm_path=SHM_PATH_ROAD,
            width=CAM_W, height=CAM_H,
            tensor_name="right")

        # Camera 1 = left = 120° FOV → wide camera
        frame_to_shm_wide = FrameToShmOp(
            self, name="frame_to_shm_wide",
            shm_path=SHM_PATH_WIDE,
            width=CAM_W, height=CAM_H,
            tensor_name="left")

        # -- Wire up the pipeline --
        self.add_flow(receiver_left, csi_to_bayer_left, {("output", "input")})
        self.add_flow(receiver_right, csi_to_bayer_right, {("output", "input")})
        self.add_flow(csi_to_bayer_left, image_processor_left, {("output", "input")})
        self.add_flow(csi_to_bayer_right, image_processor_right, {("output", "input")})
        self.add_flow(image_processor_left, demosaic_left, {("output", "receiver")})
        self.add_flow(image_processor_right, demosaic_right, {("output", "receiver")})
        # Left camera (120°) → wide stream shm
        self.add_flow(demosaic_left, frame_to_shm_wide, {("transmitter", "receivers")})
        # Right camera (90°) → road stream shm
        self.add_flow(demosaic_right, frame_to_shm_road, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser(
        description="Capture IMX274 stereo cameras and publish NV12 frames to /dev/shm")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="Exit after N frames")
    parser.add_argument("--camera-mode", type=int,
                        default=hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
                        help="IMX274 mode")
    parser.add_argument("--hololink", default="192.168.0.2",
                        help="IP address of Hololink board")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Logging level")
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "holoscan-sensor-bridge",
                                   "examples", "example_configuration.yaml")
    parser.add_argument("--configuration", default=default_config,
                        help="Holoscan configuration YAML")
    args = parser.parse_args()

    hololink_module.logging_level(args.log_level)
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    logging.info("Sunnypilot Holoscan Frame Publisher starting")
    logging.info(f"  CuPy available: {HAS_CUPY}")
    logging.info(f"  Road SHM: {SHM_PATH_ROAD}")
    logging.info(f"  Wide SHM: {SHM_PATH_WIDE}")

    # Initialize CUDA
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Find cameras
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=args.hololink)
    logging.info(f"Found channel: {channel_metadata}")

    # Camera 1 (left, 120° FOV) → wide
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 1)
    # Camera 0 (right, 90° FOV) → road
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 0)

    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)

    camera_left = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_left, expander_configuration=1)
    camera_right = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_right, expander_configuration=0)

    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode)

    # Build application
    app = SunnypilotCameraApp(
        args.headless, cu_context, cu_device_ordinal,
        hololink_channel_left, camera_left,
        hololink_channel_right, camera_right,
        camera_mode, args.frame_limit)

    if os.path.exists(args.configuration):
        app.config(args.configuration)

    # Run
    hololink = hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    hololink.start()
    try:
        hololink.reset()
        camera_left.setup_clock()
        camera_left.configure(camera_mode)
        camera_left.set_digital_gain_reg(0x4)
        camera_right.configure(camera_mode)
        camera_right.set_digital_gain_reg(0x4)

        logging.info("Starting Holoscan pipeline — publishing frames to /dev/shm")
        app.run()
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    logging.info("Done")


if __name__ == "__main__":
    main()
