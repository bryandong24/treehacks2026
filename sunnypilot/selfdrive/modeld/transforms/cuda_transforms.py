"""CUDA-accelerated image transforms for sunnypilot on NVIDIA Jetson.

Replaces the OpenCL pipeline (transform.cl, loadyuv.cl, commonmodel.cc)
with CuPy/CUDA for Jetson AGX Thor which does not support OpenCL.

Pipeline:
  Raw NV12 YUV data (numpy array)
    -> CUDA warpPerspective RawKernel (3x: Y, U, V)
    -> CuPy array slicing (YUV channel packing)
    -> CuPy array copy (temporal buffer management)
    -> numpy array -> ONNX Runtime
"""

import os
import ctypes
import numpy as np

# Preload libnvrtc.so.12 for CuPy on Jetson Thor (CUDA 13 ships libnvrtc.so.13)
_venv_lib = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.venv', 'lib')
_nvrtc_path = os.path.join(os.path.abspath(_venv_lib), 'libnvrtc.so.12')
if not os.path.exists(_nvrtc_path):
  _nvrtc_path = '/usr/local/cuda/lib64/libnvrtc.so.12'
if os.path.exists(_nvrtc_path):
  ctypes.CDLL(_nvrtc_path, mode=ctypes.RTLD_GLOBAL)

import cupy as cp

# --- CUDA Warp Perspective Kernel ---
# Line-by-line translation of selfdrive/modeld/transforms/transform.cl

WARP_KERNEL_CODE = r"""
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

extern "C" __global__
void warpPerspective(
    const unsigned char* src,
    int src_row_stride, int src_px_stride, int src_offset,
    int src_rows, int src_cols,
    unsigned char* dst,
    int dst_row_stride, int dst_offset,
    int dst_rows, int dst_cols,
    const float* M)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < dst_cols && dy < dst_rows)
    {
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W  = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? (float)INTER_TAB_SIZE / W : 0.0f;
        int X = __float2int_rn(X0 * W);
        int Y = __float2int_rn(Y0 * W);

        int sx = X >> INTER_BITS;
        int sy = Y >> INTER_BITS;

        // Clamp to source bounds
        int sx_clamp = max(0, min(sx, src_cols - 1));
        int sx_p1_clamp = max(0, min(sx + 1, src_cols - 1));
        int sy_clamp = max(0, min(sy, src_rows - 1));
        int sy_p1_clamp = max(0, min(sy + 1, src_rows - 1));

        // Fetch 4 source pixels for bilinear interpolation
        int v0 = (int)src[sy_clamp * src_row_stride + src_offset + sx_clamp * src_px_stride];
        int v1 = (int)src[sy_clamp * src_row_stride + src_offset + sx_p1_clamp * src_px_stride];
        int v2 = (int)src[sy_p1_clamp * src_row_stride + src_offset + sx_clamp * src_px_stride];
        int v3 = (int)src[sy_p1_clamp * src_row_stride + src_offset + sx_p1_clamp * src_px_stride];

        // Sub-pixel fractional parts
        int ay = Y & (INTER_TAB_SIZE - 1);
        int ax = X & (INTER_TAB_SIZE - 1);
        float taby = 1.0f / INTER_TAB_SIZE * ay;
        float tabx = 1.0f / INTER_TAB_SIZE * ax;

        // Bilinear interpolation weights (fixed-point)
        int itab0 = __float2int_rn((1.0f - taby) * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
        int itab1 = __float2int_rn((1.0f - taby) * tabx * INTER_REMAP_COEF_SCALE);
        int itab2 = __float2int_rn(taby * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
        int itab3 = __float2int_rn(taby * tabx * INTER_REMAP_COEF_SCALE);

        int val = v0 * itab0 + v1 * itab1 + v2 * itab2 + v3 * itab3;

        // Round and clamp to [0, 255]
        unsigned char pix = (unsigned char)max(0, min(
            (val + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS, 255));
        dst[dy * dst_row_stride + dst_offset + dx] = pix;
    }
}
"""

_warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warpPerspective')

# Thread block size for 2D kernel launch
_BLOCK_SIZE = (16, 16)


def _warp_perspective(src_gpu, src_row_stride, src_px_stride, src_offset,
                      src_rows, src_cols, dst_gpu, dst_row_stride, dst_offset,
                      dst_rows, dst_cols, M_gpu):
    """Launch the CUDA warpPerspective kernel."""
    grid = (
        (dst_cols + _BLOCK_SIZE[0] - 1) // _BLOCK_SIZE[0],
        (dst_rows + _BLOCK_SIZE[1] - 1) // _BLOCK_SIZE[1],
    )
    _warp_kernel(
        grid, _BLOCK_SIZE,
        (src_gpu, np.int32(src_row_stride), np.int32(src_px_stride), np.int32(src_offset),
         np.int32(src_rows), np.int32(src_cols),
         dst_gpu, np.int32(dst_row_stride), np.int32(dst_offset),
         np.int32(dst_rows), np.int32(dst_cols),
         M_gpu)
    )


def transform_scale_buffer(projection, s):
    """Scale projection matrix for pixel-center origin at scale s.

    Direct port of common/mat.h:transform_scale_buffer.
    Maps: in_pt = (transform(out_pt/s + 0.5) - 0.5) * s
    """
    transform_out = np.array([
        [1.0 / s, 0.0, 0.5],
        [0.0, 1.0 / s, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    transform_in = np.array([
        [s, 0.0, -0.5 * s],
        [0.0, s, -0.5 * s],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return transform_in @ projection @ transform_out


def loadyuv(y_warped, u_warped, v_warped):
    """Pack warped YUV planes into 6-channel format expected by the driving model.

    Translates loadyuv.cl's loadys + loaduv kernels.

    The Y plane is decomposed into 4 sub-planes via 2x2 block subsampling:
      slot 0: Y[even_row, even_col]  (UV_SIZE bytes)
      slot 1: Y[odd_row,  even_col]
      slot 2: Y[even_row, odd_col]
      slot 3: Y[odd_row,  odd_col]
      slot 4: U plane
      slot 5: V plane

    Each sub-plane has dimensions (H/2, W/2) = UV_SIZE.
    Total = 4*UV_SIZE + 2*UV_SIZE = W*H*3/2 = MODEL_FRAME_SIZE.

    Args:
        y_warped: CuPy array (H, W) uint8 - warped Y plane
        u_warped: CuPy array (H/2, W/2) uint8 - warped U plane
        v_warped: CuPy array (H/2, W/2) uint8 - warped V plane

    Returns:
        CuPy array (MODEL_FRAME_SIZE,) uint8 - packed 6-channel output
    """
    return cp.concatenate([
        y_warped[0::2, 0::2].ravel(),  # slot 0: Y[even_row, even_col]
        y_warped[1::2, 0::2].ravel(),  # slot 1: Y[odd_row,  even_col]
        y_warped[0::2, 1::2].ravel(),  # slot 2: Y[even_row, odd_col]
        y_warped[1::2, 1::2].ravel(),  # slot 3: Y[odd_row,  odd_col]
        u_warped.ravel(),               # slot 4: U plane
        v_warped.ravel(),               # slot 5: V plane
    ])


class DrivingModelFrame:
    """CUDA replacement for the OpenCL DrivingModelFrame.

    Manages the full image preprocessing pipeline for the driving model:
    1. Perspective warp of NV12 Y/U/V planes
    2. YUV channel packing (loadyuv)
    3. Temporal buffer (keeps temporal_skip+1 frames, outputs oldest + newest)
    """
    MODEL_WIDTH = 512
    MODEL_HEIGHT = 256
    MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2  # 196608

    def __init__(self, temporal_skip=4):
        self.temporal_skip = temporal_skip
        self.frame_size = self.MODEL_FRAME_SIZE
        self.buf_size = self.MODEL_FRAME_SIZE * 2  # output: oldest + newest

        # Temporal ring buffer on GPU: (temporal_skip+1) frames
        self.img_buffer_20hz = cp.zeros(
            (self.temporal_skip + 1) * self.frame_size, dtype=cp.uint8)

        # GPU buffers for warped Y/U/V planes
        self.y_warped = cp.zeros(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=cp.uint8)
        self.u_warped = cp.zeros(
            (self.MODEL_HEIGHT // 2, self.MODEL_WIDTH // 2), dtype=cp.uint8)
        self.v_warped = cp.zeros(
            (self.MODEL_HEIGHT // 2, self.MODEL_WIDTH // 2), dtype=cp.uint8)

    def prepare(self, buf_data, width, height, stride, uv_offset, projection):
        """Process a raw NV12 frame into model input format.

        Args:
            buf_data: numpy uint8 array - raw NV12 frame data (flat)
            width: int - frame width in pixels
            height: int - frame height in pixels
            stride: int - row stride in bytes
            uv_offset: int - byte offset to interleaved UV plane
            projection: numpy float32 array (3,3) or (9,) - warp matrix

        Returns:
            numpy uint8 array (MODEL_FRAME_SIZE * 2,) - two packed frames
            [oldest temporal frame | newest temporal frame]
        """
        projection = np.asarray(projection, dtype=np.float32).reshape(3, 3)

        # Upload source frame to GPU
        src_gpu = cp.asarray(np.ascontiguousarray(buf_data).view(np.uint8).ravel())

        # Compute UV projection (half-scale)
        projection_uv = transform_scale_buffer(projection, 0.5)

        # Upload projection matrices to GPU
        M_y_gpu = cp.asarray(projection.ravel(), dtype=cp.float32)
        M_uv_gpu = cp.asarray(projection_uv.ravel(), dtype=cp.float32)

        # --- Warp Y plane ---
        _warp_perspective(
            src_gpu, stride, 1, 0,
            height, width,
            self.y_warped, self.MODEL_WIDTH, 0,
            self.MODEL_HEIGHT, self.MODEL_WIDTH,
            M_y_gpu
        )

        # --- Warp U plane (interleaved UV, px_stride=2, offset=uv_offset) ---
        _warp_perspective(
            src_gpu, stride, 2, uv_offset,
            height // 2, width // 2,
            self.u_warped, self.MODEL_WIDTH // 2, 0,
            self.MODEL_HEIGHT // 2, self.MODEL_WIDTH // 2,
            M_uv_gpu
        )

        # --- Warp V plane (interleaved UV, px_stride=2, offset=uv_offset+1) ---
        _warp_perspective(
            src_gpu, stride, 2, uv_offset + 1,
            height // 2, width // 2,
            self.v_warped, self.MODEL_WIDTH // 2, 0,
            self.MODEL_HEIGHT // 2, self.MODEL_WIDTH // 2,
            M_uv_gpu
        )

        # --- Shift temporal buffer (move each slot forward by one) ---
        fs = self.frame_size
        for i in range(self.temporal_skip):
            self.img_buffer_20hz[i * fs:(i + 1) * fs] = \
                self.img_buffer_20hz[(i + 1) * fs:(i + 2) * fs]

        # --- Pack YUV into the last slot ---
        packed = loadyuv(self.y_warped, self.u_warped, self.v_warped)
        self.img_buffer_20hz[self.temporal_skip * fs:(self.temporal_skip + 1) * fs] = packed

        # --- Output: oldest frame (slot 0) + newest frame (last slot) ---
        output = cp.empty(self.buf_size, dtype=cp.uint8)
        output[:fs] = self.img_buffer_20hz[:fs]
        output[fs:] = self.img_buffer_20hz[self.temporal_skip * fs:(self.temporal_skip + 1) * fs]

        return cp.asnumpy(output)


class MonitoringModelFrame:
    """CUDA replacement for the OpenCL MonitoringModelFrame.

    Warps the Y plane only for the driver monitoring model.
    """
    MODEL_WIDTH = 1440
    MODEL_HEIGHT = 960
    MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT  # 1382400 (Y-only)

    def __init__(self):
        self.buf_size = self.MODEL_FRAME_SIZE
        # GPU buffer for warped Y plane
        self.y_warped = cp.zeros(
            (self.MODEL_HEIGHT, self.MODEL_WIDTH), dtype=cp.uint8)

    def prepare(self, buf_data, width, height, stride, uv_offset, projection):
        """Process a raw NV12 frame into driver monitoring model input.

        Args:
            buf_data: numpy uint8 array - raw NV12 frame data (flat)
            width: int - frame width in pixels
            height: int - frame height in pixels
            stride: int - row stride in bytes
            uv_offset: int - byte offset to UV plane (unused, Y-only)
            projection: numpy float32 array (3,3) or (9,) - warp matrix

        Returns:
            numpy uint8 array (MODEL_FRAME_SIZE,) - warped Y plane
        """
        projection = np.asarray(projection, dtype=np.float32).reshape(3, 3)

        # Upload source frame to GPU
        src_gpu = cp.asarray(np.ascontiguousarray(buf_data).view(np.uint8).ravel())

        # Upload projection matrix to GPU
        M_gpu = cp.asarray(projection.ravel(), dtype=cp.float32)

        # Warp Y plane only
        _warp_perspective(
            src_gpu, stride, 1, 0,
            height, width,
            self.y_warped, self.MODEL_WIDTH, 0,
            self.MODEL_HEIGHT, self.MODEL_WIDTH,
            M_gpu
        )

        return cp.asnumpy(self.y_warped.ravel())
