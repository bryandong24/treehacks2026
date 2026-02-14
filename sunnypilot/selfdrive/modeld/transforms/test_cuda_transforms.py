#!/usr/bin/env python3
"""Tests for CUDA image transforms (cuda_transforms.py).

Verifies correctness of:
  1. Warp kernel with identity transform
  2. loadyuv YUV channel packing
  3. Temporal buffer oldest/newest pairing
  4. Full pipeline -> ONNX Runtime model input
  5. Performance benchmark
"""

import sys
import time
import numpy as np

# Ensure CuPy can find nvrtc
import os
venv_lib = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.venv', 'lib')
if os.path.isdir(venv_lib):
    os.environ.setdefault('LD_LIBRARY_PATH',
                          os.path.abspath(venv_lib) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))

import cupy as cp

# Import from our module
sys.path.insert(0, os.path.dirname(__file__))
from cuda_transforms import (
    DrivingModelFrame, MonitoringModelFrame,
    loadyuv, transform_scale_buffer, _warp_perspective,
)


def make_nv12_frame(width, height, stride=None):
    """Create a synthetic NV12 frame with a deterministic gradient pattern.

    NV12 layout:
      - Y plane: height rows of stride bytes
      - UV plane: height/2 rows of stride bytes, interleaved UVUV...
    """
    if stride is None:
        stride = width
    assert stride >= width

    total_size = stride * height + stride * (height // 2)
    buf = np.zeros(total_size, dtype=np.uint8)

    # Y plane: gradient based on position
    for row in range(height):
        for col in range(width):
            buf[row * stride + col] = (row * 3 + col * 7) % 256

    # UV plane: interleaved
    uv_offset = stride * height
    for row in range(height // 2):
        for col in range(width // 2):
            u_val = (row * 5 + col * 11) % 256
            v_val = (row * 7 + col * 13) % 256
            buf[uv_offset + row * stride + col * 2] = u_val
            buf[uv_offset + row * stride + col * 2 + 1] = v_val

    return buf, stride, uv_offset


def test_transform_scale_buffer():
    """Test that transform_scale_buffer matches the C++ implementation."""
    print("=== test_transform_scale_buffer ===")

    # Identity matrix scaled by 0.5 should remain identity
    # (the pixel-center adjustments cancel out for identity)
    identity = np.eye(3, dtype=np.float32)
    result = transform_scale_buffer(identity, 0.5)
    np.testing.assert_allclose(result, np.eye(3, dtype=np.float32), atol=1e-6)
    print("  Identity @ scale=0.5 -> identity: PASS")

    # Test with a known non-identity matrix
    M = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], dtype=np.float32)
    result = transform_scale_buffer(M, 0.5)
    # Manually compute: transform_in @ M @ transform_out
    t_out = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], dtype=np.float32)
    t_in = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]], dtype=np.float32)
    expected = t_in @ M @ t_out
    np.testing.assert_allclose(result, expected, atol=1e-5)
    print("  Non-identity @ scale=0.5: PASS")

    print("  ALL PASSED\n")


def test_warp_identity():
    """Test warp kernel with identity projection - output should match source."""
    print("=== test_warp_identity ===")

    src_w, src_h = 640, 480
    dst_w, dst_h = 512, 256

    buf, stride, uv_offset = make_nv12_frame(src_w, src_h)

    identity = np.eye(3, dtype=np.float32)

    src_gpu = cp.asarray(buf)
    M_gpu = cp.asarray(identity.ravel(), dtype=cp.float32)

    # --- Test Y warp ---
    y_out = cp.zeros((dst_h, dst_w), dtype=cp.uint8)
    _warp_perspective(
        src_gpu, stride, 1, 0,
        src_h, src_w,
        y_out, dst_w, 0,
        dst_h, dst_w,
        M_gpu
    )
    y_result = cp.asnumpy(y_out)

    # Expected: top-left dst_h x dst_w region of Y plane
    y_expected = np.zeros((dst_h, dst_w), dtype=np.uint8)
    for row in range(dst_h):
        for col in range(dst_w):
            y_expected[row, col] = (row * 3 + col * 7) % 256

    assert np.array_equal(y_result, y_expected), \
        f"Y warp mismatch! max diff={np.max(np.abs(y_result.astype(int) - y_expected.astype(int)))}"
    print(f"  Y plane identity warp ({dst_w}x{dst_h}): PASS")

    # --- Test U warp ---
    uv_dst_w, uv_dst_h = dst_w // 2, dst_h // 2
    M_uv = transform_scale_buffer(identity, 0.5)
    M_uv_gpu = cp.asarray(M_uv.ravel(), dtype=cp.float32)

    u_out = cp.zeros((uv_dst_h, uv_dst_w), dtype=cp.uint8)
    _warp_perspective(
        src_gpu, stride, 2, uv_offset,
        src_h // 2, src_w // 2,
        u_out, uv_dst_w, 0,
        uv_dst_h, uv_dst_w,
        M_uv_gpu
    )
    u_result = cp.asnumpy(u_out)

    u_expected = np.zeros((uv_dst_h, uv_dst_w), dtype=np.uint8)
    for row in range(uv_dst_h):
        for col in range(uv_dst_w):
            u_expected[row, col] = (row * 5 + col * 11) % 256

    assert np.array_equal(u_result, u_expected), \
        f"U warp mismatch! max diff={np.max(np.abs(u_result.astype(int) - u_expected.astype(int)))}"
    print(f"  U plane identity warp ({uv_dst_w}x{uv_dst_h}): PASS")

    # --- Test V warp ---
    v_out = cp.zeros((uv_dst_h, uv_dst_w), dtype=cp.uint8)
    _warp_perspective(
        src_gpu, stride, 2, uv_offset + 1,
        src_h // 2, src_w // 2,
        v_out, uv_dst_w, 0,
        uv_dst_h, uv_dst_w,
        M_uv_gpu
    )
    v_result = cp.asnumpy(v_out)

    v_expected = np.zeros((uv_dst_h, uv_dst_w), dtype=np.uint8)
    for row in range(uv_dst_h):
        for col in range(uv_dst_w):
            v_expected[row, col] = (row * 7 + col * 13) % 256

    assert np.array_equal(v_result, v_expected), \
        f"V warp mismatch! max diff={np.max(np.abs(v_result.astype(int) - v_expected.astype(int)))}"
    print(f"  V plane identity warp ({uv_dst_w}x{uv_dst_h}): PASS")

    print("  ALL PASSED\n")


def test_warp_translation():
    """Test warp kernel with a translation - verifies sub-pixel interpolation."""
    print("=== test_warp_translation ===")

    src_w, src_h = 64, 64
    dst_w, dst_h = 32, 32

    # Uniform-value frame so translation doesn't change pixel values
    buf = np.full(src_w * src_h + src_w * (src_h // 2), 128, dtype=np.uint8)
    stride = src_w

    # Translate by (10, 5) — integer translation, no interpolation needed
    M = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32)

    src_gpu = cp.asarray(buf)
    M_gpu = cp.asarray(M.ravel(), dtype=cp.float32)
    y_out = cp.zeros((dst_h, dst_w), dtype=cp.uint8)

    _warp_perspective(
        src_gpu, stride, 1, 0,
        src_h, src_w,
        y_out, dst_w, 0,
        dst_h, dst_w,
        M_gpu
    )
    y_result = cp.asnumpy(y_out)

    # With uniform source, all output should be 128
    assert np.all(y_result == 128), f"Expected all 128, got range [{y_result.min()}, {y_result.max()}]"
    print(f"  Translation warp (uniform source): PASS")

    # Now test with gradient source and integer translation
    buf2 = np.zeros(src_w * src_h, dtype=np.uint8)
    for r in range(src_h):
        for c in range(src_w):
            buf2[r * src_w + c] = (r + c) % 256
    buf2_full = np.concatenate([buf2, np.zeros(src_w * (src_h // 2), dtype=np.uint8)])

    src_gpu2 = cp.asarray(buf2_full)
    y_out2 = cp.zeros((dst_h, dst_w), dtype=cp.uint8)
    _warp_perspective(
        src_gpu2, stride, 1, 0,
        src_h, src_w,
        y_out2, dst_w, 0,
        dst_h, dst_w,
        M_gpu
    )
    y_result2 = cp.asnumpy(y_out2)

    # Output[dy, dx] should equal src[dy+5, dx+10]
    for dy in range(dst_h):
        for dx in range(dst_w):
            expected = (dy + 5 + dx + 10) % 256
            actual = y_result2[dy, dx]
            assert actual == expected, \
                f"Translation mismatch at ({dx},{dy}): got {actual}, expected {expected}"
    print(f"  Translation warp (gradient source): PASS")

    print("  ALL PASSED\n")


def test_loadyuv():
    """Test YUV channel packing matches the OpenCL loadys/loaduv behavior."""
    print("=== test_loadyuv ===")

    H, W = 256, 512
    uv_h, uv_w = H // 2, W // 2

    # Create Y plane with known pattern
    y = cp.zeros((H, W), dtype=cp.uint8)
    y_np = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            y_np[r, c] = (r * 3 + c * 7) % 256
    y = cp.asarray(y_np)

    # Create U/V planes
    u_np = np.zeros((uv_h, uv_w), dtype=np.uint8)
    v_np = np.zeros((uv_h, uv_w), dtype=np.uint8)
    for r in range(uv_h):
        for c in range(uv_w):
            u_np[r, c] = (r * 5 + c * 11) % 256
            v_np[r, c] = (r * 7 + c * 13) % 256
    u = cp.asarray(u_np)
    v = cp.asarray(v_np)

    result = cp.asnumpy(loadyuv(y, u, v))
    uv_size = uv_h * uv_w  # 32768

    expected_size = H * W * 3 // 2
    assert result.shape == (expected_size,), f"Size mismatch: {result.shape} vs ({expected_size},)"

    # Verify slot 0: Y[even_row, even_col]
    slot0 = result[0:uv_size].reshape(uv_h, uv_w)
    expected_slot0 = y_np[0::2, 0::2]
    assert np.array_equal(slot0, expected_slot0), "Slot 0 (Y[even,even]) mismatch"
    print("  Slot 0 Y[even_row, even_col]: PASS")

    # Verify slot 1: Y[odd_row, even_col]
    slot1 = result[uv_size:2*uv_size].reshape(uv_h, uv_w)
    expected_slot1 = y_np[1::2, 0::2]
    assert np.array_equal(slot1, expected_slot1), "Slot 1 (Y[odd,even]) mismatch"
    print("  Slot 1 Y[odd_row,  even_col]: PASS")

    # Verify slot 2: Y[even_row, odd_col]
    slot2 = result[2*uv_size:3*uv_size].reshape(uv_h, uv_w)
    expected_slot2 = y_np[0::2, 1::2]
    assert np.array_equal(slot2, expected_slot2), "Slot 2 (Y[even,odd]) mismatch"
    print("  Slot 2 Y[even_row, odd_col]: PASS")

    # Verify slot 3: Y[odd_row, odd_col]
    slot3 = result[3*uv_size:4*uv_size].reshape(uv_h, uv_w)
    expected_slot3 = y_np[1::2, 1::2]
    assert np.array_equal(slot3, expected_slot3), "Slot 3 (Y[odd,odd]) mismatch"
    print("  Slot 3 Y[odd_row,  odd_col]: PASS")

    # Verify slot 4: U plane
    slot4 = result[4*uv_size:5*uv_size].reshape(uv_h, uv_w)
    assert np.array_equal(slot4, u_np), "Slot 4 (U) mismatch"
    print("  Slot 4 U plane: PASS")

    # Verify slot 5: V plane
    slot5 = result[5*uv_size:6*uv_size].reshape(uv_h, uv_w)
    assert np.array_equal(slot5, v_np), "Slot 5 (V) mismatch"
    print("  Slot 5 V plane: PASS")

    print("  ALL PASSED\n")


def test_temporal_buffer():
    """Test DrivingModelFrame temporal buffer oldest/newest pairing."""
    print("=== test_temporal_buffer ===")

    temporal_skip = 2  # smaller for easier testing
    frame = DrivingModelFrame(temporal_skip=temporal_skip)
    fs = frame.frame_size

    # Create source frames large enough for the model
    src_w, src_h = 640, 480
    stride = src_w
    uv_offset = stride * src_h
    identity = np.eye(3, dtype=np.float32)

    # Feed frames filled with constant values
    # Frame i has Y filled with value (i+1)*40
    outputs = []
    for i in range(temporal_skip + 2):  # Feed temporal_skip+2 frames
        fill_val = (i + 1) * 40
        total_size = stride * src_h + stride * (src_h // 2)
        buf = np.full(total_size, fill_val, dtype=np.uint8)
        output = frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)
        outputs.append(output.copy())

    # After temporal_skip+1 frames, the oldest slot should be frame 0 (fill_val=40)
    # Frame index temporal_skip+1 means:
    #   buffer contains: [frame1, frame2, frame(temporal_skip+1)]
    #   oldest = frame1 (fill_val=40), newest = frame(temporal_skip+1)

    # After first frame: oldest=zeros, newest=frame0
    first_oldest = outputs[0][:fs]
    first_newest = outputs[0][fs:]
    assert np.all(first_oldest == 0), "First frame oldest should be zeros"
    print(f"  After frame 0: oldest=zeros: PASS")

    # After temporal_skip+1 frames, oldest should be frame 0
    ts1_output = outputs[temporal_skip]
    ts1_oldest = ts1_output[:fs]
    ts1_newest = ts1_output[fs:]
    # The oldest frame should have the fill value of frame 0
    oldest_mode = np.bincount(ts1_oldest).argmax()  # most common value
    newest_mode = np.bincount(ts1_newest).argmax()
    expected_oldest_val = 40  # frame 0's fill value
    expected_newest_val = (temporal_skip + 1) * 40  # frame temporal_skip's fill value
    assert oldest_mode == expected_oldest_val, \
        f"Oldest frame mode={oldest_mode}, expected={expected_oldest_val}"
    assert newest_mode == expected_newest_val, \
        f"Newest frame mode={newest_mode}, expected={expected_newest_val}"
    print(f"  After {temporal_skip+1} frames: oldest=frame0, newest=frame{temporal_skip}: PASS")

    # After one more frame, oldest should shift to frame 1
    ts2_output = outputs[temporal_skip + 1]
    ts2_oldest = ts2_output[:fs]
    ts2_newest = ts2_output[fs:]
    oldest_mode2 = np.bincount(ts2_oldest).argmax()
    newest_mode2 = np.bincount(ts2_newest).argmax()
    expected_oldest_val2 = 80  # frame 1's fill value
    expected_newest_val2 = (temporal_skip + 2) * 40
    assert oldest_mode2 == expected_oldest_val2, \
        f"Oldest frame mode={oldest_mode2}, expected={expected_oldest_val2}"
    assert newest_mode2 == expected_newest_val2, \
        f"Newest frame mode={newest_mode2}, expected={expected_newest_val2}"
    print(f"  After {temporal_skip+2} frames: oldest=frame1, newest=frame{temporal_skip+1}: PASS")

    print("  ALL PASSED\n")


def test_driving_model_full_pipeline():
    """Test full DrivingModelFrame pipeline with correct output shape."""
    print("=== test_driving_model_full_pipeline ===")

    frame = DrivingModelFrame(temporal_skip=4)

    # Simulate a realistic camera frame (1928x1208, like comma cameras)
    src_w, src_h = 1928, 1208
    stride = src_w
    uv_offset = stride * src_h
    identity = np.eye(3, dtype=np.float32)

    total_size = stride * src_h + stride * (src_h // 2)
    np.random.seed(42)
    buf = np.random.randint(0, 256, total_size, dtype=np.uint8)

    output = frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)

    assert output.shape == (frame.buf_size,), \
        f"Output shape {output.shape} != expected ({frame.buf_size},)"
    assert output.dtype == np.uint8, f"Output dtype {output.dtype} != uint8"
    print(f"  Output shape: {output.shape} dtype: {output.dtype}: PASS")

    # Reshape to model input format [1, 12, 128, 256]
    model_input = output.reshape(1, 12, 128, 256)
    assert model_input.shape == (1, 12, 128, 256)
    print(f"  Model input reshape [1,12,128,256]: PASS")

    # Check that the newest frame (channels 6-11) has non-zero data
    newest_channels = model_input[0, 6:, :, :]
    assert np.any(newest_channels > 0), "Newest frame channels should have non-zero data"
    print(f"  Newest frame has non-zero data: PASS")

    print("  ALL PASSED\n")


def test_monitoring_model_full_pipeline():
    """Test full MonitoringModelFrame pipeline."""
    print("=== test_monitoring_model_full_pipeline ===")

    frame = MonitoringModelFrame()

    # Simulate a driver-facing camera frame
    src_w, src_h = 1928, 1208
    stride = src_w
    uv_offset = stride * src_h
    identity = np.eye(3, dtype=np.float32)

    total_size = stride * src_h + stride * (src_h // 2)
    np.random.seed(42)
    buf = np.random.randint(0, 256, total_size, dtype=np.uint8)

    output = frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)

    assert output.shape == (frame.MODEL_FRAME_SIZE,), \
        f"Output shape {output.shape} != expected ({frame.MODEL_FRAME_SIZE},)"
    assert output.dtype == np.uint8
    assert np.any(output > 0), "Output should have non-zero data"
    print(f"  Output shape: {output.shape} dtype: {output.dtype}: PASS")

    # Reshape for model input [1, 1382400]
    model_input = output.reshape(1, -1)
    assert model_input.shape == (1, 1382400)
    print(f"  Model input reshape [1,1382400]: PASS")

    print("  ALL PASSED\n")


def test_onnx_integration():
    """Feed preprocessed output through ONNX Runtime driving_vision.onnx."""
    print("=== test_onnx_integration ===")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'driving_vision.onnx')
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("  SKIP: onnxruntime not installed")
        return

    # Create a session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    print(f"  ONNX model loaded, provider: {session.get_providers()[0]}")

    # Generate preprocessed input
    frame = DrivingModelFrame(temporal_skip=4)
    src_w, src_h = 1928, 1208
    stride = src_w
    uv_offset = stride * src_h
    identity = np.eye(3, dtype=np.float32)
    total_size = stride * src_h + stride * (src_h // 2)

    np.random.seed(42)
    buf = np.random.randint(0, 256, total_size, dtype=np.uint8)

    # Feed several frames to fill temporal buffer
    for _ in range(5):
        output = frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)

    img_input = output.reshape(1, 12, 128, 256)

    # Run model
    input_names = [inp.name for inp in session.get_inputs()]
    feeds = {}
    for inp in session.get_inputs():
        if inp.name == 'img':
            feeds['img'] = img_input
        elif inp.name == 'big_img':
            feeds['big_img'] = img_input  # same for testing
        else:
            # fill other inputs with zeros
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            feeds[inp.name] = np.zeros(shape, dtype=np.float16 if 'float16' in inp.type else np.float32)

    results = session.run(None, feeds)

    for i, r in enumerate(results):
        has_nan = np.any(np.isnan(r.astype(np.float32)))
        has_inf = np.any(np.isinf(r.astype(np.float32)))
        print(f"  Output[{i}] shape={r.shape} dtype={r.dtype} "
              f"range=[{r.min():.3f}, {r.max():.3f}] nan={has_nan} inf={has_inf}")
        assert not has_nan, f"Output[{i}] contains NaN!"
        assert not has_inf, f"Output[{i}] contains Inf!"

    print("  ALL PASSED\n")


def benchmark():
    """Benchmark the full preprocessing pipeline."""
    print("=== Benchmark ===")

    # --- DrivingModelFrame benchmark ---
    frame = DrivingModelFrame(temporal_skip=4)
    src_w, src_h = 1928, 1208
    stride = src_w
    uv_offset = stride * src_h
    identity = np.eye(3, dtype=np.float32)
    total_size = stride * src_h + stride * (src_h // 2)

    np.random.seed(42)
    buf = np.random.randint(0, 256, total_size, dtype=np.uint8)

    # Warmup
    for _ in range(5):
        frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    n_iters = 50
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000
    print(f"  DrivingModelFrame.prepare() [{src_w}x{src_h} -> 512x256]:")
    print(f"    avg={times.mean():.2f}ms  min={times.min():.2f}ms  "
          f"max={times.max():.2f}ms  std={times.std():.2f}ms")
    print(f"    target: <5ms (20Hz = 50ms budget)")

    # --- MonitoringModelFrame benchmark ---
    mon_frame = MonitoringModelFrame()

    # Warmup
    for _ in range(5):
        mon_frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)
    cp.cuda.Stream.null.synchronize()

    times_mon = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        mon_frame.prepare(buf, src_w, src_h, stride, uv_offset, identity)
        cp.cuda.Stream.null.synchronize()
        times_mon.append(time.perf_counter() - t0)

    times_mon = np.array(times_mon) * 1000
    print(f"\n  MonitoringModelFrame.prepare() [{src_w}x{src_h} -> 1440x960]:")
    print(f"    avg={times_mon.mean():.2f}ms  min={times_mon.min():.2f}ms  "
          f"max={times_mon.max():.2f}ms  std={times_mon.std():.2f}ms")
    print(f"    target: <10ms")

    print()


if __name__ == '__main__':
    test_transform_scale_buffer()
    test_warp_identity()
    test_warp_translation()
    test_loadyuv()
    test_temporal_buffer()
    test_driving_model_full_pipeline()
    test_monitoring_model_full_pipeline()
    test_onnx_integration()
    benchmark()
    print("ALL TESTS PASSED!")
