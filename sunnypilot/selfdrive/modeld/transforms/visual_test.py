#!/usr/bin/env python3
"""Visual verification of CUDA image transforms.

Captures a frame from a USB webcam (or generates a synthetic scene),
runs it through the preprocessing pipeline, and saves PNGs for inspection.

Usage:
  # With webcam plugged into Jetson Thor:
  python visual_test.py

  # With a specific video device:
  python visual_test.py --device 0

  # Without webcam (synthetic gradient scene):
  python visual_test.py --synthetic

Output files saved to selfdrive/modeld/transforms/visual_output/:
  01_source_bgr.png          - Original camera frame
  02_source_y.png            - Y plane (grayscale)
  03_source_uv.png           - UV planes side by side
  04_warped_y.png            - Warped Y plane (512x256)
  05_warped_u.png            - Warped U plane (256x128)
  06_warped_v.png            - Warped_v plane (256x128)
  07_packed_6ch_tiled.png    - 6 loadyuv channels tiled (4 Y sub-planes + U + V)
  08_model_input_12ch.png    - Full model input [12, 128, 256] tiled
  09_monitoring_y.png        - Monitoring model warped Y (1440x960)
"""

import os
import sys
import argparse
import numpy as np

# CuPy library path setup
venv_lib = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.venv', 'lib')
if os.path.isdir(venv_lib):
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = os.path.abspath(venv_lib) + ':' + ld

import cv2
import cupy as cp

sys.path.insert(0, os.path.dirname(__file__))
from cuda_transforms import (
    DrivingModelFrame, MonitoringModelFrame,
    loadyuv, transform_scale_buffer, _warp_perspective,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'visual_output')


def bgr_to_nv12(bgr):
    """Convert BGR image to NV12 format (Y plane + interleaved UV)."""
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr.shape[:2]

    # I420 layout: Y (h*w), U (h/2 * w/2), V (h/2 * w/2) — all planar
    y = yuv[:h].ravel()
    u = yuv[h:h + h // 4].ravel()
    v = yuv[h + h // 4:].ravel()

    # Convert to NV12: Y plane + interleaved UV
    uv_interleaved = np.empty(len(u) * 2, dtype=np.uint8)
    uv_interleaved[0::2] = u
    uv_interleaved[1::2] = v

    nv12 = np.concatenate([y, uv_interleaved])
    return nv12, w, h, w, w * h  # buf, width, height, stride, uv_offset


def make_synthetic_scene(width=1280, height=720):
    """Create a synthetic scene with clear asymmetry for flip/rotation detection."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background (darker at top, brighter at bottom)
    for r in range(height):
        img[r, :] = [int(r / height * 200), int(r / height * 150), int(r / height * 100)]

    # Draw directional elements so flips/rotations are immediately visible
    # Arrow pointing RIGHT in the upper-left
    cv2.arrowedLine(img, (50, 80), (250, 80), (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(img, "RIGHT ->", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Arrow pointing DOWN on the left side
    cv2.arrowedLine(img, (30, 120), (30, 300), (0, 255, 255), 3, tipLength=0.3)
    cv2.putText(img, "DOWN", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # "TOP-LEFT" label
    cv2.putText(img, "TOP-LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # "BOTTOM-RIGHT" label
    cv2.putText(img, "BOTTOM-RIGHT", (width - 350, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Colored corners for orientation
    cv2.rectangle(img, (0, 0), (40, 40), (0, 0, 255), -1)        # Red = top-left
    cv2.rectangle(img, (width-40, 0), (width, 40), (0, 255, 0), -1)  # Green = top-right
    cv2.rectangle(img, (0, height-40), (40, height), (255, 0, 0), -1)  # Blue = bottom-left
    cv2.rectangle(img, (width-40, height-40), (width, height), (255, 255, 0), -1)  # Cyan = bottom-right

    # Simulated road lines
    cv2.line(img, (width // 2 - 100, height), (width // 2 - 30, height // 2),
             (255, 255, 255), 2)
    cv2.line(img, (width // 2 + 100, height), (width // 2 + 30, height // 2),
             (255, 255, 255), 2)

    return img


def capture_webcam_frame(device=0):
    """Capture a single frame from a USB webcam."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video device {device}. "
                           f"Check /dev/video* and try --device N")

    # Let auto-exposure settle
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame from webcam")

    print(f"  Captured {frame.shape[1]}x{frame.shape[0]} from device {device}")
    return frame


def save_tiled(arrays, labels, path, cols=3):
    """Save multiple grayscale arrays as a single tiled image with labels."""
    n = len(arrays)
    rows = (n + cols - 1) // cols

    # Find max dims
    max_h = max(a.shape[0] for a in arrays) + 20  # space for label
    max_w = max(a.shape[1] for a in arrays)

    canvas = np.zeros((rows * max_h, cols * max_w), dtype=np.uint8)

    for i, (arr, label) in enumerate(zip(arrays, labels)):
        r, c = divmod(i, cols)
        y0, x0 = r * max_h + 20, c * max_w
        h, w = arr.shape
        canvas[y0:y0 + h, x0:x0 + w] = arr
        cv2.putText(canvas, label, (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)

    cv2.imwrite(path, canvas)
    print(f"  Saved: {os.path.basename(path)} ({canvas.shape[1]}x{canvas.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Visual test of CUDA transforms")
    parser.add_argument('--device', type=int, default=0, help='Video device index')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic scene instead of webcam')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # --- Step 1: Get a source frame ---
    if args.synthetic:
        print("Using synthetic scene")
        bgr = make_synthetic_scene(1280, 720)
    else:
        print(f"Capturing from webcam (device {args.device})...")
        bgr = capture_webcam_frame(args.device)

    cv2.imwrite(os.path.join(OUTPUT_DIR, '01_source_bgr.png'), bgr)
    print(f"  Saved: 01_source_bgr.png ({bgr.shape[1]}x{bgr.shape[0]})")

    # --- Step 2: Convert to NV12 ---
    nv12_buf, width, height, stride, uv_offset = bgr_to_nv12(bgr)
    print(f"\nNV12: {width}x{height}, stride={stride}, uv_offset={uv_offset}, total={len(nv12_buf)} bytes")

    # Save Y plane
    y_plane = nv12_buf[:height * stride].reshape(height, stride)[:, :width]
    cv2.imwrite(os.path.join(OUTPUT_DIR, '02_source_y.png'), y_plane)
    print(f"  Saved: 02_source_y.png ({width}x{height})")

    # Save UV visualization
    uv_data = nv12_buf[uv_offset:]
    u_plane = uv_data[0::2].reshape(height // 2, width // 2)
    v_plane = uv_data[1::2].reshape(height // 2, width // 2)
    uv_vis = np.hstack([u_plane, v_plane])
    cv2.imwrite(os.path.join(OUTPUT_DIR, '03_source_uv.png'), uv_vis)
    print(f"  Saved: 03_source_uv.png (U | V)")

    # --- Step 3: Run warp kernels individually ---
    print("\nRunning CUDA warp kernels...")
    identity = np.eye(3, dtype=np.float32)
    src_gpu = cp.asarray(nv12_buf)

    M_y_gpu = cp.asarray(identity.ravel(), dtype=cp.float32)
    M_uv = transform_scale_buffer(identity, 0.5)
    M_uv_gpu = cp.asarray(M_uv.ravel(), dtype=cp.float32)

    # Warp Y
    y_warped = cp.zeros((256, 512), dtype=cp.uint8)
    _warp_perspective(src_gpu, stride, 1, 0, height, width,
                      y_warped, 512, 0, 256, 512, M_y_gpu)
    y_warped_np = cp.asnumpy(y_warped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, '04_warped_y.png'), y_warped_np)
    print(f"  Saved: 04_warped_y.png (512x256)")

    # Warp U
    u_warped = cp.zeros((128, 256), dtype=cp.uint8)
    _warp_perspective(src_gpu, stride, 2, uv_offset, height // 2, width // 2,
                      u_warped, 256, 0, 128, 256, M_uv_gpu)
    u_warped_np = cp.asnumpy(u_warped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, '05_warped_u.png'), u_warped_np)
    print(f"  Saved: 05_warped_u.png (256x128)")

    # Warp V
    v_warped = cp.zeros((128, 256), dtype=cp.uint8)
    _warp_perspective(src_gpu, stride, 2, uv_offset + 1, height // 2, width // 2,
                      v_warped, 256, 0, 128, 256, M_uv_gpu)
    v_warped_np = cp.asnumpy(v_warped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, '06_warped_v.png'), v_warped_np)
    print(f"  Saved: 06_warped_v.png (256x128)")

    # --- Step 4: loadyuv 6-channel tiled ---
    print("\nPacking YUV channels (loadyuv)...")
    packed = cp.asnumpy(loadyuv(
        cp.asarray(y_warped_np), cp.asarray(u_warped_np), cp.asarray(v_warped_np)))
    uv_size = 128 * 256  # 32768

    ch_arrays = [
        packed[0*uv_size:1*uv_size].reshape(128, 256),
        packed[1*uv_size:2*uv_size].reshape(128, 256),
        packed[2*uv_size:3*uv_size].reshape(128, 256),
        packed[3*uv_size:4*uv_size].reshape(128, 256),
        packed[4*uv_size:5*uv_size].reshape(128, 256),
        packed[5*uv_size:6*uv_size].reshape(128, 256),
    ]
    ch_labels = [
        "Y[even_r,even_c]", "Y[odd_r,even_c]",
        "Y[even_r,odd_c]", "Y[odd_r,odd_c]",
        "U plane", "V plane",
    ]
    save_tiled(ch_arrays, ch_labels, os.path.join(OUTPUT_DIR, '07_packed_6ch_tiled.png'), cols=3)

    # --- Step 5: Full DrivingModelFrame pipeline ---
    print("\nRunning full DrivingModelFrame pipeline...")
    frame = DrivingModelFrame(temporal_skip=4)

    # Feed 5 frames to fill temporal buffer
    for i in range(5):
        output = frame.prepare(nv12_buf, width, height, stride, uv_offset, identity)

    # Reshape to [12, 128, 256] and tile all channels
    model_input = output.reshape(12, 128, 256)
    mi_arrays = [model_input[i] for i in range(12)]
    mi_labels = [
        "old Y00", "old Y10", "old Y01", "old Y11", "old U", "old V",
        "new Y00", "new Y10", "new Y01", "new Y11", "new U", "new V",
    ]
    save_tiled(mi_arrays, mi_labels, os.path.join(OUTPUT_DIR, '08_model_input_12ch.png'), cols=6)

    # --- Step 6: MonitoringModelFrame ---
    print("\nRunning MonitoringModelFrame pipeline...")
    mon_frame = MonitoringModelFrame()
    mon_output = mon_frame.prepare(nv12_buf, width, height, stride, uv_offset, identity)
    mon_y = mon_output.reshape(960, 1440)
    cv2.imwrite(os.path.join(OUTPUT_DIR, '09_monitoring_y.png'), mon_y)
    print(f"  Saved: 09_monitoring_y.png (1440x960)")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"View them with: scp <jetson>:{OUTPUT_DIR}/*.png .")
    print(f"Or start a web server: cd {OUTPUT_DIR} && python -m http.server 8080")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
