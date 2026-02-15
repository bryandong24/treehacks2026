#!/usr/bin/env python3
"""Test driving model pipeline with live USB webcam input.

Captures frames from a USB webcam, converts to NV12, runs through
the full CUDA preprocessing -> ONNX vision -> ONNX policy pipeline,
and prints model outputs.

The model outputs won't be meaningful since the webcam isn't mounted
like a car camera, but this verifies the full pipeline works with
real image data from a real sensor.
"""

import os
import sys
import time
import pickle
import numpy as np
import cv2

# Ensure LD_LIBRARY_PATH for CuPy
os.environ.setdefault('LD_LIBRARY_PATH',
    '/home/subha/treehacks2026/sunnypilot/.venv/lib:/usr/local/cuda/lib64')

import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, '/home/subha/treehacks2026/sunnypilot')
from selfdrive.modeld.transforms.cuda_transforms import DrivingModelFrame, transform_scale_buffer
from selfdrive.modeld.parse_model_outputs import Parser
from selfdrive.modeld.constants import ModelConstants, Plan

MODEL_DIR = Path(__file__).parent / 'models'


def bgr_to_nv12(bgr_frame):
    """Convert a BGR OpenCV frame to NV12 format (Y plane + interleaved UV)."""
    h, w = bgr_frame.shape[:2]
    # Convert BGR -> YUV I420 via OpenCV: output shape is (h*3//2, w)
    yuv_i420 = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
    # I420 layout (all contiguous, shape h*3//2 x w):
    #   rows [0 : h)          -> Y plane
    #   rows [h : h+h//4)     -> U plane (h/2 * w/2 stored as h/4 rows of w)
    #   rows [h+h//4 : h*3//2) -> V plane (same packing)
    y_plane = yuv_i420[:h, :]
    u_plane = yuv_i420[h:h + h // 4, :].reshape(h // 2, w // 2)
    v_plane = yuv_i420[h + h // 4:, :].reshape(h // 2, w // 2)

    # NV12: Y plane followed by interleaved UV
    uv_interleaved = np.empty((h // 2, w), dtype=np.uint8)
    uv_interleaved[:, 0::2] = u_plane
    uv_interleaved[:, 1::2] = v_plane

    nv12 = np.concatenate([y_plane.ravel(), uv_interleaved.ravel()])
    return nv12


def make_webcam_warp_matrix(cam_w, cam_h, cam_fl):
    """Create a warp matrix mapping model coords -> webcam coords.

    Uses medmodel intrinsics (what the model expects) and webcam intrinsics
    with zero calibration (camera = device = calib frame).
    """
    from openpilot.common.transformations.camera import view_frame_from_device_frame
    from openpilot.common.transformations.model import calib_from_medmodel

    # Webcam pinhole intrinsics
    cam_K = np.array([
        [cam_fl, 0.0,    cam_w / 2.0],
        [0.0,    cam_fl, cam_h / 2.0],
        [0.0,    0.0,    1.0],
    ], dtype=np.float64)

    # camera_from_calib = K @ view_from_device @ device_from_calib
    # With zero calibration, device_from_calib = I
    camera_from_calib = cam_K @ view_frame_from_device_frame  # (3x3)

    # warp = camera_from_calib @ calib_from_model
    warp = camera_from_calib @ calib_from_medmodel
    return warp.astype(np.float32)


def main():
    print("=" * 60)
    print("  Webcam -> Driving Model Pipeline Test")
    print("=" * 60)

    # --- Open webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam at /dev/video0")
        return 1

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {cam_w}x{cam_h}")

    # Estimate focal length (~65 deg HFOV typical USB webcam)
    cam_fl = cam_w / (2 * np.tan(np.radians(32.5)))
    print(f"Estimated focal length: {cam_fl:.1f}px (assuming ~65 deg HFOV)")

    # --- Load model metadata ---
    print("\nLoading model metadata...")
    with open(MODEL_DIR / 'driving_vision_metadata.pkl', 'rb') as f:
        vision_meta = pickle.load(f)
    with open(MODEL_DIR / 'driving_policy_metadata.pkl', 'rb') as f:
        policy_meta = pickle.load(f)

    vision_input_shapes = vision_meta['input_shapes']
    vision_output_slices = vision_meta['output_slices']
    policy_input_shapes = policy_meta['input_shapes']
    policy_output_slices = policy_meta['output_slices']

    print(f"  Vision inputs: {list(vision_input_shapes.keys())} -> shapes {list(vision_input_shapes.values())}")
    print(f"  Policy inputs: {list(policy_input_shapes.keys())} -> shapes {list(policy_input_shapes.values())}")

    # --- Load ONNX models ---
    print("\nLoading ONNX models with CUDAExecutionProvider...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cuda_opts = {'cudnn_conv_algo_search': 'EXHAUSTIVE'}
    providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']

    t0 = time.monotonic()
    vision_sess = ort.InferenceSession(str(MODEL_DIR / 'driving_vision.onnx'), sess_opts, providers=providers)
    policy_sess = ort.InferenceSession(str(MODEL_DIR / 'driving_policy.onnx'), sess_opts, providers=providers)
    print(f"Models loaded in {time.monotonic() - t0:.1f}s")

    # --- CUDA preprocessing ---
    temporal_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4
    frames = {name: DrivingModelFrame(temporal_skip) for name in vision_input_shapes}

    # Warp matrix
    warp = make_webcam_warp_matrix(cam_w, cam_h, cam_fl)
    print(f"\nWarp matrix:\n{warp}")

    # --- Policy state ---
    parser = Parser()
    policy_np = {k: np.zeros(policy_input_shapes[k], dtype=np.float32) for k in policy_input_shapes}
    # Left-hand traffic (US/Europe)
    policy_np['traffic_convention'][0, 0] = 1.0

    # Simple features buffer: shift oldest out, append newest
    fb_shape = policy_input_shapes['features_buffer']  # e.g. [1, 25, 512]
    fb_len = fb_shape[1]  # temporal length

    # NV12 params
    stride = cam_w
    uv_offset = cam_w * cam_h

    def slice_outputs(raw, slices):
        return {k: raw[np.newaxis, v] for k, v in slices.items()}

    # --- Run inference loop ---
    n_frames = 30
    latencies = []
    print(f"\n{'='*60}")
    print(f"  Running {n_frames} frames of webcam -> model inference")
    print(f"{'='*60}\n")

    for i in range(n_frames):
        ret, bgr = cap.read()
        if not ret:
            print(f"  Frame {i}: FAILED to read from webcam")
            continue

        nv12 = bgr_to_nv12(bgr)

        t1 = time.perf_counter()

        # --- CUDA preprocess both camera inputs (use same webcam for img + big_img) ---
        imgs = {}
        for name in vision_input_shapes:
            imgs[name] = frames[name].prepare(
                nv12, cam_w, cam_h, stride, uv_offset,
                warp.flatten()
            ).reshape(vision_input_shapes[name])

        # --- Vision inference ---
        vision_result = vision_sess.run(None, imgs)
        vision_raw = vision_result[0].astype(np.float32).flatten()
        vision_dict = parser.parse_vision_outputs(slice_outputs(vision_raw, vision_output_slices))

        # --- Update features buffer ---
        hidden = vision_dict['hidden_state'].flatten()[:ModelConstants.FEATURE_LEN]
        policy_np['features_buffer'][0, :-1] = policy_np['features_buffer'][0, 1:]
        policy_np['features_buffer'][0, -1] = hidden

        # --- Policy inference ---
        policy_feeds = {k: policy_np[k].astype(np.float16) for k in policy_np}
        policy_result = policy_sess.run(None, policy_feeds)
        policy_raw = policy_result[0].astype(np.float32).flatten()
        policy_dict = parser.parse_policy_outputs(slice_outputs(policy_raw, policy_output_slices))

        t2 = time.perf_counter()
        latency_ms = (t2 - t1) * 1000
        latencies.append(latency_ms)

        combined = {**vision_dict, **policy_dict}

        # Print outputs periodically
        if i % 5 == 0 or i == n_frames - 1:
            print(f"--- Frame {i:3d} | {latency_ms:6.1f} ms ---")

            plan = combined.get('plan')
            if plan is not None:
                # plan shape: (1, IDX_N=33, PLAN_WIDTH=15)
                # Plan.VELOCITY = slice(3,6), Plan.ACCELERATION = slice(6,9)
                vel = plan[0, :4, 3]  # velocity x at first 4 time steps
                acc = plan[0, :4, 6]  # accel x at first 4 time steps
                print(f"  Plan velocity  @t0..t3:     [{', '.join(f'{v:6.2f}' for v in vel)}]")
                print(f"  Plan accel     @t0..t3:     [{', '.join(f'{a:6.2f}' for a in acc)}]")

            ll_prob = combined.get('lane_lines_prob')
            if ll_prob is not None:
                print(f"  Lane line probs (4 lines):  [{', '.join(f'{p:.3f}' for p in ll_prob.flatten()[:4])}]")

            lead_prob = combined.get('lead_prob')
            if lead_prob is not None:
                print(f"  Lead vehicle probs:         [{', '.join(f'{p:.3f}' for p in lead_prob.flatten()[:3])}]")

            meta = combined.get('meta')
            if meta is not None:
                engaged = meta.flatten()[0]
                print(f"  Engaged prob:               {engaged:.3f}")

            pose = combined.get('pose')
            if pose is not None:
                # pose shape: (1, 6) — [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
                trans = pose[0, :3]
                rot = pose[0, 3:]
                print(f"  Pose translation (x,y,z):   [{', '.join(f'{v:7.3f}' for v in trans)}]")
                print(f"  Pose rotation (r,p,y):      [{', '.join(f'{v:7.4f}' for v in rot)}]")

            desire = combined.get('desire_state')
            if desire is not None:
                top_desire = np.argmax(desire.flatten())
                print(f"  Desire state (top class):   {top_desire}")

            print()

    cap.release()

    # --- Summary ---
    if len(latencies) > 2:
        arr = np.array(latencies[2:])  # skip first 2 warmup
        print(f"{'='*60}")
        print(f"  LATENCY SUMMARY (excluding 2 warmup frames)")
        print(f"{'='*60}")
        print(f"  Frames: {len(arr)}")
        print(f"  Avg:    {arr.mean():.1f} ms")
        print(f"  Min:    {arr.min():.1f} ms")
        print(f"  Max:    {arr.max():.1f} ms")
        print(f"  p95:    {np.percentile(arr, 95):.1f} ms")
        print(f"  Stddev: {arr.std():.1f} ms")
        budget = 50.0
        print(f"  Budget: {budget:.0f} ms (20 Hz)")
        print(f"  Within budget: {'YES' if arr.max() < budget else 'NO'}")
        print()
        print("  Note: Model outputs are NOT meaningful — webcam is not")
        print("  mounted like a car camera. This verifies the full pipeline")
        print("  works with real sensor data end-to-end on CUDA.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
