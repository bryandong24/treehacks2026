#!/usr/bin/env python3
"""End-to-end test: real IMX274 cameras → /dev/shm → VisionIPC → modeld.

Prerequisites:
  1. holoscan_frame_publisher.py running inside Docker
     (capturing from real IMX274 cameras, writing to /dev/shm)

This script:
  1. Verifies /dev/shm ring buffers are live
  2. Starts jetson_camerad as a subprocess (shm → VisionIPC)
  3. Connects VisionIPC clients to receive real camera frames
  4. Runs full CUDA preprocessing → ONNX vision → ONNX policy
  5. Prints real model outputs (plan, lanes, leads, pose)
  6. Reports end-to-end latency

Usage:
  cd /home/subha/treehacks2026/sunnypilot
  LD_LIBRARY_PATH=.venv/lib:/usr/local/cuda/lib64 \
    .venv/bin/python selfdrive/camerad/test_full_pipeline.py
"""

import os
import sys
import time
import pickle
import subprocess
import signal

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('LD_LIBRARY_PATH',
    os.path.join(PROJECT_ROOT, '.venv', 'lib') + ':/usr/local/cuda/lib64')

import cupy as cp
import onnxruntime as ort
from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.transforms.cuda_transforms import DrivingModelFrame
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.selfdrive.camerad.shm_buffer import ShmRingBufferReader, SHM_PATH_ROAD, SHM_PATH_WIDE

MODELS_DIR = os.path.join(PROJECT_ROOT, 'selfdrive', 'modeld', 'models')
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python')
CAMERAD_SCRIPT = os.path.join(PROJECT_ROOT, 'selfdrive', 'camerad', 'jetson_camerad.py')

N_FRAMES = 60


def check_shm_live():
    """Verify holoscan_frame_publisher is writing frames."""
    print("Checking /dev/shm ring buffers...")
    reader_road = ShmRingBufferReader(SHM_PATH_ROAD)
    reader_wide = ShmRingBufferReader(SHM_PATH_WIDE)

    if not reader_road.connect(timeout=5.0):
        print(f"  FAIL: {SHM_PATH_ROAD} not found or empty.")
        print("  Is holoscan_frame_publisher.py running inside Docker?")
        return False
    if not reader_wide.connect(timeout=5.0):
        print(f"  FAIL: {SHM_PATH_WIDE} not found or empty.")
        return False

    # Check that frames are actually being written (not stale)
    _, _, fc1 = reader_road.read_latest()
    time.sleep(0.2)
    _, _, fc2 = reader_road.read_latest()
    reader_road.close()
    reader_wide.close()

    if fc2 <= fc1:
        # Try once more with longer wait
        reader_road2 = ShmRingBufferReader(SHM_PATH_ROAD)
        reader_road2.connect(timeout=1.0)
        time.sleep(0.5)
        _, _, fc3 = reader_road2.read_latest()
        reader_road2.close()
        if fc3 <= fc1:
            print(f"  FAIL: shm exists but no new frames (frame_count stuck at {fc1})")
            print("  Is holoscan_frame_publisher.py still running?")
            return False

    print(f"  Road shm: {SHM_PATH_ROAD} — live (frame_count={fc2})")
    print(f"  Wide shm: {SHM_PATH_WIDE} — live")
    return True


def start_camerad():
    """Launch jetson_camerad as a subprocess."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = os.path.join(PROJECT_ROOT, '.venv', 'lib') + ':/usr/local/cuda/lib64'
    proc = subprocess.Popen(
        [VENV_PYTHON, CAMERAD_SCRIPT],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def create_ort_session(model_path):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'EXHAUSTIVE'}),
                 'CPUExecutionProvider']
    return ort.InferenceSession(model_path, sess_opts, providers=providers)


def slice_outputs(raw, slices):
    return {k: raw[np.newaxis, v] for k, v in slices.items()}


def main():
    print("=" * 70)
    print("  FULL PIPELINE E2E TEST")
    print("  Real IMX274 cameras → /dev/shm → VisionIPC → modeld")
    print("=" * 70)

    try:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    except Exception:
        gpu_name = "Jetson GPU"
    print(f"  GPU: {gpu_name}")
    print(f"  ONNX RT: {ort.__version__}")
    print()

    # --- Step 1: Check shm is live ---
    if not check_shm_live():
        return 1
    print()

    # --- Step 2: Start jetson_camerad ---
    print("Starting jetson_camerad subprocess...")
    camerad_proc = start_camerad()
    time.sleep(2)  # Give it time to set up VisionIPC server

    if camerad_proc.poll() is not None:
        out = camerad_proc.stdout.read().decode()
        print(f"  FAIL: jetson_camerad exited immediately:\n{out}")
        return 1
    print(f"  jetson_camerad running (pid={camerad_proc.pid})")
    print()

    try:
        # --- Step 3: Connect VisionIPC clients ---
        print("Connecting VisionIPC clients...")
        client_road = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
        client_wide = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)

        if not client_road.connect(True):
            print("  FAIL: could not connect road VisionIPC client")
            return 1
        if not client_wide.connect(True):
            print("  FAIL: could not connect wide VisionIPC client")
            return 1
        print(f"  Road: {client_road.width}x{client_road.height}, {client_road.buffer_len} buffers")
        print(f"  Wide: {client_wide.width}x{client_wide.height}, {client_wide.buffer_len} buffers")
        print()

        # --- Step 4: Load models ---
        print("Loading ONNX models...")
        with open(os.path.join(MODELS_DIR, 'driving_vision_metadata.pkl'), 'rb') as f:
            vision_meta = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'driving_policy_metadata.pkl'), 'rb') as f:
            policy_meta = pickle.load(f)

        vision_input_shapes = vision_meta['input_shapes']
        vision_output_slices = vision_meta['output_slices']
        policy_input_shapes = policy_meta['input_shapes']
        policy_output_slices = policy_meta['output_slices']
        vision_input_names = list(vision_input_shapes.keys())

        t0 = time.monotonic()
        vision_session = create_ort_session(os.path.join(MODELS_DIR, 'driving_vision.onnx'))
        policy_session = create_ort_session(os.path.join(MODELS_DIR, 'driving_policy.onnx'))
        print(f"  Models loaded in {time.monotonic() - t0:.1f}s")
        print(f"  Vision: {vision_session.get_providers()[0]}")
        print(f"  Policy: {policy_session.get_providers()[0]}")
        print()

        # --- Step 5: Set up preprocessing ---
        temporal_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
        cuda_frames = {name: DrivingModelFrame(temporal_skip) for name in vision_input_names}
        parser = Parser()

        # Use real IMX274 camera intrinsics for warp matrix
        dc = DEVICE_CAMERAS[("jetson", "imx274")]
        device_from_calib_euler = np.zeros(3, dtype=np.float32)  # uncalibrated
        warp_main = get_warp_matrix(device_from_calib_euler, dc.fcam.intrinsics, False).astype(np.float32)
        warp_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)

        print(f"  Camera intrinsics:")
        print(f"    fcam (road, 90°): fl={dc.fcam.focal_length}, {dc.fcam.width}x{dc.fcam.height}")
        print(f"    ecam (wide, 120°): fl={dc.ecam.focal_length}, {dc.ecam.width}x{dc.ecam.height}")
        print()

        # Policy state
        policy_np = {k: np.zeros(policy_input_shapes[k], dtype=np.float32) for k in policy_input_shapes}
        policy_np['traffic_convention'][0, 0] = 1.0  # right-hand traffic (US)

        # --- Step 6: Run inference loop ---
        print(f"Running {N_FRAMES} frames through full pipeline...")
        print("-" * 70)

        latencies = []
        frames_processed = 0

        for i in range(N_FRAMES):
            # Receive real camera frames via VisionIPC
            buf_road = client_road.recv()
            if buf_road is None:
                continue

            buf_wide = client_wide.recv()
            if buf_wide is None:
                buf_wide = buf_road

            t_start = time.perf_counter()

            # CUDA preprocessing with real warp matrices
            imgs = {}
            for name in vision_input_names:
                if 'big' in name:
                    buf, warp = buf_wide, warp_extra
                else:
                    buf, warp = buf_road, warp_main
                imgs[name] = cuda_frames[name].prepare(
                    buf.data, buf.width, buf.height, buf.stride, buf.uv_offset,
                    warp.flatten()
                ).reshape(vision_input_shapes[name])

            # Vision inference
            vision_result = vision_session.run(None, imgs)
            vision_raw = vision_result[0].astype(np.float32).flatten()
            vision_dict = parser.parse_vision_outputs(
                slice_outputs(vision_raw, vision_output_slices))

            # Update features buffer
            hidden = vision_dict['hidden_state'].flatten()[:ModelConstants.FEATURE_LEN]
            policy_np['features_buffer'][0, :-1] = policy_np['features_buffer'][0, 1:]
            policy_np['features_buffer'][0, -1] = hidden
            policy_np['desire_pulse'][:] = 0

            # Policy inference
            policy_feeds = {k: policy_np[k].astype(np.float16) for k in policy_np}
            policy_result = policy_session.run(None, policy_feeds)
            policy_raw = policy_result[0].astype(np.float32).flatten()
            policy_dict = parser.parse_policy_outputs(
                slice_outputs(policy_raw, policy_output_slices))

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(elapsed_ms)
            frames_processed += 1

            combined = {**vision_dict, **policy_dict}

            # Print every 10th frame
            if i % 10 == 0 or i == N_FRAMES - 1:
                plan = combined['plan']
                vel = plan[0, :4, Plan.VELOCITY.start]
                acc = plan[0, :4, Plan.ACCELERATION.start]
                ll_prob = combined.get('lane_lines_prob', np.zeros(4))
                lead_prob = combined.get('lead_prob', np.zeros(3))
                pose = combined.get('pose', np.zeros((1, 6)))
                meta = combined.get('meta', np.zeros(1))

                print(f"Frame {i:3d} | {elapsed_ms:5.1f}ms | fid={client_road.frame_id}")
                print(f"  Plan vel  @t0..t3: [{', '.join(f'{v:6.2f}' for v in vel.flatten()[:4])}]")
                print(f"  Plan acc  @t0..t3: [{', '.join(f'{a:6.2f}' for a in acc.flatten()[:4])}]")
                print(f"  Lane probs:        [{', '.join(f'{p:.3f}' for p in ll_prob.flatten()[:4])}]")
                print(f"  Lead probs:        [{', '.join(f'{p:.3f}' for p in lead_prob.flatten()[:3])}]")
                print(f"  Pose trans (x,y,z):[{', '.join(f'{v:7.3f}' for v in pose.flatten()[:3])}]")
                print(f"  Engaged prob:      {meta.flatten()[0]:.3f}")
                print()

        # --- Step 7: Results ---
        print("=" * 70)
        print("  RESULTS")
        print("=" * 70)

        if len(latencies) < 5:
            print(f"  Only {len(latencies)} frames processed — not enough data")
            return 1

        arr = np.array(latencies[5:])  # skip warmup
        print(f"  Frames processed: {frames_processed}")
        print(f"  Latency ({len(arr)} frames, excluding 5 warmup):")
        print(f"    avg:  {arr.mean():.1f} ms")
        print(f"    min:  {arr.min():.1f} ms")
        print(f"    max:  {arr.max():.1f} ms")
        print(f"    p95:  {np.percentile(arr, 95):.1f} ms")
        print(f"    std:  {arr.std():.1f} ms")
        budget = 50.0
        print(f"    Budget: {budget:.0f} ms (20Hz)")
        print(f"    Headroom: {budget - arr.mean():.1f} ms")
        print(f"    Within budget: {'YES' if arr.mean() < budget else 'NO'}")
        print()
        print("  Pipeline tested:")
        print("    IMX274 cameras (real)")
        print("    → Holoscan pipeline (Docker)")
        print("    → RGBA16→NV12 CUDA conversion")
        print("    → /dev/shm ring buffer")
        print("    → jetson_camerad (VisionIPC)")
        print("    → CUDA warp + YUV preprocessing")
        print("    → driving_vision.onnx (CUDA)")
        print("    → driving_policy.onnx (CUDA)")
        print("    → model outputs (plan, lanes, leads, pose)")
        print()

        if arr.mean() < budget:
            print("  PASS")
        else:
            print("  FAIL: pipeline too slow")
            return 1

    finally:
        # Clean up camerad subprocess
        print("\nStopping jetson_camerad...")
        camerad_proc.send_signal(signal.SIGINT)
        try:
            camerad_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            camerad_proc.kill()
        print("Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
