#!/usr/bin/env python3
"""Jetson camera daemon — reads NV12 frames from /dev/shm and publishes via VisionIPC.

Runs on the HOST (not inside Docker). Reads frames that holoscan_frame_publisher.py
writes to /dev/shm ring buffers and publishes them through VisionIPC so that
modeld and other sunnypilot processes can consume camera data.

Camera mapping:
  /dev/shm/sunnypilot_cam_road → VISION_STREAM_ROAD   (roadCameraState)
  /dev/shm/sunnypilot_cam_wide → VISION_STREAM_WIDE_ROAD (wideRoadCameraState)

Usage:
  LD_LIBRARY_PATH=.venv/lib:/usr/local/cuda/lib64 \
    .venv/bin/python selfdrive/camerad/jetson_camerad.py
"""

import os
import sys
import time

import numpy as np

from cereal import messaging
from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.realtime import Ratekeeper

from openpilot.selfdrive.camerad.shm_buffer import (
    ShmRingBufferReader, SHM_PATH_ROAD, SHM_PATH_WIDE,
)

# Camera resolution (must match holoscan_frame_publisher)
CAM_W = 1920
CAM_H = 1080
NUM_VIPC_BUFFERS = 4


def main():
    print("=" * 60)
    print("  Jetson Camera Daemon (jetson_camerad)")
    print("=" * 60)
    print(f"  Road SHM: {SHM_PATH_ROAD}")
    print(f"  Wide SHM: {SHM_PATH_WIDE}")
    print(f"  Resolution: {CAM_W}x{CAM_H}")
    print(f"  Target rate: 20 Hz")
    print()

    # Connect to shared memory ring buffers
    reader_road = ShmRingBufferReader(SHM_PATH_ROAD)
    reader_wide = ShmRingBufferReader(SHM_PATH_WIDE)

    print("Waiting for road camera shm...")
    if not reader_road.connect(timeout=30.0):
        print(f"ERROR: Timed out waiting for {SHM_PATH_ROAD}")
        print("Is holoscan_frame_publisher.py running in the Docker container?")
        return 1
    print(f"  Road camera connected: {reader_road.width}x{reader_road.height}")

    print("Waiting for wide camera shm...")
    if not reader_wide.connect(timeout=30.0):
        print(f"ERROR: Timed out waiting for {SHM_PATH_WIDE}")
        print("Is holoscan_frame_publisher.py running in the Docker container?")
        return 1
    print(f"  Wide camera connected: {reader_wide.width}x{reader_wide.height}")

    # Set up VisionIPC server (same name as stock camerad)
    vipc_server = VisionIpcServer("camerad")
    vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD,
                               NUM_VIPC_BUFFERS, CAM_W, CAM_H)
    vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD,
                               NUM_VIPC_BUFFERS, CAM_W, CAM_H)
    vipc_server.start_listener()
    print("VisionIPC server started")

    # Set up cereal publishers
    pm = messaging.PubMaster(["roadCameraState", "wideRoadCameraState"])

    # Identity transform (no ISP warp applied)
    identity_transform = [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0]

    frame_id = 0
    rk = Ratekeeper(20, print_delay_threshold=None)

    frames_sent = 0
    t_start = time.monotonic()
    t_last_log = t_start

    print("Publishing frames at 20Hz...\n")

    while True:
        # Read latest frames from shm
        road_data, road_ts, road_fc = reader_road.read_latest()
        wide_data, wide_ts, wide_fc = reader_wide.read_latest()

        ts_ns = int(time.monotonic_ns())

        # Publish road camera
        if road_data is not None:
            vipc_server.send(VisionStreamType.VISION_STREAM_ROAD,
                             road_data, frame_id, road_ts, ts_ns)

            dat = messaging.new_message("roadCameraState", valid=True)
            dat.roadCameraState = {
                "frameId": frame_id,
                "transform": identity_transform,
                "sensor": "unknown",
            }
            pm.send("roadCameraState", dat)

        # Publish wide camera
        if wide_data is not None:
            vipc_server.send(VisionStreamType.VISION_STREAM_WIDE_ROAD,
                             wide_data, frame_id, wide_ts, ts_ns)

            dat = messaging.new_message("wideRoadCameraState", valid=True)
            dat.wideRoadCameraState = {
                "frameId": frame_id,
                "transform": identity_transform,
                "sensor": "unknown",
            }
            pm.send("wideRoadCameraState", dat)

        frame_id += 1
        frames_sent += 1

        # Log periodically
        now = time.monotonic()
        if now - t_last_log >= 5.0:
            elapsed = now - t_start
            fps = frames_sent / elapsed if elapsed > 0 else 0
            print(f"  frame_id={frame_id}  total={frames_sent}  "
                  f"avg_fps={fps:.1f}  "
                  f"road_shm_frames={road_fc}  wide_shm_frames={wide_fc}")
            t_last_log = now

        rk.keep_time()


if __name__ == "__main__":
    sys.exit(main() or 0)
