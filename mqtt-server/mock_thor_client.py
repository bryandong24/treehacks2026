#!/usr/bin/env python3
"""Mock Thor client — simulates the AGX Thor sending data to the H100 server.

Generates synthetic JPEG frames and ego-motion data and streams them over
WebSocket at ~10Hz. Use this to test the full pipeline without the real car.

Usage:
    python mock_thor_client.py [--server ws://localhost:8000/ws/thor]
"""

import argparse
import asyncio
import io
import math
import time

import msgpack
from PIL import Image

DEFAULT_SERVER = "ws://localhost:8000/ws/thor"

# Synthetic frame size (road camera resolution)
FRAME_W = 640
FRAME_H = 480


def make_synthetic_jpeg(frame_idx: int) -> bytes:
    """Generate a synthetic JPEG frame with a moving gradient."""
    img = Image.new("RGB", (FRAME_W, FRAME_H))
    pixels = img.load()
    offset = frame_idx * 5
    for y in range(FRAME_H):
        for x in range(FRAME_W):
            r = (x + offset) % 256
            g = (y + offset) % 256
            b = (x + y + offset) % 256
            pixels[x, y] = (r, g, b)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def make_ego_data(t: float) -> dict:
    """Generate mock ego-motion data simulating driving in a gentle curve.

    Produces orientation as quaternion [w, x, y, z] and velocity in device frame.
    """
    # Simulate heading that slowly rotates (gentle left turn)
    yaw = 0.05 * t  # radians, ~3 deg/s
    speed = 10.0     # m/s forward

    # Quaternion for yaw-only rotation (around z-axis)
    # q = [cos(yaw/2), 0, 0, sin(yaw/2)]
    w = math.cos(yaw / 2)
    x = 0.0
    y = 0.0
    z = math.sin(yaw / 2)

    # Velocity in device frame (forward = x)
    vx = speed
    vy = 0.0
    vz = 0.0

    return {
        "timestamp": t,
        "orientation_ned": [w, x, y, z],
        "velocity_device": [vx, vy, vz],
    }


async def run(server_url: str, duration: float = 30.0):
    """Stream mock data to the server for `duration` seconds."""
    import websockets

    print(f"Connecting to {server_url} ...")
    async with websockets.connect(server_url, max_size=4 * 1024 * 1024) as ws:
        print(f"Connected. Streaming for {duration}s at 10Hz...")
        start = time.time()
        frame_idx = 0

        while time.time() - start < duration:
            t = time.time()
            jpeg = make_synthetic_jpeg(frame_idx)
            ego = make_ego_data(t)

            msg = msgpack.packb(
                {"frame_jpeg": jpeg, "ego": ego},
                use_bin_type=True,
            )

            await ws.send(msg)
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = time.time() - start
                print(f"  Sent {frame_idx} frames ({elapsed:.1f}s elapsed)")

            await asyncio.sleep(0.1)  # 10 Hz

        print(f"Done. Sent {frame_idx} frames total.")


def main():
    parser = argparse.ArgumentParser(description="Mock Thor client for testing")
    parser.add_argument(
        "--server", default=DEFAULT_SERVER, help=f"WebSocket URL (default: {DEFAULT_SERVER})"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Duration in seconds (default: 30)"
    )
    args = parser.parse_args()
    asyncio.run(run(args.server, args.duration))


if __name__ == "__main__":
    main()
