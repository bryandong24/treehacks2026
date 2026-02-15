"""Thread-safe ring buffer for JPEG frames and ego-motion data.

Stores recent frames from Thor and provides inference snapshots with
4 frames at 100ms intervals as a (4, 3, H, W) tensor plus ego data.
"""

import io
import threading
import time
from collections import deque

import torch
from PIL import Image

from . import config
from .ego_motion import EgoMotionAccumulator


class FrameBuffer:
    """Ring buffer holding JPEG frames and ego-motion snapshots."""

    def __init__(self, max_frames: int = 100):
        self._lock = threading.Lock()
        # Each entry: (timestamp_s, jpeg_bytes)
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=max_frames)
        self.ego = EgoMotionAccumulator()
        self._latest_jpeg: bytes | None = None

    def push(self, frame_jpeg: bytes, ego: dict | None = None):
        """Add a frame and optional ego-motion sample.

        Args:
            frame_jpeg: JPEG-encoded frame bytes.
            ego: Dict with 'timestamp', 'orientation_ned', 'velocity_device'.
        """
        ts = time.time()
        with self._lock:
            self._frames.append((ts, frame_jpeg))
            self._latest_jpeg = frame_jpeg
            if ego is not None:
                self.ego.push(
                    timestamp_s=ego.get("timestamp", ts),
                    orientation_ned=ego["orientation_ned"],
                    velocity_device=ego["velocity_device"],
                )

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_inference_snapshot(self) -> dict | None:
        """Return 4 frames at ~100ms intervals + ego data for inference.

        Returns dict with:
            frames: torch.Tensor (4, 3, H, W) float32 [0-255]
            ego_history_xyz: torch.Tensor (1, 1, 16, 3)
            ego_history_rot: torch.Tensor (1, 1, 16, 3, 3)

        Returns None if not enough data is available.
        """
        with self._lock:
            if len(self._frames) < config.NUM_FRAMES:
                return None

            ego_data = self.ego.get_ego_tensors()
            if ego_data is None:
                return None

            # Pick 4 frames: take the most recent ones spaced ~100ms apart
            all_frames = list(self._frames)
            selected = self._select_frames(all_frames, config.NUM_FRAMES, config.TIME_STEP)

            # Decode JPEGs to tensors
            frame_tensors = []
            for _, jpeg_bytes in selected:
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                arr = torch.tensor(
                    list(img.getdata()), dtype=torch.float32
                ).reshape(img.height, img.width, 3).permute(2, 0, 1)  # (3, H, W)
                frame_tensors.append(arr)

            frames_tensor = torch.stack(frame_tensors)  # (4, 3, H, W)

            return {
                "frames": frames_tensor,
                **ego_data,
            }

    @staticmethod
    def _select_frames(
        frames: list[tuple[float, bytes]], n: int, interval_s: float
    ) -> list[tuple[float, bytes]]:
        """Select n frames from the buffer, spaced approximately interval_s apart.

        Works backwards from the most recent frame.
        """
        if len(frames) <= n:
            return frames[-n:]

        selected = [frames[-1]]
        target_ts = frames[-1][0] - interval_s

        for f in reversed(frames[:-1]):
            if len(selected) >= n:
                break
            if f[0] <= target_ts:
                selected.append(f)
                target_ts -= interval_s

        # If we didn't get enough (frames too sparse), pad from most recent
        while len(selected) < n:
            selected.append(frames[-1])

        selected.reverse()
        return selected
