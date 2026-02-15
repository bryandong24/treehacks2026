"""Convert livePose data into Alpamayo ego-motion format.

Buffers (timestamp, orientation_ned, velocity_device) at 10Hz and produces
ego_history_xyz (1,1,16,3) and ego_history_rot (1,1,16,3,3) tensors in the
local frame centered at t0.

Reference: alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py lines 128-154
"""

from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import config


class EgoMotionAccumulator:
    """Accumulates livePose snapshots and produces ego-motion tensors."""

    def __init__(self, max_len: int = 200):
        # Each entry: (timestamp_s, orientation_quat_xyzw, velocity_device)
        self._buffer: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=max_len)

    def push(self, timestamp_s: float, orientation_ned: list[float], velocity_device: list[float]):
        """Add a livePose sample.

        Args:
            timestamp_s: Unix timestamp in seconds.
            orientation_ned: Quaternion [w, x, y, z] from livePose (NED frame).
            velocity_device: Velocity in device frame [vx, vy, vz] m/s.
        """
        # livePose gives quaternion as [w, x, y, z]; scipy expects [x, y, z, w]
        w, x, y, z = orientation_ned
        quat_xyzw = np.array([x, y, z, w], dtype=np.float64)
        vel = np.array(velocity_device, dtype=np.float64)
        self._buffer.append((timestamp_s, quat_xyzw, vel))

    @property
    def count(self) -> int:
        return len(self._buffer)

    def get_ego_tensors(self) -> dict[str, torch.Tensor] | None:
        """Produce ego_history_xyz and ego_history_rot in local frame.

        Returns None if fewer than NUM_HISTORY_STEPS samples are buffered.
        """
        n = config.NUM_HISTORY_STEPS
        if len(self._buffer) < n:
            return None

        # Take the most recent n samples
        samples = list(self._buffer)[-n:]

        # Extract arrays
        timestamps = np.array([s[0] for s in samples])
        quats = np.array([s[1] for s in samples])       # (n, 4) xyzw
        velocities = np.array([s[2] for s in samples])   # (n, 3) device frame

        # Convert quaternions to Rotation objects
        rotations = Rotation.from_quat(quats)  # scipy uses xyzw
        rot_matrices = rotations.as_matrix()    # (n, 3, 3)

        # Integrate velocity in world frame to reconstruct positions
        # velocity_world = R @ velocity_device
        positions = np.zeros((n, 3), dtype=np.float64)
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i - 1]
            vel_world = rot_matrices[i - 1] @ velocities[i - 1]
            positions[i] = positions[i - 1] + vel_world * dt

        # Transform to local frame at t0 (last sample)
        t0_pos = positions[-1].copy()
        t0_rot = rotations[-1]
        t0_rot_inv = t0_rot.inv()

        # xyz_local = R_t0_inv @ (xyz - xyz_t0)
        ego_history_xyz = t0_rot_inv.apply(positions - t0_pos)

        # rot_local = R_t0_inv * R_world
        ego_history_rot = (t0_rot_inv * rotations).as_matrix()

        # Convert to tensors with shape (1, 1, n, 3) and (1, 1, n, 3, 3)
        xyz_tensor = torch.tensor(ego_history_xyz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        rot_tensor = torch.tensor(ego_history_rot, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return {
            "ego_history_xyz": xyz_tensor,
            "ego_history_rot": rot_tensor,
        }
