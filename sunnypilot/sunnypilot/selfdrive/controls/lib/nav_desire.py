"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json
import time

from cereal import log
from openpilot.common.params import Params

Desire = log.Desire

# If NavDesire is older than this, ignore it
STALE_THRESHOLD = 3.0


class NavDesireController:
  def __init__(self, desire_helper):
    self.DH = desire_helper
    self.mem_params = Params("/dev/shm/params")
    self.desire = Desire.none
    self.param_read_counter = 0

  def update_params(self) -> None:
    """Read NavDesire param periodically (every 10 frames = 0.5s at 20Hz)."""
    if self.param_read_counter % 10 == 0:
      self._read_nav_desire()
    self.param_read_counter += 1

  def _read_nav_desire(self):
    """Parse NavDesire JSON from shared memory params."""
    try:
      raw = self.mem_params.get("NavDesire")
      if raw is None:
        self.desire = Desire.none
        return

      data = json.loads(raw)
      desire_val = data.get("desire", 0)
      timestamp = data.get("timestamp", 0)

      # Check staleness
      if time.monotonic() - timestamp > STALE_THRESHOLD:
        self.desire = Desire.none
        return

      # Validate desire value (0-6 maps to Desire enum)
      if 0 <= desire_val <= 6:
        self.desire = desire_val
      else:
        self.desire = Desire.none
    except (json.JSONDecodeError, TypeError, KeyError):
      self.desire = Desire.none

  def get_desire(self) -> int:
    """Return the current navigation desire."""
    return self.desire
