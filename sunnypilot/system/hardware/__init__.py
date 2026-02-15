import os
from typing import cast

from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
JETSON = os.path.isfile('/JETSON')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI and not JETSON


class Jetson(Pc):
  def get_device_type(self):
    return "pc"  # cereal schema only has: unknown, neo, tici, pc, tizi, mici

if TICI:
  HARDWARE = cast(HardwareBase, Tici())
elif JETSON:
  HARDWARE = cast(HardwareBase, Jetson())
else:
  HARDWARE = cast(HardwareBase, Pc())
