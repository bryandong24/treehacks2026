#!/usr/bin/env python3
"""
Mock driver monitoring daemon for Jetson — no driver-facing camera.

Publishes driverMonitoringState at 20Hz with "driver is attentive" so
selfdrived never triggers distraction/unresponsive disengagement events.

This replaces dmonitoringd + dmonitoringmodeld on platforms without a
driver monitoring camera.
"""
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, config_realtime_process


def main():
  config_realtime_process(5, 53)

  pm = messaging.PubMaster(['driverMonitoringState'])
  rk = Ratekeeper(20, print_delay_threshold=None)  # 20Hz like stock dmonitoringd

  while True:
    msg = messaging.new_message('driverMonitoringState', valid=True)
    dms = msg.driverMonitoringState
    dms.faceDetected = True
    dms.isDistracted = False
    dms.awarenessStatus = 1.0
    dms.distractedType = 0
    dms.isActiveMode = True
    dms.isRHD = False
    dms.isLowStd = True
    dms.hiStdCount = 0
    dms.posePitchValidCount = 100
    dms.poseYawValidCount = 100
    # dms.events stays empty — no distraction events
    pm.send('driverMonitoringState', msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
