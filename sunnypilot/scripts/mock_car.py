#!/usr/bin/env python3
"""
Mock panda + car daemon for E2E testing without physical panda/vehicle.

Publishes:
  - pandaStates (10Hz) — ignition=True, controlsAllowed=True
  - peripheralState (2Hz) — voltage, fan speed
  - can (100Hz) — empty CAN frames
  - carState (100Hz) — speed from GPS, gear=drive, cruise enabled
  - carOutput (100Hz) — empty actuator feedback
  - carParams (once) — Honda Civic Bosch config written to Params

Also reads sendcan and discards (no real CAN bus).
"""
import json
import os
import sys
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cereal.messaging as messaging
from cereal import car, log, custom
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog

PandaType = log.PandaState.PandaType
SafetyModel = car.CarParams.SafetyModel


def create_honda_bosch_carparams() -> bytes:
  """Create minimal CarParams for Honda Civic Bosch and return serialized bytes."""
  from opendbc.car import structs

  cp = structs.CarParams()
  cp.brand = "honda"
  cp.carFingerprint = "HONDA_CIVIC"

  # Safety
  safety_cfg = structs.CarParams.SafetyConfig()
  safety_cfg.safetyModel = structs.CarParams.SafetyModel.hondaBosch
  safety_cfg.safetyParam = 2  # BOSCH_LONG
  cp.safetyConfigs = [safety_cfg]

  # Steering
  cp.steerControlType = structs.CarParams.SteerControlType.torque
  cp.steerRatio = 15.38
  cp.steerActuatorDelay = 0.1
  cp.minSteerSpeed = 0.89  # ~2 mph

  # Longitudinal
  cp.openpilotLongitudinalControl = True
  cp.longitudinalActuatorDelay = 0.5
  cp.radarUnavailable = True
  cp.pcmCruise = False

  # Physical
  cp.mass = 1326.0
  cp.wheelbase = 2.7
  cp.centerToFront = 1.08
  cp.tireStiffnessFront = 80000.0
  cp.tireStiffnessRear = 80000.0

  # Speed limits
  cp.minEnableSpeed = -1.0
  cp.vEgoStopping = 0.1
  cp.vEgoStarting = 0.1

  # Lateral tuning — PID
  cp.lateralTuning.pid.kpBP = [0.0]
  cp.lateralTuning.pid.kpV = [0.8]
  cp.lateralTuning.pid.kiBP = [0.0]
  cp.lateralTuning.pid.kiV = [0.24]
  cp.lateralTuning.pid.kf = 0.00006
  cp.lateralParams.torqueBP = [0, 4096]
  cp.lateralParams.torqueV = [0, 4096]

  # Transmission
  cp.transmissionType = structs.CarParams.TransmissionType.automatic
  cp.wheelSpeedFactor = 1.0

  return cp.to_bytes()


def publish_panda_states(pm: messaging.PubMaster, rk: Ratekeeper):
  """Publish pandaStates at 10Hz with ignition on."""
  while True:
    msg = messaging.new_message('pandaStates', 1, valid=True)
    ps = msg.pandaStates[0]
    ps.pandaType = PandaType.tres
    ps.ignitionLine = True
    ps.ignitionCan = False
    ps.controlsAllowed = True
    ps.safetyModel = SafetyModel.hondaBosch
    ps.safetyParam = 2  # BOSCH_LONG
    ps.harnessStatus = log.PandaState.HarnessStatus.normal
    ps.heartbeatLost = False
    ps.faultStatus = log.PandaState.FaultStatus.none
    pm.send('pandaStates', msg)
    rk.keep_time()


def publish_peripheral_state(pm: messaging.PubMaster, rk: Ratekeeper):
  """Publish peripheralState at 2Hz."""
  while True:
    msg = messaging.new_message('peripheralState', valid=True)
    ps = msg.peripheralState
    ps.pandaType = PandaType.tres
    ps.voltage = 12000  # 12V
    ps.current = 500
    ps.fanSpeedRpm = 0
    pm.send('peripheralState', msg)
    rk.keep_time()


def publish_can(pm: messaging.PubMaster, rk: Ratekeeper):
  """Publish empty CAN at 100Hz."""
  while True:
    msg = messaging.new_message('can', 0)
    pm.send('can', msg)
    rk.keep_time()


def publish_car_state(pm: messaging.PubMaster, params: Params, rk: Ratekeeper):
  """Publish carState at 100Hz with GPS-derived speed."""
  mem_params = Params("/dev/shm/params")
  speed = 0.0

  while True:
    # Try to get speed from GPS
    try:
      raw = mem_params.get("LastGPSPosition")
      if raw:
        data = json.loads(raw)
        # GPS speed not in LastGPSPosition, use 0 for now
        # Real speed will come from the GPS daemon via gpsLocationExternal
    except Exception:
      pass

    msg = messaging.new_message('carState', valid=True)
    cs = msg.carState
    cs.vEgo = speed
    cs.vEgoRaw = speed
    cs.vEgoCluster = speed
    cs.aEgo = 0.0
    cs.standstill = speed < 0.1
    cs.canValid = True
    cs.canTimeout = False
    cs.canErrorCounter = 0
    cs.steeringAngleDeg = 0.0
    cs.steeringRateDeg = 0.0
    cs.steeringTorque = 0.0
    cs.steeringPressed = False
    cs.steerFaultPermanent = False
    cs.steerFaultTemporary = False
    cs.gasPressed = False
    cs.brake = 0.0
    cs.brakePressed = False
    cs.gearShifter = car.CarState.GearShifter.drive
    cs.cruiseState.enabled = True
    cs.cruiseState.available = True
    cs.cruiseState.speed = 11.176  # 25 mph in m/s
    cs.cruiseState.standstill = False
    cs.seatbeltUnlatched = False
    cs.doorOpen = False
    cs.parkingBrake = False
    cs.leftBlinker = False
    cs.rightBlinker = False
    cs.wheelSpeeds.fl = speed
    cs.wheelSpeeds.fr = speed
    cs.wheelSpeeds.rl = speed
    cs.wheelSpeeds.rr = speed
    cs.yawRate = 0.0
    pm.send('carState', msg)
    rk.keep_time()


def publish_car_output(pm: messaging.PubMaster, rk: Ratekeeper):
  """Publish carOutput at 100Hz."""
  while True:
    msg = messaging.new_message('carOutput', valid=True)
    pm.send('carOutput', msg)
    rk.keep_time()


def drain_sendcan():
  """Read and discard sendcan messages."""
  sm = messaging.SubMaster(['sendcan'])
  while True:
    sm.update(100)


def main():
  params = Params()

  # Write CarParams + CarParamsSP to params store so controlsd/selfdrived can read them
  cloudlog.info("mock_car: writing Honda Civic Bosch CarParams + CarParamsSP")
  cp_bytes = create_honda_bosch_carparams()
  params.put("CarParams", cp_bytes)
  params.put("CarParamsCache", cp_bytes)

  cp_sp = custom.CarParamsSP.new_message()
  params.put("CarParamsSP", cp_sp.to_bytes())
  cloudlog.info("mock_car: CarParams + CarParamsSP written")

  pm = messaging.PubMaster([
    'pandaStates', 'peripheralState', 'can',
    'carState', 'carOutput',
  ])

  # Launch publisher threads
  threads = [
    threading.Thread(target=publish_panda_states, args=(pm, Ratekeeper(10)), daemon=True),
    threading.Thread(target=publish_peripheral_state, args=(pm, Ratekeeper(2)), daemon=True),
    threading.Thread(target=publish_can, args=(pm, Ratekeeper(100)), daemon=True),
    threading.Thread(target=publish_car_state, args=(pm, params, Ratekeeper(100)), daemon=True),
    threading.Thread(target=publish_car_output, args=(pm, Ratekeeper(100)), daemon=True),
    threading.Thread(target=drain_sendcan, daemon=True),
  ]

  for t in threads:
    t.start()

  cloudlog.info("mock_car: all publishers running")

  # Keep main thread alive
  while True:
    time.sleep(1)


if __name__ == "__main__":
  main()
