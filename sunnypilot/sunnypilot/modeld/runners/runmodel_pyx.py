"""Stub for runmodel_pyx Cython module.

On Jetson Thor we use ONNX Runtime directly instead of the tinygrad/C++ runner.
This stub provides the RunModel and Runtime classes so the import chain doesn't break.
"""


class Runtime:
    CPU = 0
    GPU = 1
    DSP = 2


class RunModel:
    """Stub base class — not used on Jetson Thor (ONNX RT is used directly)."""
    THNEED = 'THNEED'
    ONNX = 'ONNX'

    def addInput(self, name, buffer):
        raise NotImplementedError("RunModel stub — use ONNX Runtime directly on Jetson")

    def setInputBuffer(self, name, buffer):
        raise NotImplementedError("RunModel stub — use ONNX Runtime directly on Jetson")

    def getCLBuffer(self, name):
        return None

    def execute(self):
        raise NotImplementedError("RunModel stub — use ONNX Runtime directly on Jetson")
