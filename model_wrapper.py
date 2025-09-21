import ctypes
import numpy as np
import os

LIB_PATH = os.path.join(os.path.dirname(__file__), "build/lib/libedge_impulse_model.so")
_lib = ctypes.cdll.LoadLibrary(LIB_PATH)

# function signatures
_lib.ei_model_init.restype = ctypes.c_int
_lib.ei_model_cleanup.restype = None
_lib.ei_model_is_initialized.restype = ctypes.c_int
_lib.ei_model_get_input_shape.argtypes = [ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
_lib.ei_model_get_input_shape.restype = ctypes.c_int
_lib.ei_model_get_output_dims.argtypes = [ctypes.POINTER(ctypes.c_int)]
_lib.ei_model_get_output_dims.restype = ctypes.c_int
_lib.ei_model_infer.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                ctypes.POINTER(ctypes.c_float)]
_lib.ei_model_infer.restype = ctypes.c_int

class EdgeImpulseModel:
    def __init__(self):
        if _lib.ei_model_init() != 0:
            raise RuntimeError("Failed to init EI model")

        w = h = c = ctypes.c_int()
        _lib.ei_model_get_input_shape(ctypes.byref(w), ctypes.byref(h), ctypes.byref(c))
        self.input_shape = (h.value, w.value, c.value)

        n = ctypes.c_int()
        _lib.ei_model_get_output_dims(ctypes.byref(n))
        self.num_classes = n.value

    def infer(self, img: np.ndarray) -> np.ndarray:
        """img: 96x96 uint8 grayscale"""
        if img.shape != (96, 96):
            raise ValueError("Expected 96x96 grayscale")
        scores = (ctypes.c_float * self.num_classes)()
        arr = img.flatten().astype(np.uint8)
        ret = _lib.ei_model_infer(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), scores)
        if ret != 0:
            raise RuntimeError("Inference failed")
        return np.array(scores, dtype=np.float32)

    def close(self):
        _lib.ei_model_cleanup()
