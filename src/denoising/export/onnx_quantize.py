"""Static int8 quantization of an ONNX model using noisy training crops for calibration."""

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class _CropReader(CalibrationDataReader):
    def __init__(self, arrays: list[np.ndarray]):
        self._batches = iter([{"noisy": a} for a in arrays])

    def get_next(self):
        return next(self._batches, None)


def quantize_int8(fp32_path: str, int8_path: str, calib_arrays: list[np.ndarray]) -> None:
    quantize_static(
        fp32_path,
        int8_path,
        _CropReader(calib_arrays),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
