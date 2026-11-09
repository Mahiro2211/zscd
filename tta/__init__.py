"""Test-time adaptation for the zero-shot change tracks.

TentCD: TENT-style LayerNorm entropy minimization (binary + anti-collapse).
QuantileCalibrator: streaming quantile calibration control (direction A).
"""
from .tent_cd import QuantileCalibrator, TentCD, collect_params_ln

__all__ = ['TentCD', 'QuantileCalibrator', 'collect_params_ln']
