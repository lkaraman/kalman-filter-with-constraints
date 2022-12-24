from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MeasuredData:
    s: np.ndarray
    d: np.ndarray
    t: np.ndarray

@dataclass
class KalmanOutput:
    P_minus_arr: np.ndarray
    P_plus_arr: np.ndarray
    x_minus_arr: np.ndarray
    x_plus_arr: np.ndarray
    K_arr: np.ndarray

    A: np.ndarray


@dataclass
class RtsOutput:
    x_smooth: np.ndarray
    K_smooth: np.ndarray
    P_smooth: np.ndarray

@dataclass
class Vehicle:
    id: int
    s: List[float]
    d: List[float]
    t: List[float]
    frames: List[int]
