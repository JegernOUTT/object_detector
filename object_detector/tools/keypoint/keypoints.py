from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Keypoints:
    points: np.ndarray
    meta: any = None
