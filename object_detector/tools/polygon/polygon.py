from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Polygon:
    points: np.ndarray
    meta: any = None
