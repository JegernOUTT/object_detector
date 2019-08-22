from typing import Union, Tuple, List

import numpy as np
from dataclasses import dataclass

from object_detector.tools.keypoint.keypoint import Keypoint
from object_detector.tools.mask.mask import Mask
from object_detector.tools.structs import Size2D

BboxIterableType = Union[Tuple[float, float, float, float], List[float], np.ndarray]


@dataclass(frozen=True)
class Bbox:
    xyxy: np.ndarray
    mask: Union[None, Mask] = None
    keypoints: Union[None, List[Keypoint]] = None

    def _clip_coords(self):
        self.xyxy[0] = max(min(self.xyxy[0], 1.), 0.)
        self.xyxy[1] = max(min(self.xyxy[1], 1.), 0.)
        self.xyxy[2] = max(min(self.xyxy[2], 1.), 0.)
        self.xyxy[3] = max(min(self.xyxy[3], 1.), 0.)

    def width(self, image_size: Union[Size2D, None] = None):
        return self.xyxy[2] - self.xyxy[0] if image_size is None else \
            (self.xyxy[2] - self.xyxy[0]) * image_size.width

    def height(self, image_size: Union[Size2D, None] = None):
        return self.xyxy[3] - self.xyxy[1] if image_size is None else \
            (self.xyxy[3] - self.xyxy[1]) * image_size.height

    def area(self, image_size: Union[Size2D, None] = None):
        return self.width(image_size) * self.height(image_size)

    def get_xyxy(self, image_size: Union[Size2D, None] = None) -> np.ndarray:
        if image_size:
            xyxy = self.xyxy * [image_size.width, image_size.height]
        else:
            xyxy = self.xyxy.copy()

        return xyxy

    def get_xywh(self, image_size: Union[Size2D, None] = None) -> np.ndarray:
        if image_size:
            xyxy = self.xyxy * [image_size.width, image_size.height]
        else:
            xyxy = self.xyxy.copy()

        return np.array([*xyxy[:2], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]], dtype=np.float32)

    def get_center_xywh(self, image_size: Union[Size2D, None] = None) -> np.ndarray:
        if image_size:
            xyxy = self.xyxy * [image_size.width, image_size.height]
        else:
            xyxy = self.xyxy.copy()

        return np.array([(xyxy[2] + xyxy[0]) / 2, (xyxy[3] + xyxy[1]) / 2, xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        dtype=np.float32)


def from_xyxy(xyxy: BboxIterableType,
              image_size: Union[Size2D, None] = None,
              *args, **kwargs):
    if image_size is not None:
        xyxy = xyxy[0] / image_size.width, xyxy[1] / image_size.height, \
               xyxy[2] / image_size.width, xyxy[3] / image_size.height

    return Bbox(xyxy=np.array(xyxy, dtype=np.float32),
                *args, **kwargs)


def from_xywh(xywh: BboxIterableType,
              image_size: Union[Size2D, None] = None,
              *args, **kwargs):
    xyxy = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
    if image_size is not None:
        xyxy = xyxy[0] / image_size.width, xyxy[1] / image_size.height, \
               xyxy[2] / image_size.width, xyxy[3] / image_size.height

    return Bbox(xyxy=np.array(xyxy, dtype=np.float32),
                *args, **kwargs)


def from_center_xywh(center_xywh: BboxIterableType,
                     image_size: Union[Size2D, None] = None,
                     *args, **kwargs):
    x1, y1 = center_xywh[0] - (center_xywh[2] / 2), center_xywh[1] - (center_xywh[3] / 2)
    x2, y2 = center_xywh[0] + (center_xywh[2] / 2), center_xywh[1] + (center_xywh[3] / 2)
    xyxy = [x1, y1, x2, y2]
    if image_size is not None:
        xyxy = xyxy[0] / image_size.width, xyxy[1] / image_size.height, \
               xyxy[2] / image_size.width, xyxy[3] / image_size.height

    return Bbox(xyxy=np.array(xyxy, dtype=np.float32),
                *args, **kwargs)
