from typing import Tuple

import numpy as np
from dataclasses import dataclass
from pycocotools import mask as mask_util
from skimage.transform import resize

from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.structs import Size2D


@dataclass(frozen=True)
class Mask:
    mask: np.ndarray
    meta: any = None


class CocoMaskWrapper:
    def __init__(self, rle: np.ndarray, image_size: Size2D, bbox_info: Tuple[bool, Bbox] = (False, None)):
        self._rle = rle
        self._image_size = image_size
        self._from_bbox, self._bbox = bbox_info

    def numpy(self):
        if self._from_bbox:
            x1, y1, x2, y2 = self._bbox.get_xyxy(image_size=self._image_size)
            mask = np.zeros((self._image_size.height, self._image_size.width), dtype=np.uint8)
            bbox_mask = resize(mask_util.decode(self._rle), (y2 - y1, x2 - x1))
            mask[y1:y2, x1:x2] = (bbox_mask * 255.).astype(np.uint8)
        else:
            mask = mask_util.decode(self._rle)

        return mask

    @staticmethod
    def from_polygon(polygon: dict, image_size: Size2D):
        if type(polygon) == list:
            rles = mask_util.frPyObjects(polygon, image_size.height, image_size.width)
            rle = mask_util.merge(rles)
        elif type(polygon['counts']) == list:
            rle = mask_util.frPyObjects(polygon, image_size.height, image_size.width)
        else:
            # rle
            rle = polygon

        return CocoMaskWrapper(rle=rle, image_size=image_size)


class CocoKeypointsWrapper:
    def __init__(self, keypoints: np.ndarray, image_size: Size2D, bbox: Bbox):
        self.x1, self.y1, self.x2, self.y2 = bbox.get_xyxy(image_size)
        self._image_size = image_size
        keypoints = np.array(keypoints, dtype=np.int32).reshape((-1, 3))
        self._coco_keypoints = keypoints

    def __len__(self):
        return len(self._coco_keypoints)

    def numpy(self, mask_num):
        keypoints = self._coco_keypoints.copy()

        assert mask_num < len(self._coco_keypoints)

        keypoints_mask = np.zeros((self._image_size.height, self._image_size.width), dtype=np.float32)
        kp_w, kp_h, vis = keypoints[mask_num]

        if vis in (1, 2):
            keypoints_mask[kp_h, kp_w] = 1.
        else:
            return None

        return keypoints_mask
