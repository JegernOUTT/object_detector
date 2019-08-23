from dataclasses import dataclass
from typing import List, Type

import numpy as np

from object_detector.data.model_heads_formatters.abstract import AbstractHeadFormatter
from object_detector.data.structs import AnnotationInformation, HeadFormatterConfig
from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.structs import Size2D


@dataclass
class CenterNetHeadFormatterConfig(HeadFormatterConfig):
    xy_tensor_dict_name: str = 'xy'
    xy_offset_tensor_dict_name: str = 'xy_offset'
    wh_tensor_dict_name: str = 'wh'
    ignore_masks_dict_name: str = 'ignore_masks'
    output_stride: int = 4
    wh_log_scale: bool = True

    def owner_type(self) -> Type['CenterNetDetectionHeads']:
        return CenterNetDetectionHeads


class CenterNetDetectionHeads(AbstractHeadFormatter):
    def __init__(self, head_formatter_config: CenterNetHeadFormatterConfig):
        super().__init__(head_formatter_config)

    @staticmethod
    def _get_gaussian_2D(shape, sigma=1.):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def _calc_gaussian_radius(bbox: Bbox, image_size: Size2D, min_overlap: float = 0.7):
        width, height = bbox.width(image_size), bbox.height(image_size)

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        return min(r1, r2, r3)

    def __call__(self, image: np.ndarray, annotations: List[AnnotationInformation]) -> dict:
        img_size = Size2D(width=image.shape[1], height=image.shape[0])
        output_size_w, output_size_h = img_size.width // self._config.output_stride, \
                                       img_size.height // self._config.output_stride

        bboxes = [ann for ann in annotations if type(ann.annotation) == Bbox and not ann.ignore]
        ignore_annotations = [ann for ann in annotations if ann.ignore]

        ignore_masks_b = np.zeros((output_size_h, output_size_w, self._config.categories_count), dtype=np.float32)
        xy = np.zeros((output_size_h, output_size_w, self._config.categories_count), dtype=np.float32)
        xy_offset = np.zeros((output_size_h, output_size_w, 2), dtype=np.float32)
        wh = np.zeros((output_size_h, output_size_w, 2), dtype=np.float32)

        for ann in bboxes:
            bbox: Bbox = ann.annotation
            if bbox.width() == 0 or bbox.height() == 0:
                continue
            x, y, w, h = bbox.get_center_xywh(image_size=img_size) / self._config.output_stride

            x_idx, y_idx = map(int, [x, y])
            x_offset, y_offset = x - x_idx, y - y_idx

            radius = self._calc_gaussian_radius(bbox=bbox, image_size=img_size)
            radius = max(0, int(radius))

            diameter = 2 * radius + 1
            gaussian = self._get_gaussian_2D((diameter, diameter), sigma=diameter / 6)

            left, right = min(x_idx, radius), min(output_size_w - x_idx, radius + 1)
            top, bottom = min(y_idx, radius), min(output_size_h - y_idx, radius + 1)

            masked_xy = xy[y_idx - top:y_idx + bottom, x_idx - left:x_idx + right, ann.category_id]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            masked_xy = np.maximum(masked_xy, masked_gaussian)

            xy[y_idx - top:y_idx + bottom, x_idx - left:x_idx + right, ann.category_id] = masked_xy
            xy_offset[y_idx, x_idx] = [x_offset, y_offset]
            if self._config.wh_log_scale:
                wh[y_idx, x_idx] = [np.log(w + np.finfo(np.float32).eps), np.log(h + np.finfo(np.float32).eps)]
            else:
                wh[y_idx, x_idx] = [w, h]

        ignore_masks = render_annotations_on_bitmap(ignore_annotations,
                                                    categories_count=self._config.categories_count,
                                                    bitmap_size=Size2D(width=output_size_w, height=output_size_h))
        ignore_masks *= np.where(xy > 0., 0., ignore_masks_b)

        return {
            self._config.xy_tensor_dict_name: xy,
            self._config.xy_offset_tensor_dict_name: xy_offset,
            self._config.wh_tensor_dict_name: wh,
            self._config.ignore_masks_dict_name: ignore_masks,
            'bboxes': bboxes,
            'ignore_annotations': ignore_annotations
        }
