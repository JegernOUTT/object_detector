from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from object_detector.data.structs import TransformConfig, AnnotationInformation
from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.keypoint.keypoints import Keypoints
from object_detector.tools.mask.mask import Mask
from object_detector.tools.polygon.polygon import Polygon
from object_detector.tools.structs import Size2D


class AbstractTransformer(ABC):
    def __init__(self, transform_config: TransformConfig, order: int):
        self._config = transform_config
        self._order: int = order

    def get_order(self) -> int:
        return self._order

    def __call__(self, image: np.ndarray, annotations: List[AnnotationInformation]) \
            -> Tuple[np.ndarray, List[AnnotationInformation]]:
        self._create_deterministic_transformer()
        before_image_size = Size2D(width=image.shape[1], height=image.shape[0])
        transformed_image = self._transform_image(image)
        after_image_size = Size2D(width=transformed_image.shape[1], height=transformed_image.shape[0])

        transformed_annotations = {idx: ann for idx, ann in enumerate(annotations)}
        for idx, ann in transformed_annotations.items():
            ann.annotation.meta = idx

        bboxes = [ann.annotation for ann in annotations if type(ann.annotation) == Bbox]
        bboxes_with_masks = [bb for bb in bboxes if bb.mask is not None]
        bboxes_with_keypoints = [bb for bb in bboxes if bb.keypoints is not None]
        masks = [ann.annotation for ann in annotations if type(ann.annotation) == Mask]
        keypoints = [ann.annotation for ann in annotations if type(ann.annotation) == Keypoints]
        polygons = [ann.annotation for ann in annotations if type(ann.annotation) == Polygon]

        bboxes = self._transform_bboxes(bboxes, before_image_size, after_image_size)
        for ann in bboxes:
            transformed_annotations[ann.meta].annotation = ann

        if len(bboxes_with_masks) > 0:
            bboxes_with_masks = self._transform_bboxes_masks(
                bboxes_with_masks, before_image_size, after_image_size)
            for ann in bboxes_with_masks:
                transformed_annotations[ann.meta].annotation = ann

        if len(bboxes_with_keypoints) > 0:
            bboxes_with_keypoints = self._transform_bboxes_keypoints(
                bboxes_with_keypoints, before_image_size, after_image_size)
            for ann in bboxes_with_keypoints:
                transformed_annotations[ann.meta].annotation = ann

        if len(masks) > 0:
            masks = self._transform_masks(masks, before_image_size, after_image_size)
            for ann in masks:
                transformed_annotations[ann.meta].annotation = ann

        if len(keypoints) > 0:
            keypoints = self._transform_keypoints(keypoints, before_image_size, after_image_size)
            for ann in keypoints:
                transformed_annotations[ann.meta].annotation = ann

        if len(polygons) > 0:
            polygons = self._transform_polygons(polygons, before_image_size, after_image_size)
            for ann in polygons:
                transformed_annotations[ann.meta].annotation = ann

        return transformed_image, list(transformed_annotations.values())

    @abstractmethod
    def _create_deterministic_transformer(self):
        pass

    @abstractmethod
    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _transform_bboxes(self, bboxes: List[Bbox],
                          before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_bboxes_masks(self, bboxes: List[Bbox],
                                before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_bboxes_keypoints(self, bboxes: List[Bbox],
                                    before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_masks(self, masks: List[Mask],
                         before_img_size: Size2D, after_img_size: Size2D) -> List[Mask]:
        pass

    @abstractmethod
    def _transform_polygons(self, polygons: List[Polygon],
                            before_img_size: Size2D, after_img_size: Size2D) -> List[Polygon]:
        pass

    @abstractmethod
    def _transform_keypoints(self, keypoints: List[Keypoints],
                             before_img_size: Size2D, after_img_size: Size2D) -> List[Keypoints]:
        pass
