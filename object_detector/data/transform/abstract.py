from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from object_detector.data.structs import TransformConfig, AnnotationInformation
from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.keypoint.keypoints import Keypoints
from object_detector.tools.mask.mask import Mask
from object_detector.tools.polygon.polygon import Polygon


class AbstractTransformer(ABC):
    def __init__(self, transform_config: TransformConfig, order: int):
        self._config = transform_config
        self._order: int = order

    def get_order(self) -> int:
        return self._order

    def __call__(self, image: np.ndarray, annotations: List[AnnotationInformation]) \
            -> Tuple[np.ndarray, List[AnnotationInformation]]:
        self._create_deterministic_transformer()
        transformed_image = image.copy()
        self._transform_image(transformed_image)

        transformed_annotations = {idx: ann for idx, ann in enumerate(annotations)}
        for idx, ann in transformed_annotations.items():
            ann.annotation.meta = idx

        bboxes = [ann.annotation for ann in annotations if type(ann.annotation) == Bbox]
        bboxes_with_masks = [bb for bb in bboxes if bb.mask is not None]
        bboxes_with_keypoints = [bb for bb in bboxes if bb.keypoints is not None]
        masks = [ann.annotation for ann in annotations if type(ann.annotation) == Mask]
        keypoints = [ann.annotation for ann in annotations if type(ann.annotation) == Keypoints]
        polygons = [ann.annotation for ann in annotations if type(ann.annotation) == Polygon]

        bboxes = self._transform_bboxes(bboxes)
        for ann in bboxes:
            transformed_annotations[ann.meta].annotation = ann

        bboxes_with_masks = self._transform_bboxes_masks(bboxes_with_masks)
        for ann in bboxes_with_masks:
            transformed_annotations[ann.meta].annotation = ann

        bboxes_with_keypoints = self._transform_bboxes_keypoints(bboxes_with_keypoints)
        for ann in bboxes_with_keypoints:
            transformed_annotations[ann.meta].annotation = ann

        masks = self._transform_masks(masks)
        for ann in masks:
            transformed_annotations[ann.meta].annotation = ann

        keypoints = self._transform_keypoints(keypoints)
        for ann in keypoints:
            transformed_annotations[ann.meta].annotation = ann

        polygons = self._transform_polygons(polygons)
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
    def _transform_bboxes(self, bboxes: List[Bbox]) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_bboxes_masks(self, bboxes: List[Bbox]) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_bboxes_keypoints(self, bboxes: List[Bbox]) -> List[Bbox]:
        pass

    @abstractmethod
    def _transform_masks(self, masks: List[Mask]) -> List[Mask]:
        pass

    @abstractmethod
    def _transform_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        pass

    @abstractmethod
    def _transform_keypoints(self, keypoints: List[Keypoints]) -> List[Keypoints]:
        pass
