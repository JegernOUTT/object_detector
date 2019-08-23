from dataclasses import dataclass
from typing import List, Union, Type

import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from object_detector.data.structs import TransformConfig
from object_detector.data.transform.abstract import AbstractTransformer
from object_detector.tools.bbox.bbox import Bbox, from_xyxy
from object_detector.tools.keypoint.keypoints import Keypoints
from object_detector.tools.mask.mask import Mask
from object_detector.tools.polygon.polygon import Polygon
from object_detector.tools.structs import Size2D

__all__ = ['ImgaugTransformer']


@dataclass
class ImgaugTransformerConfig(TransformConfig):
    aug_pipeline: iaa.Augmenter

    def owner_type(self) -> Type['ImgaugTransformer']:
        return ImgaugTransformer


class ImgaugTransformer(AbstractTransformer):
    def __init__(self,
                 transform_config: ImgaugTransformerConfig,
                 order: int):
        super().__init__(transform_config, order)
        self._aug_pipeline: iaa.Augmenter = self._config.aug_pipeline
        self._deterministic_pipeline: Union[None, iaa.Augmenter] = None

    def _set_aug_pipeline(self, aug_pipeline: iaa.Augmenter):
        self._aug_pipeline: iaa.Augmenter = aug_pipeline

    def _create_deterministic_transformer(self):
        self._deterministic_pipeline = self._aug_pipeline.to_deterministic()

    def _transform_image(self, image: np.ndarray) -> np.ndarray:
        assert self._deterministic_pipeline is not None
        return self._deterministic_pipeline.augment_image(image)

    def _transform_bboxes(self, bboxes: List[Bbox],
                          before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        assert self._deterministic_pipeline is not None
        imgaug_bboxes = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=bbox)
            for bbox, (x1, y1, x2, y2) in map(lambda x: (x, x.get_xyxy(before_img_size)), bboxes)],
            shape=(before_img_size.height, before_img_size.width))
        imgaug_bboxes = self._deterministic_pipeline.augment_bounding_boxes([imgaug_bboxes])[0]
        imgaug_bboxes = imgaug_bboxes.remove_out_of_image(fully=True, partly=False).clip_out_of_image()

        return [
            from_xyxy(
                xyxy=(aug_bbox.x1, aug_bbox.y1, aug_bbox.x2, aug_bbox.y2),
                image_size=after_img_size,
                mask=aug_bbox.label.mask,
                keypoints=aug_bbox.label.keypoints,
                meta=aug_bbox.label.meta
            )
            for aug_bbox in imgaug_bboxes.bounding_boxes
        ]

    def _transform_bboxes_masks(self, bboxes: List[Bbox],
                                before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        assert self._deterministic_pipeline is not None
        raise NotImplementedError()
        # for idx, masks in enumerate(annotations.get_bboxes_masks()):
        #     assert type(masks) is np.ndarray
        #     bbox = annotations.get_bbox_by_idx(idx=idx)
        #     _, _, bbox_w, bbox_h = bbox.get_xywh_image_coords(image_size=new_image_size)
        #     imgaug_heatmaps = ia.HeatmapsOnImage(masks, shape=(bbox_h, bbox_w))
        #     imgaug_heatmaps = deterministic_aug_pipeline.augment_heatmaps([imgaug_heatmaps])[0]
        #     annotations.set_bbox_mask(bbox_idx=idx, masks=imgaug_heatmaps.arr_0to1)

    def _transform_bboxes_keypoints(self, bboxes: List[Bbox],
                                    before_img_size: Size2D, after_img_size: Size2D) -> List[Bbox]:
        assert self._deterministic_pipeline is not None
        raise NotImplementedError()
        # for idx, keypoints in enumerate(annotations.get_bboxes_keypoints()):
        #     assert type(keypoints) is np.ndarray
        #     bbox = annotations.get_bbox_by_idx(idx=idx)
        #     _, _, bbox_w, bbox_h = bbox.get_xywh_image_coords(image_size=new_image_size)
        #     imgaug_keypoints = ia.KeypointsOnImage([ia.Keypoint(*k) for k in keypoints], shape=(bbox_h, bbox_w))
        #     imgaug_keypoints = deterministic_aug_pipeline.augment_keypoints([imgaug_keypoints])[0]
        #     annotations.set_bbox_keypoints(bbox_idx=idx, keypoints=imgaug_keypoints.arr_0to1)

    def _transform_masks(self, masks: List[Mask],
                         before_img_size: Size2D, after_img_size: Size2D) -> List[Mask]:
        assert self._deterministic_pipeline is not None
        raise NotImplementedError()
        # w, h = old_image_size
        # heatmap_h, heatmap_w = heatmaps.shape[:2]
        # imgaug_heatmaps = ia.HeatmapsOnImage(heatmaps, shape=(h, w))
        # imgaug_heatmaps = deterministic_aug_pipeline.augment_heatmaps([imgaug_heatmaps])[0]
        # imgaug_heatmaps = imgaug_heatmaps.resize((heatmap_h, heatmap_w))
        # return imgaug_heatmaps.get_arr()

    def _transform_polygons(self, polygons: List[Polygon],
                            before_img_size: Size2D, after_img_size: Size2D) -> List[Polygon]:
        assert self._deterministic_pipeline is not None
        raise NotImplementedError()
        # w, h = old_image_size
        # augmented_polygons = []
        # for polygons in polygons_by_categories:
        #     if len(polygons['polygon']) == 0:
        #         augmented_polygons.append({
        #             'name': polygons['name'],
        #             'polygon': []
        #         })
        #         continue
        #
        #     imgaug_polygons = ia.PolygonsOnImage([
        #         ia.Polygon(list(map(lambda x: tuple(x), p * [w, h])))
        #         for p in polygons['polygon']], shape=(h, w))
        #     imgaug_polygons = deterministic_aug_pipeline.augment_polygons([imgaug_polygons])[0]
        #     imgaug_polygons = imgaug_polygons.remove_out_of_image(fully=True, partly=False)
        #     try:
        #         imgaug_polygons = imgaug_polygons.clip_out_of_image()
        #     except:
        #         pass
        #     augmented_polygons.append({
        #         'name': polygons['name'],
        #         'polygon': [(p.exterior / imgaug_polygons.shape[::-1])
        #                     for p in imgaug_polygons.polygons]
        #     })
        # return augmented_polygons

    def _transform_keypoints(self, keypoints: List[Keypoints],
                             before_img_size: Size2D, after_img_size: Size2D) -> List[Keypoints]:
        assert self._deterministic_pipeline is not None
        raise NotImplementedError()
        # w, h = old_image_size
        # imgaug_keypoints = ia.KeypointsOnImage([ia.Keypoint(*k) for k in keypoints], shape=(h, w))
        # imgaug_keypoints = deterministic_aug_pipeline.augment_keypoints([imgaug_keypoints])[0]
        # return imgaug_keypoints.get_arr()
