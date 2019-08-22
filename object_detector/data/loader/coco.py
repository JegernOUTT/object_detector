from pathlib import Path
from typing import List, Tuple, Type

from dataclasses import dataclass
from pycocotools.coco import COCO

from object_detector.data.loader.abstract import AbstractLoader
from object_detector.data.structs import LoaderConfig, ImageInformation, AnnotationInformation
from object_detector.tools.bbox.bbox import from_xywh
from object_detector.tools.mask.mask import CocoMaskWrapper
from object_detector.tools.structs import Size2D


@dataclass
class CocoLoaderConfig(LoaderConfig):
    annotations_path: Path = Path()
    images_path: Path = Path()
    check_images: bool = False
    min_image_size: int = 32
    is_crowd_load_type: LoaderConfig.LoadType = LoaderConfig.LoadType.AsIgnore
    load_bbox_masks: bool = False
    load_keypoints: bool = False

    def _loader_type(self) -> Type['CocoDataset']:
        return CocoDataset


class CocoDataset(AbstractLoader):
    AVAILABLE_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                         'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
                         'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
                         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                         'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                         'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
                         'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                         'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
                         'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
                         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                         'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, loader_config: CocoLoaderConfig):
        super().__init__(loader_config)
        assert not self._config.load_keypoints, "Coco keypoints not implemented"
        assert all([CocoDataset.AVAILABLE_CLASSES in c for c in self._config.categories])
        self._img_infos = self._load_annotations()
        self._filter_imgs()

    def _load_single(self, idx: int) -> Tuple[ImageInformation, List[AnnotationInformation]]:
        image_filename = self._config.images_path / self._img_infos[idx]['filename']
        assert image_filename.exists(), f"Image not found: {image_filename}"

        ann_ids = self._coco.getAnnIds(imgIds=[self._img_infos[idx]['id']])
        ann_info = self._coco.loadAnns(ann_ids)
        return ImageInformation(filename=image_filename), self._parse_ann_info(self._img_infos[idx], ann_info)

    def images_count(self) -> int:
        return len(self._img_infos)

    def _load_annotations(self) -> List[dict]:
        self._coco = COCO(str(self._config.annotations_path))
        self._cat_ids = self._coco.getCatIds(catNms=self._config.categories)
        self._cat2label = {cat_id: i for i, cat_id in enumerate(self._cat_ids)}
        self._img_ids = self._coco.getImgIds()
        img_infos = []
        for i in self._img_ids:
            info = self._coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def _filter_imgs(self):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self._coco.anns.values())
        for i, img_info in enumerate(self._img_infos):
            if self._img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= self._config.min_image_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info: dict, ann_info: List[dict]) -> List[AnnotationInformation]:
        image_size = Size2D(width=img_info['width'], height=img_info['height'])
        annotations: List[AnnotationInformation] = []

        for i, ann in enumerate(ann_info):
            bbox = from_xywh(xywh=ann['bbox'], image_size=image_size)
            ignore = False

            if bbox.area(image_size=image_size) == 0:
                continue

            if ann['iscrowd']:
                if self._config.is_crowd_load_type == LoaderConfig.LoadType.Skip:
                    continue
                elif self._config.is_crowd_load_type == LoaderConfig.LoadType.AsIgnore:
                    ignore = True

            category_id = self._cat2label[ann['category_id']]

            if self._config.load_bbox_masks:
                bbox.mask = CocoMaskWrapper.from_polygon(ann, image_size).numpy()

            annotations.append(AnnotationInformation(
                annotation=bbox, confidence=1., category_id=category_id, ignore=ignore))

        return annotations
