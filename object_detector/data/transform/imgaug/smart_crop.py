import random
from typing import List, Tuple

from imgaug import augmenters as iaa

from object_detector.data.structs import AnnotationInformation
from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.structs import Size2D

__all__ = ['get_smart_crop_augmenter']


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # Sanity check
    if not all([interval[1] >= interval[0] for interval in intervals]):
        raise RuntimeError("Last elements in intervals should be bigger than first")
    if not len(intervals):
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    result = []
    idx = 0
    while idx < len(intervals):
        merged_begin, merged_end = intervals[idx]
        idx += 1
        while idx < len(intervals) and intervals[idx][0] <= merged_end:
            merged_end = max(intervals[idx][1], merged_end)
            idx += 1
        result.append((merged_begin, merged_end))

    return result


def _interval_difference(minuted_interval: Tuple[float, float], deducted_interval: Tuple[float, float]) \
        -> List[Tuple[float, float]]:
    if deducted_interval[0] <= minuted_interval[0]:
        if deducted_interval[1] >= minuted_interval[1]:
            return []
        elif deducted_interval[1] > minuted_interval[0]:
            return [(deducted_interval[1], minuted_interval[1])]
        else:
            return [minuted_interval]
    elif deducted_interval[0] <= minuted_interval[1]:
        if deducted_interval[1] >= minuted_interval[1]:
            return [(minuted_interval[0], deducted_interval[0])]
        else:
            return [(minuted_interval[0], deducted_interval[0]), (deducted_interval[1], minuted_interval[1])]
    else:
        return [minuted_interval]


def _multiple_interval_difference(origin_interval: Tuple[float, float],
                                  intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # intervals are sorted after _merge_intervals
    result = [origin_interval]
    if not len(intervals):
        return result

    for interval in intervals:
        minuted_interval = result[-1]
        interval_difference = _interval_difference(minuted_interval, interval)
        if len(interval_difference) == 2:
            result[-1] = interval_difference[0]
            result.append(interval_difference[1])
        elif len(interval_difference) == 1:
            result[-1] = interval_difference[0]
        else:
            # Nothing left of last interval
            result = result[:-1]
            break

    return result


def _compute_crop_coordinates_1d(box_intervals: List[Tuple[float, float]]):
    allowed_intervals = _multiple_interval_difference((0, 1.0), _merge_intervals(box_intervals))
    if len(allowed_intervals) == 0:
        allowed_intervals = [(0, 0), (1.0, 1.0)]
    else:
        if allowed_intervals[0][0] > 0:
            allowed_intervals = [(0, 0)] + allowed_intervals
        if allowed_intervals[-1][1] < 1.0:
            allowed_intervals.append((1.0, 1.0))

    random.shuffle(allowed_intervals)
    chosen_intervals = sorted(allowed_intervals[:2], key=lambda x: x[0])
    coord_begin = random.uniform(chosen_intervals[0][0], chosen_intervals[0][1])
    coord_end = random.uniform(chosen_intervals[1][0], chosen_intervals[1][1])
    return coord_begin, coord_end


def get_smart_crop_augmenter(
        image_size: Size2D,
        annotations: List[AnnotationInformation],
        min_crop_ratio: float,
        max_crop_ratio: float,
        min_crop_pix_size: int,
        max_crop_pix_size: int) -> iaa.Augmenter:
    repeats = 0

    if len(annotations) <= 1:
        return iaa.Noop()

    # Did not read crop algorithm with attention, may be this rules can be passed to _compute_crop_coordinates_1d
    #  directly for making more effective algorithm
    while True:
        x_intervals = [(ann.annotation.get_xyxy()[0], ann.annotation.get_xyxy()[2])
                       for ann in annotations
                       if type(ann.annotation) == Bbox]
        x_begin, x_end = _compute_crop_coordinates_1d(x_intervals)

        y_intervals = [(ann.annotation.get_xyxy()[1], ann.annotation.get_xyxy()[3])
                       for ann in annotations
                       if type(ann.annotation) == Bbox]
        y_begin, y_end = _compute_crop_coordinates_1d(y_intervals)

        w, h = x_end - x_begin, y_end - y_begin
        ratio = w / h

        if min_crop_ratio <= ratio <= max_crop_ratio and \
                min_crop_pix_size <= int(w * image_size.width) <= max_crop_pix_size and \
                min_crop_pix_size <= int(h * image_size.height) <= max_crop_pix_size:
            break

        repeats += 1
        if repeats > 100:
            return iaa.Noop()

    x_begin, x_end = int(x_begin * image_size.width), int(x_end * image_size.width)
    y_begin, y_end = int(y_begin * image_size.height), int(y_end * image_size.height)

    return iaa.Crop(px=(y_begin, image_size.width - x_end, image_size.height - y_end, x_begin),
                    deterministic=True,
                    keep_size=False)
