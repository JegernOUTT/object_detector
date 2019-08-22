from typing import List, Dict

from dataclasses import dataclass

from object_detector.data.structs import LoaderConfig, TransformConfig, HeadFormatterConfig


@dataclass
class DataSettings:
    loaders: List[LoaderConfig]
    transformers: Dict[int, TransformConfig]
    head_formatters: List[HeadFormatterConfig]
    sampler: object
    collate_fn: object
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool
