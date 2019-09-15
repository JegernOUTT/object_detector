from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import torch

from object_detector.encode_decode.abstract import Encoder, Decoder, EncoderDecoderConfig
from object_detector.tools.bbox.bbox import Bbox


@dataclass
class CenternetDetectionEncoderDecoderConfig(EncoderDecoderConfig):
    gt_data_names: Tuple[str] = ('centernet_xy', 'centernet_xy_offset', 'centernet_wh')
    decoder_tensor_names: Tuple[str] = ('centernet_xy', 'centernet_xy_offset', 'centernet_wh')

    def create_encoder(self):
        return CenternetDetectionEncoder(self)

    def create_decoder(self):
        return CenternetDetectionDecoder(self)


class CenternetDetectionEncoder(Encoder):
    def __init__(self, config: CenternetDetectionEncoderDecoderConfig):
        self._config = config

    def encode(self, gt: dict) -> Dict[str, torch.Tensor]:
        for name in self._config.gt_data_names:
            assert name in gt and type(gt[name]) == torch.Tensor
        return gt  # All work done in formatter


class CenternetDetectionDecoder(Decoder):
    def __init__(self, config: CenternetDetectionEncoderDecoderConfig):
        self._config = config

    def decode(self, predicted: Dict[str, torch.Tensor]) -> Dict[str, List[Bbox]]:
        raise NotImplementedError()
