from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import torch


class Encoder(ABC):
    @abstractmethod
    def encode(self, gt: dict) -> Dict[str, torch.Tensor]:
        pass


class Decoder(ABC):
    @abstractmethod
    def decode(self, predicted: Dict[str, torch.Tensor]) -> Dict[Any, Any]:
        pass


@dataclass
class EncoderDecoderConfig(ABC):
    @abstractmethod
    def create_encoder(self) -> Encoder:
        pass

    @abstractmethod
    def create_decoder(self) -> Decoder:
        pass
