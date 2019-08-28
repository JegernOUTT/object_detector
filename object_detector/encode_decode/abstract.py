from abc import ABC, abstractmethod
from dataclasses import dataclass


class EncodeDecodeModel(ABC):
    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass



@dataclass
class Detector:
    l


data_loader = DataLoader()



for data in data_loader:
    images, gt = data
    predicted = model(images)
