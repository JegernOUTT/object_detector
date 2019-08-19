from abc import ABC, abstractmethod

from object_detector.data.structs import DataLoaderResult, LoaderConfig


class AbstractLoader(ABC):
    def __init__(self, loader_config: LoaderConfig):
        self._config: LoaderConfig = loader_config

    @abstractmethod
    def load(self) -> DataLoaderResult:
        pass
