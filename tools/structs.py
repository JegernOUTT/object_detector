from dataclasses import dataclass


@dataclass(frozen=True)
class Size2D:
    width: int
    height: int


@dataclass(frozen=True)
class Size3D:
    width: int
    height: int
    channels: int
