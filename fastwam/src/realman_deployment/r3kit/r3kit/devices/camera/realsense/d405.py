from typing import Optional

from r3kit.devices.camera.realsense.config import D405_ID, D405_STREAMS
from r3kit.devices.camera.realsense.d415 import D415


class D405(D415):
    """RealSense D405 备用实现，沿用 D415 逻辑。"""

    STREAMS = D405_STREAMS
    DEFAULT_ID = D405_ID

    def __init__(self, id: Optional[str] = None, depth: bool = True, name: str = "D405") -> None:
        super().__init__(id=id, depth=depth, name=name)
