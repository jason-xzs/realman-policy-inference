from typing import Optional

from r3kit.devices.camera.realsense.config import D435_ID, D435_STREAMS
from r3kit.devices.camera.realsense.d415 import D415


class D435(D415):
    """RealSense D435，相机流/默认序列号可根据 config 调整。"""

    STREAMS = D435_STREAMS
    DEFAULT_ID = D435_ID

    def __init__(
        self,
        id: Optional[str] = None,
        depth: bool = True,
        name: str = "D435",
        color_width: Optional[int] = None,
        color_height: Optional[int] = None,
        color_fps: Optional[int] = None,
    ) -> None:
        super().__init__(id=id, depth=depth, name=name, color_width=color_width,
                         color_height=color_height, color_fps=color_fps)
