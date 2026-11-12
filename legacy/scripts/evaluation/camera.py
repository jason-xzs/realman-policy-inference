import logging

import cv2
import numpy as np
import pyrealsense2 as rs


class Camera:
    def __init__(self, serial_number="150622070802", resolution=(480, 640), fps=30, allow_profile_fallback=False):
        self.serial_number = serial_number
        self.resolution = resolution
        self.fps = fps
        self.allow_profile_fallback = allow_profile_fallback
        self.pipeline = None
        self.active_resolution = None
        self.active_fps = None

        try:
            success = self.configure_camera_pipeline()
            if not success:
                raise ConnectionError("Failed to start RealSense pipeline.")
        except Exception as exc:
            logging.error("[ERROR] 初始化相机失败: %s", exc)
            raise

    def _candidate_profiles(self):
        requested_h, requested_w = self.resolution
        requested = (requested_h, requested_w, self.fps)
        candidates = [requested]
        if not self.allow_profile_fallback:
            return candidates

        fallback_profiles = [
            (480, 640, self.fps),
            (360, 640, self.fps),
            (720, 1280, self.fps),
        ]
        fallback_profiles.extend(
            [
                (480, 640, 30),
                (360, 640, 30),
                (720, 1280, 30),
                (480, 640, 15),
                (360, 640, 15),
                (720, 1280, 15),
            ]
        )

        for candidate in fallback_profiles:
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def configure_camera_pipeline(self):
        """配置并启动 RealSense color-only 管道。"""
        last_error = None
        for height, width, fps in self._candidate_profiles():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            try:
                pipeline.start(config)
                self.pipeline = pipeline
                self.active_resolution = (height, width)
                self.active_fps = fps
                logging.info(
                    "相机启动成功 [Serial: %s, resolution=%s, fps=%s]",
                    self.serial_number,
                    self.active_resolution,
                    self.active_fps,
                )
                if (height, width, fps) != (self.resolution[0], self.resolution[1], self.fps):
                    logging.warning(
                        "相机 %s 未使用请求配置 resolution=%s fps=%s，已改用 resolution=%s fps=%s",
                        self.serial_number,
                        self.resolution,
                        self.fps,
                        self.active_resolution,
                        self.active_fps,
                    )
                return True
            except Exception as exc:
                last_error = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                logging.warning(
                    "相机 %s 无法使用 color 配置 resolution=(%s, %s), fps=%s: %s",
                    self.serial_number,
                    height,
                    width,
                    fps,
                    exc,
                )

        logging.error("[ERROR] 启动 RealSense 管道失败: %s", last_error)
        return False

    def capture_frame(self):
        """
        捕获一帧彩色图像。
        成功返回 {'rgb': rgb_image}
        失败返回 None
        """
        if self.pipeline is None:
            logging.warning("Pipeline 尚未初始化或已被关闭")
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            rgb_frame = frames.get_color_frame()
            if not rgb_frame:
                logging.warning("获取到无效 RGB 帧（可能是丢帧）")
                return None

            rgb_image = np.asanyarray(rgb_frame.get_data())
            return {"rgb": rgb_image}
        except RuntimeError as exc:
            logging.error("[RuntimeError] 捕获过程中发生错误: %s", exc)
            return None
        except Exception as exc:
            logging.error("[Exception] 捕获过程中发生未知错误: %s", exc)
            return None

    def save_frame(self, frame_data, prefix):
        """
        保存一帧 RGB 图像。
        :param frame_data: 包含 rgb 的字典
        :param prefix: 文件名前缀
        :return: 是否保存成功
        """
        rgb_image = frame_data["rgb"]
        rgb_path = f"{prefix}_rgb.png"
        rgb_success = cv2.imwrite(rgb_path, rgb_image)

        if rgb_success:
            logging.info("保存图像: %s", rgb_path)
            return True

        logging.error("图像保存失败: %s", rgb_path)
        return False

    def stop(self):
        """安全停止管道"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                logging.info("RealSense 管道已关闭")
            except Exception as exc:
                logging.error("关闭管道时出错: %s", exc)
            finally:
                self.pipeline = None
