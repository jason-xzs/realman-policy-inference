import cv2
import numpy as np

class ArucoExtCalibor(object):
    def __init__(self, dict_id=cv2.aruco.DICT_6X6_250, marker_length=0.16856):
        """
        dict_id: aruco字典类型，例如 cv2.aruco.DICT_6X6_250
        marker_length: 实际二维码边长（米）
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.obj_points = []  # 世界坐标系下marker的角点
        self.img_points = []  # 图像坐标系下的角点
        self.marker_length = marker_length
        self.camera_matrix = None
        self.dist_coeffs = None
        self.w2c_list = []

    def set_camera_intrinsics(self, K, dist):
        """传入相机内参（标定结果）"""
        self.camera_matrix = K
        self.dist_coeffs = dist

    def set_detector_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.parameters, k):
                setattr(self.parameters, k, v)
            
    def add_image(self, img, vis=True):
        """
        检测aruco二维码并保存角点
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            print("❌ 没检测到 ArUco Marker")
            # 可选：也弹出图片，辅助你直观看内容
            cv2.imshow("debug_no_marker", img)
            cv2.imwrite(f'calib0802/vis_no_marker_{len(self.w2c_list)}.png', img)
            cv2.waitKey(100)
            return False
        
        # 检测姿态
        ret = False
        for c, id in zip(corners, ids):
            if self.camera_matrix is not None:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(c, self.marker_length, self.camera_matrix, self.dist_coeffs)
                if vis:
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                    cv2.imshow("aruco detect", img)
                    cv2.waitKey(100)
                    cv2.imwrite(f'vis_aruco_{len(self.w2c_list)}.png', img)
                # 记录 w2c（marker->camera）
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                self.obj_points.append(T)
                ret = True
        return ret

    def run(self):
        """
        返回所有采集的 marker->camera 的姿态 (w2c)
        """
        if len(self.obj_points) == 0:
            return None
        return np.array(self.obj_points)
