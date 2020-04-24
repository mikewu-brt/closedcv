"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    August 2019
 * @brief
 *  Stereo helper functions
 *
"""
import os
import importlib
import numpy as np
import cv2
from libs.CalibrationInfo import *


class Stereo:

    def __init__(self, image_dir, img_shape=None):
        """
        Initialize camera matrices.

        :param image_dir: Image directory
        :param img_shape: Optional image shape.  If not provided, an image file will be read to determine the shape.
        """
        path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
        if path_to_image_dir is None:
            path_to_image_dir = '.'
        self.__image_dir = os.path.join(path_to_image_dir, image_dir)

        self.__setup = importlib.import_module("{}.setup".format(image_dir))

        self.__cal_info = CalibrationInfo(cal_dir=image_dir, calibration_json_fname="calibration.json")

        if img_shape is None:
            # Read an image file to determine the size
            img_shape = (self.__setup.SensorInfo.width, self.__setup.SensorInfo.height)
        self.__img_shape = img_shape

    @staticmethod
    def skew_symetric(vector_in):
        """
        Converts vector_in into a skew symetric matrix.  This is useful for computing cross products
        of 2 vectors through matrix multiplication.

        A x B = [A]x B where [A]x is the skew_symetric(A).  [A]x is a 3x3 matrix and B is a 3x1 matrix.

        :param vector_in: 3x1 or 1x3 vector
        :return s: [vector_in]x
        """
        if vector_in.size != 3:
            print("skew_symetric matrix must only contain 3 elements")
            exit(-1)

        s = np.zeros((3, 3), dtype=np.float)
        s[0, 1] = -vector_in[2]
        s[1, 0] = vector_in[2]

        s[0, 2] = vector_in[1]
        s[2, 0] = -vector_in[1]

        s[1, 2] = -vector_in[0]
        s[2, 1] = vector_in[0]
        return s

    @staticmethod
    def compute_projection_matrix(K, R, T):
        P = np.matmul(K, np.hstack((R, T)))
        return P

    @staticmethod
    def compute_fundamental_matrix(K1, K2, R, T):
        """
        Computes the fundamental matrix from KRT.  Assumes P1 = K1 [ I | 0]

        :return F: Fundamental matrix
        """
        P1 = Stereo.compute_projection_matrix(K1, np.identity(3), np.zeros((3, 1)))
        P2 = Stereo.compute_projection_matrix(K2, R, T)
        Pp = np.vstack((np.linalg.inv(K1), np.zeros(3)))
        C = np.zeros((4, 1))
        C[3] = 1
        F = np.matmul(Stereo.skew_symetric(np.matmul(P2, C)), np.matmul(P2, Pp))
        return F

    @staticmethod
    def compute_essential_matrix(K1, K2, F):
        """
        Computes the essential matrix from KRT.

        :return E: Essential matrix
        """
        E = np.matmul(np.matmul(K2.transpose(), F, K1))
        return E

    @staticmethod
    def compute_homography_inf(K1, K2, R):
        Hinf = np.matmul(np.matmul(K2, R), np.linalg.inv(K1))
        return Hinf

    def cal_info(self):
        return self.__cal_info

    def fundamental_matrix(self, cam_idx):
        return Stereo.compute_fundamental_matrix(self.__cal_info.K(0),
                                                 self.__cal_info.K(cam_idx),
                                                 self.__cal_info.R(cam_idx),
                                                 self.__cal_info.T(cam_idx))

    def essential_matirx(self, cam_idx):
        F = self.fundamental_matrix(cam_idx)
        return Stereo.compute_essential_matrix(self.__cal_info.K(0), self.__cal_info.K(cam_idx), F)

    def projection_matrix(self, cam_idx):
        """
        Compute the Projection Matrix for 3D to 2D transformations

        :param cam_idx: Camera Index
        :return P: Projection Matrix
        """
        return Stereo.compute_projection_matrix(self.__cal_info.K(cam_idx),
                                                self.__cal_info.R(cam_idx),
                                                self.__cal_info.T(cam_idx))

    def rectification_matrix(self, cam_idx):
        return cv2.stereoRectify(self.__cal_info.K(0),
                                 self.__cal_info.D(0),
                                 self.__cal_info.K(cam_idx),
                                 self.__cal_info.D(cam_idx),
                                 self.__img_shape,
                                 self.__cal_info.R(cam_idx),
                                 self.__cal_info.T(cam_idx))

    def homography_inf(self, cam_idx):
        return Stereo.compute_homography_inf(self.__cal_info.K(0), self.__cal_info.K(cam_idx), self.__cal_info.R(cam_idx))

    def undistort_image(self, img, cam_idx):
        return cv2.undistort(img, self.__cal_info.K(cam_idx), self.__cal_info.D(cam_idx))

    def compute_distance_disparity(self, img_pts, T=None):
        """
        Compute disparity given image points across multiple views

        :param img_pts: image plane pixel locations in each view - Numpy.array([num_cam, num_pts, x, y])
        :param T: Translation vector if input is rectified; otherwise, leave unset
        :return distance: Disparity distance in camera matrix units
                            - Numpy.array([num_cam-1, distance])
        :return disparity: Disparity pixels relative to the reference camera (ie cam_idx 0)
                            - Numpy.array([num_cam-1, disparity_pixels])
        """
        if len(img_pts) < 2:
            print("At least 2 cameras are need to compute disparity")
            return None

        disparity = np.empty((len(img_pts) - 1, img_pts[0].shape[0]))
        distance = np.empty(disparity.shape)
        for cam_idx in range(1, len(img_pts)):
            if T is None:
                # Rectify
                R1, R2, P1, P2, Q, roi1, roi2 = self.rectification_matrix()

                # Rectify reference camera points
                rect_ref = cv2.undistortPoints(src=img_pts[0], cameraMatrix=self.__cal_info.K(0),
                                               distCoeffs=self.__cal_info.D(0), R=R1, P=P1)
                rect_src = cv2.undistortPoints(src=img_pts[cam_idx], cameraMatrix=self.__cal_info.K(cam_idx),
                                                 distCoeffs=self.__cal_info.D(cam_idx), R=R2, P=P2)

                rect_ref = rect_ref[:, 0, :]
                rect_src = rect_src[:, 0, :]
                T = P2[:, 3]
            else:
                rect_ref = img_pts[0]
                rect_src = img_pts[cam_idx]

            # Compute disparity in pixels
            diff = rect_src - rect_ref
            disparity[cam_idx - 1] = diff[:, 0]

            # Compute distance in camera matrix units
            distance[cam_idx - 1] = T[0] / disparity[cam_idx - 1]
        return distance, disparity

    def compute_distance_triangulate(self, img_pts):
        """
        Computes the object distance using vector intersection in camera matrix units.

        :param img_pts: image plane pixel locations in each view - Numpy.array([num_cam, image_x_pts, image_y_pts])
        :return distance: Disparity distance in camera matrix units
                            - Numpy.array([num_cam-1, distance])
        """
        print("compute_distance_triangulate not implemented yet!")
        exit(-1)

    def reproject_points(self, pts, distance, cam_idx):
        """
        Reprojects points from one camera view to another.

        :param pts:  x,y image plane points of reference camera to re-project to another camera
                        - Numpy array Nx2x1 (Vector of 2D points)
        :param distance:  Distance in camera matrix units.
        :param cam_idx: Which camera index to re-project the points to.
        :return reprojected_pts: Reprojected points into "to" camera view
        """
        # First undistort the observed points to the ideal points
        P1 = self.projection_matrix(0)
        x = cv2.undistortPoints(src=pts, cameraMatrix=self.__cal_info.K(0),
                                distCoeffs=self.__cal_info.D(0), R=np.identity(3), P=P1)

        # Project the ideal points from the reference camera to the ideal points in cam_idx
        # x' = Hinf x + Kt/Z
        x = cv2.convertPointsToHomogeneous(x).reshape(-1, 3, 1)

        # Hinf = K' R K^(-1)
        Hinf = self.homography_inf(cam_idx)
        x_p = cv2.convertPointsFromHomogeneous(np.matmul(Hinf, x))

        # Kt/z
        x_d = np.matmul(self.__cal_info.K(cam_idx), self.__cal_info.T(cam_idx)) / distance
        x_d = x_d[:2]
        x_p += x_d.T.reshape(-1, 1, 2)

        # Add distortion of cam 2
        a = cv2.undistortPoints(src=x_p, cameraMatrix=self.__cal_info.K(cam_idx), distCoeffs=np.zeros(5))
        a = np.concatenate((a, np.ones((a.shape[0], 1, 1))), axis=2)
        reproj_pts, J = cv2.projectPoints(a, np.zeros(3), np.zeros(3),
                                          self.__cal_info.K(cam_idx), self.__cal_info.D(cam_idx))

        return reproj_pts

    def rectify_views(self, ref_img, src_img, src_cam_idx, valid_roi_only=False):
        """
        Rectifies views

        :param ref_img: Reference image
        :param src_img: Source image
        :param src_cam_idx: Camera index of Source camera
        :param valid_roi_only: Truncates the rectified images to the valid ROI
        :return: Rectified reference and source views
        :return: New rectified P1, P2 matrices
        """
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.__cal_info.K(0), self.__cal_info.D(0),
                                                          self.__cal_info.K(src_cam_idx), self.__cal_info.D(src_cam_idx),
                                                          self.__img_shape,
                                                          self.__cal_info.R(src_cam_idx), self.__cal_info.T(src_cam_idx))

        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.__cal_info.K(0), self.__cal_info.D(0), R1, P1,
                                                   self.__img_shape,
                                                   cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.__cal_info.K(src_cam_idx), self.__cal_info.D(src_cam_idx), R2, P2,
                                                   self.__img_shape,
                                                   cv2.CV_32F)

        ref_img = cv2.remap(ref_img, mapx1, mapy1, cv2.INTER_LINEAR)
        src_img = cv2.remap(src_img, mapx2, mapy2, cv2.INTER_LINEAR)

        if valid_roi_only:
            ref_img = ref_img[roi1[1]:roi1[3], roi1[0]:roi1[2]]
            src_img = src_img[roi2[1]:roi2[3], roi2[0]:roi2[2]]

        return ref_img, src_img, P1, P2, R1, R2

    @staticmethod
    def drawlinesnew(img1, img2, lines, pts1, pts2, circle_thickness, circles=True):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r = img1.shape[0]
        c = img1.shape[1]
        if np.ndim(img1) < 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if np.ndim(img2) < 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color =  (0,255,0)
            if abs(r[1]) > abs(r[0]):
                x0, y0 = map(int, [0, -r[2] / r[1]])
                x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            else:
                x0, y0 = map(int, [-r[2] / r[0], 0])
                x1, y1 = map(int, [-(r[2] + r[1] * c) / r[0], c])
            img1 = cv2.line(img1, (x0*16, y0*16), (x1*16, y1*16), color, 2, shift=4)
            pt1 = np.int32(pt1)
            pt2 = np.int32(pt2)
            if circles:
                img1 = cv2.circle(img1, (pt1[0,0],pt1[0,1]), 5, color, circle_thickness)
                img2 = cv2.circle(img2, (pt2[0,0], pt2[0,1]), 5, color, circle_thickness)
        return img1, img2

    @staticmethod
    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r = img1.shape[0]
        c = img1.shape[1]
        if np.ndim(img1) < 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if np.ndim(img2) < 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def draw_epilines(self, img_ref, img_src, pts_ref, pts_src, cam_idx, P1=None, P2=None):
        if P1 is None:
            F = self.fundamental_matrix(cam_idx)
        else:
            F = Stereo.compute_fundamental_matrix(P1[:, :3], P2[:, :3], np.identity(3), P2[:, 3].reshape((3, 1)))
        line_ref = cv2.computeCorrespondEpilines(pts_src, 1, F).reshape(-1, 3)
        line_src = cv2.computeCorrespondEpilines(pts_ref, 2, F).reshape(-1, 3)
        img_ref, img1 = Stereo.drawlines(img_ref.copy(), img_src.copy(), line_ref, pts_ref, pts_src)
        img_src, img2 = Stereo.drawlines(img_src.copy(), img_ref.copy(), line_src, pts_src, pts_ref)

        return img_ref, img_src
