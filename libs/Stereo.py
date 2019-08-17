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


class Stereo:

    def __init__(self, image_dir, img_shape=None):
        """
        Initialize camera matrices.

        :param image_dir: Image directory
        :param img_shape: Optional image shape.  If not provided, an image file will be read to determine the shape.
        """
        self.__image_dir = image_dir

        self.__setup = importlib.import_module("{}.setup".format(image_dir))

        self.__K = np.load(os.path.join(image_dir, "K.npy"))
        self.__R = np.load(os.path.join(image_dir, "R.npy"))
        self.__T = np.load(os.path.join(image_dir, "T.npy"))
        self.__D = np.load(os.path.join(image_dir, "D.npy"))

        if img_shape is None:
            # Read an image file to determine the size
            fname = self.__setup.RigInfo.image_filename[0].format("0")
            img = np.load(os.path.join(image_dir, fname))
            img_shape = img.shape
        self.__img_shape = (img_shape[0], img_shape[1])

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
    def compute_fundamental_matrix(K1, K2, R, T):
        """
        Computes the fundamental matrix from KRT.

        :return F: Fundamental matrix
        """
        P1 = np.matmul(K1, np.hstack((np.identity(3), np.zeros((3, 1)))))
        P2 = np.matmul(K2, np.hstack((R, T)))
        Pp = np.vstack((np.linalg.inv(K1), np.zeros(3)))
        C = np.zeros((4, 1))
        C[3] = 1
        #F = np.matmul(Stereo.skew_symetric(np.matmul(P2, C)), np.matmul(P2, Pp))
        F = np.cross(np.matmul(P2, C)[:,0], np.matmul(P2, Pp)[:,0])
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
    def compute_projection_matrix(K, R, T):
        P = np.hstack((np.matmul(K, R), T))
        return P

    def fundamental_matrix(self):
        return Stereo.compute_fundamental_matrix(self.__K[0], self.__K[1], self.__R[1], self.__T[1])

    def essential_matirx(self):
        F = self.fundamental_matrix()
        return Stereo.compute_essential_matrix(self.__K[0], self.__K[1], F)

    def projection_matrix(self, cam_idx):
        """
        Compute the Projection Matrix for 3D to 2D transformations

        :param cam_idx: Camera Index
        :return P: Projection Matrix
        """
        return Stereo.compute_projection_matrix(self.__K[cam_idx], self.__R[cam_idx], self.__T[cam_idx])

    def compute_distance_disparity(self, img_pts):
        """
        Compute disparity given image points across multiple views

        :param img_pts: image plane pixel locations in each view - Numpy.array([num_cam, num_pts, x, y])
        :return distance: Disparity distance in camera matrix units
                            - Numpy.array([num_cam-1, distance])
        :return disparity: Disparity pixels relative to the reference camera (ie cam_idx 0)
                            - Numpy.array([num_cam-1, disparity_pixels])
        """
        if img_pts.shape[0] < 2:
            print("At least 2 cameras are need to compute disparity")
            return None

        disparity = np.empty((img_pts.shape[0] - 1, img_pts.shape[1]))
        distance = np.empty(disparity.shape)
        for cam_idx in range(1, img_pts.shape[0]):
            # Rectify
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.__K[0], self.__D[0],
                                                              self.__K[cam_idx], self.__D[cam_idx],
                                                              self.__img_shape,
                                                              self.__R[cam_idx], self.__T[cam_idx])

            # Rectify reference camera points
            undist_ref = cv2.undistortPoints(src=img_pts[0], cameraMatrix=self.__K[0], distCoeffs=self.__D[0],
                                             R=R1, P=P1)
            undist_src = cv2.undistortPoints(src=img_pts[cam_idx], cameraMatrix=self.__K[cam_idx],
                                             distCoeffs=self.__D[cam_idx], R=R2, P=P2)

            # Compute disparity in pixels
            diff = undist_src - undist_ref
            disparity[cam_idx - 1, :] = diff[:, 0, 0]

            # Compute distance in camera matrix units
            distance[cam_idx - 1, :] = P2[0, 3] / disparity[cam_idx - 1, :]
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
        x = cv2.undistortPoints(src=pts, cameraMatrix=self.__K[0], distCoeffs=self.__D[0], R=np.identity(3), P=P1)

        # Project the ideal points from the reference camera to the ideal points in cam_idx
        # x' = Hx + Kt/Z
        x = x.reshape(-1, 2).T
        x = np.vstack((x, np.ones((1, x.shape[1]))))

        # H = K' R K^(-1)
        x_p = np.matmul(np.matmul(np.matmul(self.__K[cam_idx], self.__R[cam_idx]),
                                              np.linalg.inv(self.__K[0])), x)

        # Kt/z
        x_p += np.matmul(self.__K[cam_idx], self.__T[cam_idx]) / distance
        x_p /= x_p[2]

        # Add distortion of cam 2
        a = cv2.undistortPoints(src=x_p[0:2].T.reshape(-1, 1, 2), cameraMatrix=self.__K[cam_idx], distCoeffs=np.zeros(5))
        a = np.concatenate((a, np.ones((a.shape[0], 1, 1))), axis=2)
        reproj_pts, J = cv2.projectPoints(a, np.zeros(3), np.zeros(3), self.__K[cam_idx], self.__D[cam_idx])

        return reproj_pts


