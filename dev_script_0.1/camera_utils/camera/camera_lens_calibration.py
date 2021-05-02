# -*- coding: utf-8 -*-

import numpy as np
import cv2
import copy


def create_matrix(cam_id, calib_count_max = 20):
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")

    cam_width = 1280
    cam_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    calib_count = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            continue
        gray = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        cv2.imshow('calibration', img)
        cv2.putText(img, 'Number of capture: '+str(calib_count), (30, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(img, 'q: Finish capturing and calcurate the camera matrix and distortion', (30, 60), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))

        k = cv2.waitKey(1) & 0xFF
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        # Find the chess board corners
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('calibration', img)
            cv2.waitKey(1000)

            calib_count += 1

        if k == ord('q') or calib_count > calib_count_max:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Calc urate the camera matrix...')
    # Calc urate the camera matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print('ret --> ', ret)
    print('mtx --> ', mtx)
    print('dist --> ', dist)
    print('rvecs --> ', rvecs)
    print('tvecs --> ', tvecs)

    return ret, mtx, dist, rvecs, tvecs

def save_calib_data(mtx, dist, mtx_filename = 'cam_calib_mtx.csv', dist_filename = 'cam_calib_dist.csv'):
    # Save the csv file
    np.savetxt(mtx_filename, mtx, delimiter=",")
    np.savetxt(dist_filename, dist, delimiter=",")
    print('save', mtx_filename)
    print('save', dist_filename)

def load_calib_data(mtx_filename = 'cam_calib_mtx.csv', dist_filename = 'cam_calib_dist.csv'):
    mtx = np.loadtxt(mtx_filename,delimiter=",")
    dist = np.loadtxt(dist_filename,delimiter=",")
    print('load', mtx_filename)
    print('load', dist_filename)
    return mtx, dist

def review(cam_id, mtx, dist):
    print('calibrated review')
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")
    ret, img = cap.read()
    cam_width = 1280
    cam_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    ret, img = cap.read()
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    while True:
        ret, img = cap.read()
        if ret == False:
            continue
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow('calibration_review',dst)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_id = 0
    calib_count_max = 10
    mtx_filename = 'cam_calib_mtx.csv'
    dist_filename = 'cam_calib_dist.csv'

    ret, mtx, dist, rvecs, tvecs = create_matrix(cam_id, calib_count_max)
    save_calib_data(mtx, dist, mtx_filename, dist_filename)
    mtx, dist = load_calib_data(mtx_filename, dist_filename)
    review(cam_id, mtx, dist)
