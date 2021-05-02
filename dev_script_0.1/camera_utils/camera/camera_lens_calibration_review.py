# -*- coding: utf-8 -*-

import numpy as np
import cv2
import copy

import camera_lens_calibration

def review_example(cam_id, mtx_filename, dist_filename):
    print('calibrated review')

    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")
    ret, img = cap.read()
    cam_width = 1280
    cam_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    ret, image = cap.read()

    h,  w = image.shape[:2]
    
    ### 必要
    mtx, dist = camera_lens_calibration.load_calib_data(mtx_filename, dist_filename)
    newcameramtx, _ =cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    ###

    while True:
        ret, image = cap.read()
        if ret == False:
            continue
        ### 必要
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        ###
        cv2.imshow('calibration_review',dst)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def review_example_image(image_filename, mtx_filename, dist_filename):
    ret, image = cap.read()

    h,  w = image.shape[:2]
    mtx, dist = camera_lens_calibration.load_calib_data(mtx_filename, dist_filename)
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    image = cv2.imread(image_filename)
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    cv2.imshow('calibration_review', dst)
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_id = 0
    calib_count_max = 20
    mtx_filename = 'cam_calib_mtx.csv'
    dist_filename = 'cam_calib_dist.csv'

    review_example(cam_id, mtx_filename, dist_filename)
