# -*- coding: utf-8 -*-

import numpy as np
import cv2
import copy

def camera_default_setting(cam_id = 1):
    cap = cv2.VideoCapture(cam_id)
    ret_val, image = cap.read()
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128) #明るさ補正 デフォルト128
    cap.set(cv2.CAP_PROP_SATURATION, 128) #彩度補正 デフォルト128
    cap.set(cv2.CAP_PROP_CONTRAST, 128) #コントラスト補正 デフォルト128
    cap.set(cv2.CAP_PROP_GAIN, 128) #ゲイン補正 デフォルト128
    ret_val, image = cap.read()

def info(cam_id = 1):
    cap = cv2.VideoCapture(cam_id)
    ret_val, image = cap.read()
    ret_val, image = cap.read()
    ret_val, image = cap.read()
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128) #明るさ補正 デフォルト128
    cap.set(cv2.CAP_PROP_SATURATION, 128) #彩度補正 デフォルト128
    cap.set(cv2.CAP_PROP_CONTRAST, 128) #コントラスト補正 デフォルト128
    cap.set(cv2.CAP_PROP_GAIN, 128) #ゲイン補正 デフォルト128
    ret_val, image = cap.read()

    print("%s, %f" %('FRAME_WIDTH',cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("%s, %f" %('FRAME_HEIGHT',cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("%s, %f" %('BRIGHTNESS',cap.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("%s, %f" %('SATURATION',cap.get(cv2.CAP_PROP_SATURATION)))
    print("%s, %f" %('CONTRAST',cap.get(cv2.CAP_PROP_CONTRAST)))
    print("%s, %f" %('PROP_GAIN',cap.get(cv2.CAP_PROP_GAIN)))

    cv2.imwrite('../io_image/test.png',image)

def brightness_adjust(img, brightness):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    if brightness > 0:
        v[v > 255-brightness] = 255
        v[v <= 255-brightness] += brightness
    elif brightness < 0:
        abs_brightness = np.abs(brightness)
        v[v < 0+abs_brightness] = 0
        v[v >= 0+abs_brightness] -= abs_brightness

    hsv_img = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return img

def gamma_cvt_adjust(gamma):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return gamma_cvt

def manual_soft(cam_id = 1):
    print('camera manual setting. software control')
    gamma = 1.0
    gamma_cvt = gamma_cvt_adjust(gamma)
    brightness = 0

    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")
    ret, img = cap.read()
    while True:
        print('brightness', brightness)
        print('gamma', gamma)

        ret, img = cap.read()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.LUT(img,gamma_cvt)
        img = brightness_adjust(img, brightness)

        if ret == False:
            continue
        cv2.imshow('camera test',img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if k == ord('j'):
            brightness -= 2
        if k == ord('k'):
            brightness += 2
        if k == ord('h'):
            if gamma > 0:
                gamma -= 0.1 
            gamma_cvt = gamma_cvt_adjust(gamma)
        if k == ord('l'):
            gamma += 0.1
            gamma_cvt = gamma_cvt_adjust(gamma)
        
    cap.release()
    cv2.destroyAllWindows()

    return gamma, brightness

def auto_soft(cam_id = 1):
    print('camera manual setting.　software control')
    gamma = 1.0
    gamma_cvt = gamma_cvt_adjust(gamma)
    brightness = 0

    box_rate = 10 # startup trgetbox size rate
    move_rate = 100 # startup trgetbox size rate

    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")
    ret, img = cap.read()
    img_h, img_w = img.shape[:2]
    box_size = img_h//box_rate
    box_center_x, box_center_y = img_h//2, img_w//2
    move_step = img_h//box_rate
    while True:
        print('brightness', brightness)
        print('gamma', gamma)

        ret, img = cap.read()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.LUT(img,gamma_cvt)
        img = brightness_adjust(img, brightness)

        if ret == False:
            continue
        cv2.imshow('camera test',img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        if k == ord('j'):
            brightness -= 2
        if k == ord('k'):
            brightness += 2
        if k == ord('h'):
            if gamma > 0:
                gamma -= 0.1 
            gamma_cvt = gamma_cvt_adjust(gamma)
        if k == ord('l'):
            gamma += 0.1
            gamma_cvt = gamma_cvt_adjust(gamma)
        
    cap.release()
    cv2.destroyAllWindows()

    return gamma, brightness


def calibration(cam_id, calib_count_max = 20):
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")

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

def save_calib_data(mtx, dist, mtx_filename='cam_calib_mtx.csv', dist_filename='cam_calib_dist.csv'):
    # Save the csv file
    np.savetxt(mtx_filename, mtx, delimiter=",")
    np.savetxt(dist_filename, dist, delimiter=",")
    print('save', mtx_filename)
    print('save', dist_filename)

def load_calib_data(mtx_filename='cam_calib_mtx.csv', dist_filename='cam_calib_dist.csv'):
    """
    説明文
    """
    mtx = np.loadtxt("mtx.csv", delimiter=",")
    dist = np.loadtxt("dist.csv", delimiter=",")
    print('load', mtx_filename)
    print('load', dist_filename)
    return mtx, dist

def review(cam_id, mtx, dist):
    print('calibrated review')
    cap = cv2.VideoCapture(cam_id)

    if cap.isOpened() is False:
        raise("IO Error")
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
    cam_id = 1
    manual_soft()
