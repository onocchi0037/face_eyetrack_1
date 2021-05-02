# -*- coding: utf-8 -*-



import tensorflow as tf
import datetime
import json
import sys
import subprocess
import cv2
import numpy as np
import time
import argparse
import copy
import dlib
import math
import os
import copy

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn import preprocessing


import posenet

import accuracy_test

# 上位階層のディレクトリをsys.pathに追加
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + '/../' )
#import posenet
import lineOfSight
import lineOfSight_2
import multiFaceAPI
import faceAPIrequest

from camera_utils.camera import camera_lens_calibration 

def is_unique(seq):
    return len(seq) == len(set(seq))

def l2norm(p0,p1):
    x0, y0 = p0
    x1, y1 = p1
    return math.sqrt(((x0-x1)*(x0-x1))+((y0-y1)*(y0-y1)))

def posenet_io(sess, model_outputs, image, scale_factor, output_stride=16):
    input_image, display_image, output_scale = posenet.read_img(
        image, scale_factor=scale_factor, output_stride=output_stride)

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image})

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=30,
        min_pose_score=0.15)

    keypoint_coords *= output_scale

    # TODO this isn't particularly fast, use GL for drawing and display someday...
    overlay_image, result_keypoints = posenet.draw_skel_and_kp(
        copy.deepcopy(display_image), pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.15, min_part_score=0.1)
    display_image = overlay_image

    return result_keypoints, display_image

def main(Persons,
     FaceAPIimage,
     FaceAPIimage_input_list ,
     FaceAPIimage_output_list,
     FaceAPIdata, 
     FD_Process,
     faces_update,
     cam_id = 0,
     model = 101, 
     cam_width = 1280,
     cam_height = 720,
     scale_factor = 0.36,
     view = False,
     dummy = False,
     disp_size_w = 55,
     disp_size_h = 32,
     disp_resolution_w = 1920,
     disp_resolution_h = 1080,
     detect_dist_min = 0.55,
     detect_dist_max = 3,
     avg_num = 3
     ):
    path = os.getcwd()

    faceFile = path + '/' + '../io_image/face.png'
    facesFile = path + '/'+ '../io_image/faces.png'

    output_data_json = []

    print('ver 3.0.5.faj')

    with tf.Session() as sess:
        ## posenet detector setup
        model_cfg, model_outputs = posenet.load_model(model, sess)
        output_stride = model_cfg['output_stride']
        print(output_stride)

        ## camera setup
        cap = cv2.VideoCapture(cam_id)
        ret_val, image = cap.read()
        # v4l2の設定をsubprocessを用いて実行
        #cmd = 'v4l2-ctl --set-ctrl=white_balance_temperature_auto=0'
        #ret = subprocess.check_output(cmd, shell=True)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) # カメラ画像の横幅を1280に設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128) #明るさ補正 デフォルト128
        cap.set(cv2.CAP_PROP_SATURATION, 128) #彩度補正 デフォルト128
        cap.set(cv2.CAP_PROP_CONTRAST, 128) #コントラスト補正 デフォルト128
        cap.set(cv2.CAP_PROP_GAIN, 128) #ゲイン補正 デフォルト128
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) #
        #cap.set(cv2.CAP_PROP_EXPOSURE, 2)
        ret_val, image = cap.read()

        print("%s, %f" %('FRAME_WIDTH',cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("%s, %f" %('FRAME_HEIGHT',cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("%s, %f" %('BRIGHTNESS',cap.get(cv2.CAP_PROP_BRIGHTNESS)))
        print("%s, %f" %('SATURATION',cap.get(cv2.CAP_PROP_SATURATION)))
        print("%s, %f" %('CONTRAST',cap.get(cv2.CAP_PROP_CONTRAST)))
        print("%s, %f" %('PROP_GAIN',cap.get(cv2.CAP_PROP_GAIN)))

        ## parameter setup
        start = time.time()
        frame_count = 0
        dist_array = [0] * avg_num
        nose_pitch_array = [0] * avg_num
        nose_roll_array = [0] * avg_num
        nose_yaw_array = [0] * avg_num
        nose_pitch_array.append(0)
        nose_roll_array.append(0)
        nose_yaw_array.append(0)

        # カメラ歪み
        mtx_filename = '../camera_utils/camera/cam_calib_mtx.csv'
        dist_filename = '../camera_utils/camera/cam_calib_dist.csv'
        clc_matrix, clc_distortion = camera_lens_calibration.load_calib_data(mtx_filename, dist_filename)
        newcameramtx, _ =cv2.getOptimalNewCameraMatrix(clc_matrix, clc_distortion, (frame_width, frame_height), 1, (frame_width, frame_height))

        # 顔画像処理系
        facearea_margin_factor = 1.1
        face_image_size = 255

        # 角度系
        eye_nose_dist_3d = 1.78 # 両目の間の距離を1.0とした時の鼻の斜面の長さ
        nose_height = 1.41  # 両目の間の距離を1.0とした時の鼻の高さ
        nose_pitch_offsets = 0.11

        pitch_abs = 0.0
        pitch_down_coef = 1.7
        pitch_up_coef = 1.9
        yaw_coef = 1.0
        yaw_coef_abs = 0.8
        eye_w_negc = 0.5
        eye_w_negf = 0.5

        # 距離系
        screen_face_distance_list = np.array([55,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,320,340,360,380,400,500])
        eye_between_len_list_for_distance = np.array([198.13,183.69,153.60,132.58,116.85,98.95,91.05,83.16,77.29,69.67,64.29,60.34,56.76,53.17,49.59,47.49,45.38,43.28,41.17,39.07,37.78,36.49,35.20,33.91,32.62,31.34,29.55,27.76,25.98,24.19,22.40, 10.0])
        ear_between_len_list_for_distance = np.array([449.12,419.77,357.13,301.77,250.70,226.98,205.32,188.12,173.92,162.97,151.35,139.67,132.30,123.90,119.85,114.25,110.11,105.96,101.81,98.19,94.57,90.95,87.33,83.71,80.09,76.47,72.50,68.53,64.56,60.60,56.63, 39.0])
        # トラッキング系
        decision_factor = 0.5 # 当たり判定用係数
        # 描画系
        circle_view = False

        # accuary_test用
        block_n = 2
        block_m = 2
        T = time.time() - start
        H = -1
        K = 0
        G = 0
        Gn = 0
        K_flug = False
        H_flug = False
        K_flug_starttime = 100
        H_flug_starttime = 100
        trials = [0] * (block_m*block_n + 1)
        acc = [0] * (block_m*block_n + 1)
        test_num_list       = [ 2, 3, 3, -1, 0, 3, 2, 1,-1, 2, 1, 3, 0] # テストの順場
        test_time_list      = [ 2, 3, 2, 2, 1, 3, 1, 3, 2, 3, 1, 1, 3]
        test_interval_list  = [ 2, 3, 2, 2, 4, 1, 1, 1, 4, 1, 1, 1, 1]
        eye_sight_pix_x, eye_sight_pix_y = 0,0
        key = ''

        # save result csv
        test_number = accuracy_test.load_test_number()
        save_result_filename = str(test_number).zfill(4) + '.csv'
        save_result_dir = './result_data/'
        save_result_filename = save_result_dir + save_result_filename
        save_result_points_filename = str(test_number).zfill(4) + '_68point.csv'
        save_result_dir = './result_data/'
        save_result_points_filename = save_result_dir + save_result_points_filename
        save_data_array = []
        save_data_array_68point = []

        point_savedata_x = []
        point_savedata_y = []
        y_const = 8
        mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成
        data_count = 0
        eye_yaw =-1

        state_num = 0
        step_num = 20
        skip_flug = False
        
        H_iter = iter(test_num_list)
        time_H_iter = iter(test_time_list)
        interval_H_iter = iter(test_interval_list)
        #H = next(H_iter)
        time_H = test_time_list[0]
        interval_H = test_interval_list[0]

        # data format
        face_bbox_datalist_old = [[[0, 0, image.shape[1], image.shape[0], 0],
                                   [0, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array],    #count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array
                                   [None],  #FaceAPI
                                   ]]
        face_id_num_max = 0

        whiteBoard = np.full((disp_resolution_h,disp_resolution_w,3),255,np.uint8)

        face_data_dict = {}

        key_action = Key_action('','',2,2)

        while True:
            ###########
            # カメラ周り処理
            ret_val, image_raw = cap.read()
            if ret_val != True:
                continue
            image = cv2.undistort(image_raw, clc_matrix, clc_distortion, None, newcameramtx)
            image = cv2.flip(image, 1)
            if dummy:
                image = cv2.imread("./io_image/group-shot.jpg")
            
            ############
            # accuary_test
            block_mtx = np.zeros((block_n, block_m), dtype=np.int)
            acc_mtx = np.zeros((block_n, block_m))
            ############
            # posenet
            result_keypoints, display_image = posenet_io(sess, model_outputs, image, scale_factor, output_stride=16)
            # end posenet
            ############
            
            face_images = np.zeros((face_image_size, face_image_size, 3), np.uint8)
            face_images_list = []
            display_image_height, display_image_width = display_image.shape[:2]
            human_num = 0
            face_bbox_datalist = []
            
            # 顔の領域取得
            for human in result_keypoints:
                # Posenet のデータ処理
                nose_x, nose_y = human[0]
                leftEye_x, leftEye_y = human[1]
                rightEye_x, rightEye_y = human[2]
                leftEar_x, leftEar_y = human[3]
                rightEar_x, rightEar_y = human[4]
                # 顔がおおよそ写っているか
                if nose_x and leftEye_x and rightEye_x:
                    face_bbox_halflen = None
                    if leftEye_x:
                        face_bbox_halflen_l = 2 * abs(leftEye_x - nose_x) * facearea_margin_factor
                    if rightEye_x:
                        face_bbox_halflen_r = 2 * abs(nose_x - rightEye_x) * facearea_margin_factor

                    if face_bbox_halflen_l and face_bbox_halflen_r:
                        face_bbox_halflen = max(face_bbox_halflen_l, face_bbox_halflen_r)
                    elif face_bbox_halflen_l:
                        face_bbox_halflen = face_bbox_halflen_l
                    elif face_bbox_halflen_r:
                        face_bbox_halflen = face_bbox_halflen_r

                    if face_bbox_halflen:
                        if face_bbox_halflen < 30:
                            continue
                        face_bbox_top    = int(nose_y - face_bbox_halflen)
                        face_bbox_bottom = int(nose_y + face_bbox_halflen)
                        face_bbox_left   = int(nose_x - face_bbox_halflen)
                        face_bbox_right  = int(nose_x + face_bbox_halflen)
                        if face_bbox_top < 0:
                            face_bbox_top = 0
                        if face_bbox_bottom > display_image_height:
                            face_bbox_bottom = display_image_height
                        if face_bbox_left < 0:
                            face_bbox_left = 0
                        if face_bbox_right > display_image_width:
                            face_bbox_right = display_image_width
                        
                        #print(face_bbox_top,face_bbox_bottom, face_bbox_left,face_bbox_right)
                        face_orig = display_image[face_bbox_top:face_bbox_bottom, face_bbox_left:face_bbox_right]
                        face_bbox_height = face_bbox_bottom - face_bbox_top
                        face_bbox_width = face_bbox_right - face_bbox_left
                        #print(face.shape)
                        if face_bbox_height < 1 or face_bbox_width < 1:
                            continue

                        nose_x_offsets, nose_y_offsets = nose_x - face_bbox_left, nose_y - face_bbox_top
                        leftEye_x_offsets, leftEye_y_offsets = leftEye_x - face_bbox_left, leftEye_y - face_bbox_top
                        rightEye_x_offsets, rightEye_y_offsets = rightEye_x - face_bbox_left, rightEye_y - face_bbox_top

                        if face_bbox_height < face_bbox_width:
                            r = face_image_size/face_bbox_width
                        else:
                            r = face_image_size/face_bbox_height
                        face = cv2.resize(face_orig, (int(face_bbox_width*r), int(face_bbox_height*r)))
                        nose_x_offsets, nose_y_offsets = int(nose_x_offsets*r), int(nose_y_offsets*r)
                        leftEye_x_offsets, leftEye_y_offsets = int(leftEye_x_offsets*r), int(leftEye_y_offsets*r)
                        rightEye_x_offsets, rightEye_y_offsets = int(rightEye_x_offsets*r), int(rightEye_y_offsets*r)
                        
                        
                        # 回転処理
                        #____ロール方向の顔角度____
                        nose_roll = math.atan2((leftEye_y - rightEye_y),(leftEye_x - rightEye_x))
                        cx, cy = nose_x_offsets, nose_y_offsets
                        trans = cv2.getRotationMatrix2D((cx, cy), math.degrees(nose_roll), 1)
                        face = cv2.warpAffine(face, trans, (face.shape[1],face.shape[0]))
                        roll_matrix = np.array([[math.cos(-nose_roll), -math.sin(-nose_roll), cx - cx*math.cos(-nose_roll) + cy*math.sin(-nose_roll)],
                                                [math.sin(-nose_roll),  math.cos(-nose_roll), cy - cx*math.sin(-nose_roll) - cy*math.cos(-nose_roll)],
                                                [0, 0, 1]])
                        leftEye_x_offsets, leftEye_y_offsets, _ = np.dot(roll_matrix, [leftEye_x_offsets, leftEye_y_offsets, 1])
                        rightEye_x_offsets, rightEye_y_offsets, _ = np.dot(roll_matrix, [rightEye_x_offsets, rightEye_y_offsets, 1])
                        
                        #face = cv2.circle(face, (int(rightEye_x_offsets), int(rightEye_y_offsets)), 3, (0,255,0), -1)
                        #face = cv2.circle(face, (int(leftEye_x_offsets), int(leftEye_y_offsets)), 3, (0,255,0), -1)
                        #face = cv2.circle(face, (int(nose_x_offsets), int(nose_y_offsets)), 3, (0,255,0), -1)
                        #____ヨー方向の顔角度____
                        eye_x_center = (leftEye_x_offsets + rightEye_x_offsets)/2
                        if (rightEye_x_offsets - eye_x_center) == 0:
                            nose_yaw_nomal_dist = 1
                        else:
                            nose_yaw_nomal_dist = - (nose_x_offsets - eye_x_center) / (rightEye_x_offsets - eye_x_center)
                        if -1 < nose_yaw_nomal_dist/nose_height < 1:
                            nose_yaw = math.acos(nose_yaw_nomal_dist/nose_height) - (math.pi/2)
                            #print(math.tan(nose_yaw))
                        else:
                            nose_yaw = None

                        ##____ピッチ方向の顔角度____
                        nose_pitch = None
                        if (rightEye_x_offsets - eye_x_center) != 0:
                            eye_nose_dist_2d = abs((max(leftEye_y_offsets,rightEye_y_offsets) - nose_y_offsets)/(rightEye_x_offsets - eye_x_center)) #x
                            if  abs(eye_nose_dist_2d) < eye_nose_dist_3d:
                                nose_pitch_zero_rad = math.acos(nose_height/eye_nose_dist_3d)
                                nose_edge = eye_nose_dist_2d/eye_nose_dist_3d #sin
                                #print(math.degrees(nose_pitch_zero_rad-nose_pitch_zero_rad2))
                                if nose_edge > 1:
                                    nose_edge = 1
                                nose_pitch_amplitude_rad = math.asin(nose_edge)
                                nose_pitch = nose_pitch_amplitude_rad - nose_pitch_zero_rad + nose_pitch_offsets #+ (math.pi/4)
                                
                                #print(nose_pitch)

                                '''
                                print("_____")
                                print(eye_nose_dist_2d)
                                print(eye_nose_dist_2d-eye_nose_dist_3d)
                                print(-math.degrees(nose_pitch))
                                '''


                        face_image = np.zeros((face_image_size, face_image_size, 3), np.uint8)
                        face_image[0:face.shape[0], 0:face.shape[1]] = face
                        #face_image = cv2.circle(face_image, (int(nose_x_offsets+nose_x_offsets*eye_x_center), int(nose_y_offsets)), 5, (0,0,255), -1)
                        
                        rightShoulder = human[6]
                        rightWrist = human[10]

                        # distance
                        eye_between_pn_len = l2norm((leftEye_x, leftEye_y), (rightEye_x, rightEye_y))
                        if leftEar_x and rightEar_x:
                            ear_between_pn_len = l2norm((leftEar_x, leftEar_y), (rightEar_x, rightEar_y))
                        else:
                            ear_between_pn_len = None

                        
                        #####cv2.imshow('face', face_image)
                        faj = True
                        ############
                        # dlib
                        if not faj:
                            dets, scores, idx = detector.run(face_image, 0)
                        else:
                            cv2.imwrite(faceFile, face_image)
                            dets = None
                        dilb_bbox_size = 0
                        dist = None
                        faceAPI_data = faceAPIrequest.main(dType='single68points', imgFile=faceFile, portNum=3031)#TYNY
                        #faceAPI_d = faceAPIrequest.main(dType='AgeGender', imgFile=facesFile, portNum=3030)

                        if len(faceAPI_data) != 0 and type(faceAPI_data) is dict and 'err' not in faceAPI_data:    #顔を認識できたら認識できた顔の数 以下を実行
                            #______顔角度______
                            #特徴点を格納
                            parts = []
                            parts_float = []
                            if faj:
                                #print(faceAPI_data)
                                if '_positions' in faceAPI_data:
                                    point_data = faceAPI_data['_positions']
                                elif 'landmarks' in faceAPI_data:
                                    point_data = faceAPI_data['landmarks']['_positions']
                                for point in point_data:
                                    points = [int(point['_x']), int(point['_y'])]
                                    parts.append(points)
                                    points_float = [point['_x'], point['_y']]
                                    parts_float.append(points_float)
                                rightEye_x_offsets = (parts[39][0] - parts[36][0])/2 + parts[36][0]
                                rightEye_y_offsets = (parts[39][1] - parts[36][1])/2 + parts[36][1]
                                leftEye_x_offsets = (parts[45][0] - parts[42][0])/2 + parts[42][0]
                                leftEye_y_offsets = (parts[45][1] - parts[42][1])/2 + parts[42][1]

                                data_68point = np.array(parts_float)
                                nomal_data = mmscaler.fit_transform(data_68point)
                                if key == ord('w'):
                                    point_savedata_x.append(nomal_data)
                                    point_savedata_y.append(y_const)
                                    print(data_count)
                                    data_count += 1
                                #print(leftEye_x_offsets,leftEye_y_offsets,rightEye_x_offsets,rightEye_y_offsets)

                            else:
                                d = dets[0]
                                shape = predictor(face, d)
                                leftEye_x_offsets = (shape.part(0).x - shape.part(1).x)/2 + shape.part(1).x
                                leftEye_y_offsets = (shape.part(0).y - shape.part(1).y)/2 + shape.part(1).y
                                rightEye_x_offsets = (shape.part(3).x - shape.part(2).x)/2 + shape.part(2).x
                                rightEye_y_offsets = (shape.part(3).y - shape.part(2).y)/2 + shape.part(2).y
                                #print(leftEye_x_offsets,leftEye_y_offsets,rightEye_x_offsets,rightEye_y_offsets)
                            #nose_x_offsets = shape.part(4).x
                            #eye_between_dl_dist = l2norm((leftEye_x_offsets, leftEye_y_offsets), (rightEye_x_offsets, rightEye_y_offsets))
                            #nose_x_offsets, nose_y_offsets = shape.part(4).x, (shape.part(4).y)*0.83
                            #face_image = cv2.circle(face_image, (int(rightEye_x_offsets), int(rightEye_y_offsets)), 3, (0,0,255), -1)
                            #face_image = cv2.circle(face_image, (int(leftEye_x_offsets), int(leftEye_y_offsets)), 3, (0,0,255), -1)
                            #face_image = cv2.circle(face_image, (int(nose_x_offsets), int(nose_y_offsets)), 3, (0,0,255), -1)
                            #print(shape.part(0).x,shape.part(1).x,shape.part(2).x,shape.part(3).x,shape.part(4).x)
                            #print(rightEye_x_offsets,leftEye_x_offsets,nose_x_offsets)
                            #face_image = cv2.circle(face_image, (shape.part(i).x, shape.part(i).y), 5, (255,180,180), -1)

                            ###### 下にも同じ部分がある修正すべき
                            #____ヨー方向の顔角度____
                            eye_x_center = (leftEye_x_offsets + rightEye_x_offsets)/2
                            if (rightEye_x_offsets - eye_x_center) == 0:
                                nose_yaw_nomal_dist = 1
                            else:
                                nose_yaw_nomal_dist = - (nose_x_offsets - eye_x_center) / (rightEye_x_offsets - eye_x_center)
                            if -1 < nose_yaw_nomal_dist/nose_height < 1:
                                nose_yaw = math.acos(nose_yaw_nomal_dist/nose_height) - (math.pi/2)
                                #print(math.tan(nose_yaw))
                            else:
                                nose_yaw = None
                                
                            ##____ピッチ方向の顔角度____
                            nose_pitch = None
                            if (rightEye_x_offsets - eye_x_center) != 0:
                                eye_nose_dist_2d = abs((max(leftEye_y_offsets,rightEye_y_offsets) - nose_y_offsets)/(rightEye_x_offsets - eye_x_center)) #x                                
                                if abs(eye_nose_dist_2d) < eye_nose_dist_3d:
                                    #print(eye_nose_dist_2d)
                                    # キャリブレーション
                                    #nose_height = 1.41  # 両目の間の距離を1.0とした時の鼻の高さ
                                    #eye_nose_dist_3d = math.sqrt((eye_nose_dist_2d*eye_nose_dist_2d)+nose_height*nose_height)
                                    #print(eye_nose_dist_3d)
                                    # -> 1.45
                                    #eye_nose_dist_2d = eye_nose_dist_3d*math.cos()
                                    #eye_nose_dist_3d = 1.78 #y
                                    #nose_pitch_offsets = 0.0
                                    nose_pitch_zero_rad = math.acos(nose_height/eye_nose_dist_3d)
                                    nose_edge = eye_nose_dist_2d/eye_nose_dist_3d 
                                    if nose_edge > 1:
                                        nose_edge = 1
                                    #nose_pitch_amplitude_rad = math.asin(eye_nose_dist_2d/eye_nose_dist_3d)
                                    nose_pitch_amplitude_rad = math.asin(nose_edge)
                                    nose_pitch = nose_pitch_amplitude_rad - nose_pitch_zero_rad + nose_pitch_offsets #+ (math.pi/4)
                                    '''
                                    print("_____")
                                    print(eye_nose_dist_2d)
                                    print(eye_nose_dist_2d-eye_nose_dist_3d)
                                    print(-math.degrees(nose_pitch))
                                    '''

                            # 視線を検知
                            if faj:
                                #ret, lOSmap, pA = lineOfSight.lOS_main(detector, predictor68, face_image, 0, 500, 1.0, 1.0)
                                ret, lOSmap = lineOfSight_2.lOS_main(parts, face_image, 0, 500, 1.0, 1.0)
                                #i = 0
                                #print(pA)
                                #if len(pA) != 0:
                                #    for p in parts:
                                #        print(parts[i][0],pA[i][0],parts[i][1],pA[i][1])
                                #        i += 1

                            lOS = (lOSmap[0] - 0.5) * 1.7
                            eye_yaw = math.radians(31*lOS)

                            
                            # 距離の測定(dlib)
                            """
                            w = d.right() - d.left()
                            h = d.bottom() - d.top()
                            dilb_bbox_size = math.sqrt((w * w) + (h * h))
                            X = np.array([289.5, 169.8, 115.5, 69.3])
                            y = np.array([1,2,3,5])
                            cubic = interp1d(X, y, kind="cubic")
                            if dilb_bbox_size < min(X):
                                print("over 5m")
                            elif dilb_bbox_size > max(X):
                                    print("under 1m")
                            elif nose_yaw and nose_pitch:
                                dist = float(cubic(dilb_bbox_size))
                                nose_yaw_dist = 1 * math.tan(nose_yaw)
                                nose_pitch_dist = 1 * math.tan(nose_pitch)
                                #print(nose_yaw_dist)
                                #print(dilb_bbox_size, cubic(dilb_bbox_size), nose_yaw_dist)
                            """
                        else:
                            parts = None
                            dilb_bbox_size = None
                            eye_yaw = None

                        # posenetで距離の測定
                        """
                        # 手を上げたら測量開始
                        if rightShoulder[1] and rightWrist[1]:
                            if rightShoulder[1] > rightWrist[1]:
                                #print(eye_between_pn_len,ear_between_pn_len,dilb_bbox_size)
                        if rightShoulder[1] and human[8][1]:
                            if rightShoulder[1] > human[8][1]:
                                #print(eye_between_pn_len,ear_between_pn_len,dilb_bbox_size)
                        """
                        if nose_yaw:

                            Y_list = screen_face_distance_list

                            face_len = eye_between_pn_len
                            X_list = eye_between_len_list_for_distance
                            if ear_between_pn_len:
                                face_len = ear_between_pn_len
                                X_list = ear_between_len_list_for_distance
                            
                            face_len = face_len/math.cos(nose_yaw)
                            cubic = interp1d(X_list, Y_list, kind="cubic")
                            if face_len < min(X_list):
                                print("over distance")
                                dist = None
                                #continue
                            elif face_len > max(X_list):
                                print("under 0.5m",math.degrees(nose_yaw))
                                dist = None
                                #continue
                            elif nose_yaw and nose_pitch:
                                dist = float(cubic(face_len))
                                if dist < detect_dist_min*100 or dist > detect_dist_max*100:
                                    dist = None
                        #print(dist)
                        if nose_pitch and nose_yaw:
                            if nose_pitch > 0:
                                nose_pitch_cash = nose_pitch * pitch_down_coef * (1 + abs(math.sin(nose_yaw)))
                                nose_yaw_cash = nose_yaw * yaw_coef * (1 + abs(math.sin(nose_pitch)))
                                nose_pitch = nose_pitch_cash
                                nose_yaw = nose_yaw_cash
                            else:
                                nose_pitch = nose_pitch * pitch_up_coef
                        face_images_list.append(face_image)

                        if eye_yaw and nose_yaw:
                            nose_yaw = nose_yaw - eye_yaw
                        if nose_yaw:
                            nose_yaw = nose_yaw * yaw_coef_abs

                        count = 0
                        face_bbox_datalist.append([[face_bbox_left, face_bbox_top, face_bbox_right, face_bbox_bottom], [count, dist, nose_pitch, nose_roll, nose_yaw], []])

                human_num += 1
            # 重なっているか判定


            face_id_num_array = []
            face_id_num_array_cash = []
            face_bbox_face_id_cash = []
            face_bbox_datalist_old_cash = []
            face_bbox_piles_list = []
            face_bbox_num = 0
            bbox_old_state = []
            
            for bbox in face_bbox_datalist:
                decision = False
                piles = None
                x0, y0, x1, y1 = bbox[0]
                count, dist, nose_pitch, nose_roll, nose_yaw = bbox[1]
                for bbox_old in face_bbox_datalist_old:
                    # 当たり判定 重なったらdecision=true
                    xo0, yo0, xo1, yo1, face_id_num = bbox_old[0]
                    xc, yc = (x0 + x1)//2, (y0 + y1)//2
                    w_d, h_d = x1 - x0, y1 - y0
                    xoc, yoc = (xo0 + xo1)//2, (yo0 + yo1)//2
                    wo_d, ho_d = xo1 - xo0, yo1 - yo0
                    xc_d, yc_d = abs(xc - xoc), abs(yc - yoc)
                    wh_d, hh_d = (w_d + wo_d)/2, (h_d + ho_d)/2
                    if xc_d < wh_d * decision_factor and yc_d < hh_d * decision_factor: # あたり
                        decision = True
                        bbox_old_state = bbox_old[1]
                        count = bbox_old[1][0]
                        if piles is None:   # 重複がない場合
                            face_id_num_array_cash.append(face_id_num)
                        else:
                            face_bbox_piles_list.append(face_id_num)
                        piles = face_id_num
                    if face_id_num > face_id_num_max:
                        face_id_num_max = face_id_num

                if decision: # 重なっていたらデータ継承
                    count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array = bbox_old_state
                    count += 1
                    #print(dist_array)
                    if dist_array is not None:
                        #print(len(dist_array))
                        if len(dist_array) > avg_num:
                            dist_array = dist_array[0:avg_num]
                        if dist is not None:
                            dist_array.append(dist)
                            dist_array.pop(0)
                    if nose_pitch_array is not None:
                        if len(nose_pitch_array) > avg_num:
                            nose_pitch_array = nose_pitch_array[0:avg_num]
                        if nose_pitch is not None:
                            nose_pitch_array.append(nose_pitch)
                            nose_pitch_array.pop(0)
                    if nose_roll_array is not None:
                        if len(nose_roll_array) > avg_num:
                            nose_roll_array = nose_roll_array[0:avg_num]
                        if nose_roll is not None:
                            nose_roll_array.append(nose_roll)
                            nose_roll_array.pop(0)
                    if nose_yaw_array is not None:
                        if len(nose_yaw_array) > avg_num:
                            nose_yaw_array = nose_yaw_array[0:avg_num]
                        if nose_yaw is not None:
                            nose_yaw_array.append(nose_yaw)
                            nose_yaw_array.pop(0)
                    
                    if len(face_id_num_array_cash) < face_bbox_num: #同じ枠内に重なったパターン
                        #face_id_num_max += 1
                        face_id_num_array_cash.append(face_id_num_max + 1)
                    if is_unique(face_id_num_array_cash) is False:
                        face_id_num_max += 1
                        #print(face_id_num_array_cash)
                        face_id_num_array_cash[-1] = face_id_num_max
                        #print(face_id_num_array_cash)
                        #face_data_dict[]
                    piles_id = None
                    n = 0
                    
                        #if face_id_num == piles_id:
                        #    face_id_num_max += 1
                        #    face_id_num_array_cash[n] = face_id_num_max
                        #piles_id = face_id_num

                    #for piles_id in face_bbox_piles_list:

                    #print(face_bbox_num)
                    #print(face_id_num_array_cash)
                    #print(face_id_num_array_cash[face_bbox_num])

                    face_bbox_face_id_cash.append(face_id_num_array_cash[face_bbox_num])
                    face_bbox_datalist_old_cash.append([[x0, y0, x1, y1, face_id_num_array_cash[face_bbox_num]], [count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array], []])


                    face_data_dict[face_id_num_array_cash[face_bbox_num]] = [[x0, y0, x1, y1, face_id_num_array_cash[face_bbox_num]], [count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array], []]
                    #print(face_data_dict)
                    face_bbox_num += 1

                else: # 重なっていなければ新規IDを振る
                    face_bbox_face_id_cash.append(face_id_num_max + 1)
                    dist_array = [0] * avg_num
                    nose_pitch_array = [0] * avg_num
                    nose_roll_array = [0] * avg_num
                    nose_yaw_array = [0] * avg_num
                    if dist:
                        dist_array.append(dist)
                    if nose_pitch:
                        nose_pitch_array.append(nose_pitch)
                    if nose_roll:
                        nose_roll_array.append(nose_roll)
                    if nose_yaw:
                        nose_yaw_array.append(nose_yaw)

                    face_id_num_max += 1
                    face_bbox_datalist_old_cash.append([[x0, y0, x1, y1, face_id_num_max], [0, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array], []])
                    face_data_dict[face_id_num_max] = [[x0, y0, x1, y1, face_id_num_max], [0, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array], []]
            #print(face_id_num_array_cash)
            for face_id_num in face_id_num_array_cash:
                for m in range(n+1, len(face_id_num_array_cash)):
                    face_id_num2 = face_id_num_array_cash[m]
                    if face_id_num == face_id_num2:
                        face_id_num_max += 1
                        face_id_num_array_cash[m] = face_id_num_max
                        face_data_dict[m][0][4] = face_id_num_max
                n += 1
            #print(face_id_num_array_cash)

            face_id_num_array = face_id_num_array_cash
            #print()
            face_bbox_datalist_old = face_bbox_datalist_old_cash
            #print(face_id_num_array)
            #print(face_bbox_piles_list)
            #print(face_bbox_datalist_old)


            # face dictの作成と headPoseの平均
            faceAPIimage_input_list = []
            face_dict = {}
            if len(face_images_list) != 0:
                """
                for face_id, face_image in zip(face_id_num_array, face_images_list):
                    face_image = cv2.putText(face_image, 'id:' + str(face_id), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                    face_dict.append([face_id, face_image])
                face_dict = sorted(face_dict)
                """
                face_id_index_array = np.argsort(face_id_num_array)
                for face_id_index in face_id_index_array:
                    face_id = face_id_num_array[face_id_index]
                    #print(face_id, face_data_dict)
                    if face_id in face_data_dict:
                        data = face_data_dict[face_id]
                        count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array = data[1]
                    face_image = face_images_list[face_id_index]
                    #face_image = cv2.putText(face_image, 'id:' + str(face_id), (10, int(face_image_size*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
                    '''
                    dist = sum(dist_array)/len(dist_array)
                    nose_roll = sum(nose_roll_array)/len(nose_roll_array)
                    nose_yaw = sum(nose_yaw_array)/len(nose_yaw_array)
                    nose_pitch = sum(nose_pitch_array)/len(nose_pitch_array)
                    dist = dist/100
                    nose_yaw_dist = dist * math.tan(nose_yaw)
                    nose_pitch_dist = dist * math.tan(nose_pitch)
                    if dist_array.count(0) != avg_num:
                        dist = sum(dist_array)/(len(dist_array) - dist_array.count(0))
                        dist = dist/100
                    else:
                        dist = None
                    if nose_roll_array.count(0) != avg_num:
                        nose_roll = sum(nose_roll_array)/(len(nose_roll_array) - nose_roll_array.count(0))
                    else:
                        nose_roll = None

                    if nose_yaw_array.count(0) != avg_num and dist:
                        nose_yaw = sum(nose_yaw_array)/(len(nose_yaw_array) - nose_yaw_array.count(0))
                        nose_yaw_dist = dist * math.tan(nose_yaw)
                        eye_w = (1/2)*(1-(nose_yaw_dist/(disp_size_w/100)))
                    else:
                        nose_yaw = None
                        eye_w = None

                    if nose_pitch_array.count(0) != avg_num and dist:
                        nose_pitch = sum(nose_pitch_array)/(len(nose_pitch_array) - nose_pitch_array.count(0))
                        nose_pitch_dist = dist * math.tan(nose_pitch)
                        eye_h = (1/2)*(1+(nose_pitch_dist/(disp_size_h/100)))
                    else:
                        nose_pitch = None
                        eye_h = None
                    '''
                    face_dict[face_id] = face_image
                    faceAPIimage_input_list.append(face_id)
                #face_dict = sorted(face_dict)



                human_num = 0
                for face_id in face_dict:
                    face_image = face_dict[face_id]
                    if human_num == 0:
                        face_images = face_image
                    else:
                        face_images = cv2.hconcat([face_images, face_image])
                    human_num += 1

                
            # face_data_dict から 非アクティブなデータを消去
            face_data_dict_new = {}
            active_id_list = face_dict.keys()
            for key in face_id_num_array:
                if key in active_id_list:
                    face_data_dict_new[key] = face_data_dict[key]
            face_data_dict = face_data_dict_new
            #print(face_data_dict)

                
            
            '''
            input_image, display_image, output_scale = posenet.read_img(
                face_images, scale_factor=scale_factor, output_stride=16)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=16,
                max_pose_detections=30,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image, result_keypoints = posenet.draw_skel_and_kp(
                copy.deepcopy(display_image), pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            '''



            #if frame_count == 1000:
            #    break
            faceAPIdata_dict = {}
            FaceAPIimage_input_list[0] = faceAPIimage_input_list
            if faceAPIimage_input_list:
                FaceAPIimage[0] = face_images
                i = 0
                if "human" in FaceAPIdata:
                    FaceAPIdata_humans = FaceAPIdata["human"]
                    if type(FaceAPIdata_humans) is dict:
                        #print(FaceAPIdata_humans)
                        #print(FaceAPIimage_input_list)
                        for face_id in FaceAPIimage_output_list[0]:
                            if face_id in face_data_dict:
                                FaceAPIdata_humans_idlist = list(FaceAPIdata_humans.keys())
                                list(map(int, FaceAPIdata_humans_idlist))
                                if i in FaceAPIdata_humans_idlist:
                                    face_data_dict[face_id][2] = FaceAPIdata_humans[i]
                                    faceAPIdata_dict[face_id] = FaceAPIdata_humans[i]
                                    #print(FaceAPIdata_humans[i])
                            i += 1
            #print(FaceAPIimage_input_list,FaceAPIimage_output_list)
            """
            #face APIに渡すものが画像データかどうか確認
            if len(np.array(FaceAPIimage[0]).shape) == 3:
                cv2.imwrite("./io_image/test00.png",FaceAPIimage[0])
                FaceAPIimage_output_list = FaceAPIimage_input_list
                FaceAPIdata_humans = multiFaceAPI.faceAPI(FaceAPIimage[0])  #face APIに送信
                #emotion_dict(faceAPIの返り値)が正しいか確認
                #if type(emotion_dict) is dict:
                #    emo1 = max(emotion_dict.items(), key=lambda x:x[1])[0]
                #    emo1val = emotion_dict.pop(emo1)
                #    FaceAPIdata_humans[0] = emo1
                #    if emo1 == 'neutral':
                #        emo2 = max(emotion_dict.items(), key=lambda x:x[1])[0]
                #        emo2val = emotion_dict.pop(emo2)
                #        if emo2val * emotion_coef > emo1val:
                #            FaceAPIdata_humans[0] = emo2
                #else:
                #    FaceAPIdata_humans[0] = 'error faceAPI_pose-detector' 

            else:
                FaceAPIdata_humans = ''
            """

            Persons_list = []
            for face_id in face_data_dict:
                Status = {}

                data = face_data_dict[face_id]
                count, dist_array, nose_pitch_array, nose_roll_array, nose_yaw_array = data[1]
                if dist_array.count(0) != avg_num:
                    dist = sum(dist_array)/(len(dist_array) - dist_array.count(0))
                    dist = dist/100
                else:
                    dist = None

                if nose_roll_array.count(0) != avg_num:
                    nose_roll = sum(nose_roll_array)/(len(nose_roll_array) - nose_roll_array.count(0))
                else:
                    nose_roll = None

                if nose_yaw_array.count(0) != avg_num and dist:
                    nose_yaw = sum(nose_yaw_array)/(len(nose_yaw_array) - nose_yaw_array.count(0))
                    nose_yaw_dist = dist * math.tan(nose_yaw)
                    eye_w = (1/2)*(1-(nose_yaw_dist/(disp_size_w/100)))
                    eye_w_org = eye_w
                    eye_w = -2.0 * (eye_w - 0.5)
                    if eye_w > eye_w_negf:
                        eye_w = (eye_w - eye_w_negf)*eye_w_negc + eye_w_negf
                    if eye_w < -eye_w_negf:
                        eye_w = (eye_w + eye_w_negf)*eye_w_negc - eye_w_negf
                    eye_w_org = (eye_w/(-2.0))+0.5
                else:
                    nose_yaw = None
                    eye_w = None

                if nose_pitch_array.count(0) != avg_num and dist:
                    nose_pitch = sum(nose_pitch_array)/(len(nose_pitch_array) - nose_pitch_array.count(0))
                    nose_pitch_dist = dist * math.tan(nose_pitch)
                    eye_h = (1/2)*(1+(nose_pitch_dist/(disp_size_h/100)))
                    eye_h_org = eye_h
                    eye_h = -2.0 * (eye_h - 0.5)
                else:
                    nose_pitch = None
                    eye_h = None



                
                result = str(datetime.datetime.today())
                Status["ID"] = face_id
                Status["time"]   = result
                Status["sequence"]   = count%1024
                Status["distance"]   = dist
                Status["eye"] = "%s, %s" %(str(eye_w), str(eye_h))
                #Status["eye"] = "%s, %s" %(str(1.0), str(-1.0))
                Status["gender"] = "none"
                Status["age"] = "none"
                Status["area"] = -1
                if face_id in faceAPIdata_dict:
                    if faceAPIdata_dict[face_id]:
                        #print(faceAPIdata_dict[face_id]['faceAttributes']["gender"])
                        Status["gender"] = faceAPIdata_dict[face_id]['faceAttributes']["gender"]
                        Status["age"] = faceAPIdata_dict[face_id]['faceAttributes']["age"]
                #Status["face"] = Emotion[0]
                #Status["action"] = ','.join(map(str, action))
                if eye_w and eye_h:
                    eyesight_x = int(disp_resolution_w*eye_w_org)
                    eyesight_y = int(disp_resolution_h*eye_h_org)
                    whiteBoard = accuracy_test.number2image(whiteBoard
                    , block_n, block_m, H)
                    if circle_view:
                        whiteBoard = cv2.circle(whiteBoard, (eyesight_x, eyesight_y), disp_resolution_h//10, (0,0,255), -1)
                    areaN = accuracy_test.area2number(whiteBoard, block_n, block_m, [eyesight_x, eyesight_y])
                    Status["area"] = areaN
                    eye_sight_pix_x = int(disp_resolution_w*eye_w_org)
                    eye_sight_pix_y = int(disp_resolution_h*eye_h_org)                   
                    #print(int(disp_resolution_w*eye_w_org), int(disp_resolution_h*eye_h_org))
                areaN = Status["area"]
                if not (areaN < 0):
                    block_mtx[areaN//block_m,areaN%block_m] = 1
                else:
                    block_mtx[areaN//block_m,areaN%block_m] = -1

                    Persons_list.append(Status)
            Persons["Persons"] = Persons_list
            #print()
            ###cv2.imshow('face', face_images)
            cv2.imwrite(facesFile, face_images)
            #cv2.imwrite('./io_image/display_image.png', display_image)
            if view:
                #whiteBoard = cv2.resize(whiteBoard,(1920,1080))
                if len(face_bbox_datalist) > 2:
                    whiteBoard = np.full((disp_resolution_h,disp_resolution_w,3),0,np.uint8)
                cv2.imshow('posenet', display_image)
                cv2.imshow('faces', face_images)
                cv2.imshow('view', whiteBoard)
                whiteBoard = np.full((disp_resolution_h,disp_resolution_w,3),255,np.uint8)
            #print(result_keypoints)
            frame_count += 1

            ### key_action
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                K = 255
            else:
                K = chr(key)
            
            if K == 's':
                if not K_flug:
                    K_flug_starttime = T
                    K_flug = True
                if K_flug:
                    if T - K_flug_starttime > time_H:
                        H_flug = True
                        H_flug_starttime = T
                        K_flug = False
                    else:
                        K_flug = True
            else:
                K_flug = False

            if H_flug:
                ## white
                H = -2
                if T - H_flug_starttime > interval_H:
                    #H = accuracy_test.rand(block_n*block_m, H)
                    try:
                        H = next(H_iter)
                        time_H = next(time_H_iter)
                        interval_H = next(interval_H_iter)

                    except StopIteration:
                        break
                    H_flug = False
                else:
                    H_flug = True

            if np.sum(block_mtx) == 1:
                np.reshape(block_mtx,(block_n*block_m,))
                G = np.argmax(block_mtx)
            elif np.min(block_mtx) == -1:
                G = -1
            else:
                G = -2
            
            
            #acc_data.append([T, H, K, G])
            T = time.time() - start
            if (type(K) == str) and (H != -2):
                trials[H] += 1
                if H == G:
                    acc[H] += 1
                    pass
            for i in range(len(trials)-1):
                if 0 < trials[i]:
                    #print((acc[i]/trials[i]),', ',end='')
                    acc_mtx[i//block_m,i%block_m] = acc[i]/trials[i]
                else:
                    #print(None,', ',end='')
                    acc_mtx[i//block_m,i%block_m] = -1
            print()
            #if len(Persons['Persons']) == 0:
            if len(face_bbox_datalist) == 1:
                output_data = [datetime.datetime.today(), T, H, K, G, eye_sight_pix_x, eye_sight_pix_y, len(face_bbox_datalist), eye_yaw] # T:時間, H:見てもらいたいエリア, K:押されているキー, 視点エリア
            else:
                output_data = [datetime.datetime.today(), T,-3,-3,-3,-3,-3, len(face_bbox_datalist), eye_yaw]
            print(output_data, len(face_bbox_datalist), save_result_filename, eye_sight_pix_x, eye_sight_pix_y)
            output_data_json.append(Status)
            save_data_array.append(output_data)
            #save_data_array_68point.append(parts)
            if key == ord('q'):
                break
            if data_count > 99:
                break
            if K == 'c':
                circle_view = not circle_view
            key_action.determine_state_active_wait()
            #print(key_action.nowa_time, key_action.state_time, key_action.nowa_time_before)

            # white bord make


            #print("____")
            #print(faceAPIimage_input_list)
            #print(json.dumps(Persons, indent=2))
            #if frame_count > 200:
            #    break
            if 'Process' in FD_Process:
                if FD_Process['Process'] == False:
                    break
        np.save(save_result_filename,output_data_json)
        #np.savetxt(save_result_filename, save_data_array, delimiter=",",fmt="%s")
        #np.savetxt(save_result_points_filename, save_data_array_68point, delimiter=",",fmt="%s")
        np.savez('./'+str(y_const), x=point_savedata_x, y=point_savedata_y)
        print('stop faceDirection','Average FPS: ', frame_count / (time.time() - start))
    sess.close()



class Key_action:
    def __init__(self, key, state, m, n):
        self.key = key
        self.state = state
        self.next_state = ''
        self.nowa_state = ''
        self.state_time = 0.0
        self.nowa_time = time.time()
        self.nowa_time_before = time.time()
        self.m = m
        self.n = n
        self.stay_time = 3.0
        self.trigger = False
        self.active_wait_flag = False
        self.passive_wait_flag = False

    def key2state_phrase(self, key):
        if key == 's':
            state_phrase = 's_key'
    
    def forced_update_state(self, state):
        self.state = state

    def setup_stay_time(self, stay_time):
        self.stay_time = stay_time

    def restart_time(self):
        self.nowa_time = time.time()
        self.nowa_time_before = time.time()
        self.state_time = self.nowa_time - self.nowa_time_before

    def update_time(self):
        self.nowa_time = time.time()
        self.state_time = self.nowa_time - self.nowa_time_before

    def new_state(self, key, state):
        self.key = key
        self.state = state
        self.nowa_time_before = self.nowa_time
        self.state_time = 0.0
    
    def determine_state_active_wait(self):
        '''
        トリガーが一度でもTrueになったら以降何もしないでもstate_timeが動きます
        stay_timeまで達したらTrueを返します
        '''
        if self.trigger and self.active_wait_flag is False:
            self.active_wait_flag = True
            Key_action.restart_time(self)

        if self.active_wait_flag:
            Key_action.update_time(self)

        if self.active_wait_flag and (self.state_time > self.stay_time):
            Key_action.new_state(self, 'b', '')
            #time.sleep(3)
            self.active_wait_flag = False
            self.trigger = False

    def determine_state_passive_wait(self):
        '''
        トリガーがTrueであり続けている間state_timeが動きます
        トリガーがFalseになった瞬間state_timeはリセットされます
        stay_timeまで達したらTrueを返します
        '''
        if self.trigger and self.passive_wait_flag is False:
            self.passive_wait_flag = True
            Key_action.restart_time(self)
        elif not self.trigger:
            self.passive_wait_flag = False
        
        if self.trigger and self.passive_wait_flag:
            Key_action.update_time(self)
            if self.state_time > self.stay_time:
                Key_action.new_state(self, 'b', '')
                #print('aaa')
                self.trigger = False
                #time.sleep(3)
        else:
            Key_action.restart_time(self)
            self.trigger = False
    
        



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=1280)
    parser.add_argument('--cam_height', type=int, default=720)
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--view', type=bool, default=False)
    parser.add_argument('--dummy', type=bool, default=False)
    parser.add_argument('--disp_size_w', type=int, default=55, help="unit cm. default 55")
    parser.add_argument('--disp_size_h', type=int, default=32, help="unit cm. default 32")
    parser.add_argument('--disp_resolution_w', type=int, default=1920, help="unit pix. default 1920")
    parser.add_argument('--disp_resolution_h', type=int, default=1080, help="unit pix. default 1080")
    parser.add_argument('--detect_dist_min', type=float, default=0.55, help="unit m. default 0.55")
    parser.add_argument('--detect_dist_max', type=float, default=3, help="unit m. default 3")
    parser.add_argument('-n', type=int, default=3, help="Average number of eyetracking data. default 3")
    args = parser.parse_args()
    model = args.model
    cam_id = args.cam_id
    cam_width = args.cam_width
    cam_height = args.cam_height
    scale_factor = args.scale_factor
    view = args.view
    dummy = args.dummy

    disp_size_w = args.disp_size_w
    disp_size_h = args.disp_size_h
    disp_resolution_w = args.disp_resolution_w
    disp_resolution_h = args.disp_resolution_h
    detect_dist_min = args.detect_dist_min
    detect_dist_max = args.detect_dist_max
    avg_num = args.n

    Persons = {}
    FD_Process = {}
    FaceAPIimage = [[]]
    FaceAPIdata = [[]]
    FaceAPIimage_input_list = [[]]
    FaceAPIimage_output_list = [[]]
    FD_Process['Process'] = True
    faces_update = False

    main(Persons,
     FaceAPIimage,
     FaceAPIimage_input_list,
     FaceAPIimage_output_list,
     FaceAPIdata, 
     FD_Process,
     faces_update,
     cam_id,
     model,
     cam_width,
     cam_height,
     scale_factor,
     view,
     dummy,
     disp_size_w,
     disp_size_h,
     disp_resolution_w,
     disp_resolution_h,
     detect_dist_min,
     detect_dist_max,
     avg_num
     )
