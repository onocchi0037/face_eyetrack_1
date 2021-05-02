# -*- coding: utf-8 -*-
import socket
import threading
import json
import argparse
import time
import dlib
import cv2

import faceDirection_faceApiJs
import faceAPIrequest
import faceAPIutil

import os

import random
import numpy as np




HOST = '127.0.0.1'
PORT = 50007

Persons = {} # 結果保存用
FaceAPIimage = [[]]
FaceAPIimage_input_list = [[]]
FaceAPIimage_output_list = [[]]
FD_Process = {}
faces_update = {}
FaceAPIdata = {}


# 測定実行用スレッドのクラス
#視線姿勢
class MyThread(threading.Thread, ):
    def __init__(self,
                Persons,
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
                scale_factor = 1.0,
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
        super(MyThread, self).__init__()
        self.setDaemon(True)

    def run(self):
        faceDirection_faceApiJs.main(Persons,
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
                False,
                dummy,
                disp_size_w,
                disp_size_h,
                disp_resolution_w,
                disp_resolution_h,
                detect_dist_min,
                detect_dist_max,
                avg_num
                )

    def stop(self):
        """スレッドを停止させる"""
        self.stop_event.set()
        self.thread.join()    #スレッドが停止するのを待つ


#FaceAPI
class MyThread2(threading.Thread, ):
    def __init__(self,
                 FaceAPIimage,
                 FaceAPIimage_input_list,
                 FaceAPIimage_output_list,
                 FaceAPIdata,
                 FD_Process,
                 faces_update,
                 faceAPI_SPF = 0.0
                 ):
        super(MyThread2, self).__init__()
        self.setDaemon(True)
        self.stop_event = threading.Event() #停止させるかのフラグ

    def run(self):
        spf = 0.0
        spf_time = 0.0
        faceAPI_count = 0
        frame_count = 0
        start = time.time()

        path = os.getcwd()
        facesFile = path+'/io_image/faces.png'

        while True:
            #print(faces_update['flug'])
            #print(facesFile)
            #face APIに渡すものが画像データかどうか確認
            if len(np.array(FaceAPIimage[0]).shape) == 3 and faces_update['flug'] is True:
                faces_image = np.array(FaceAPIimage[0])
                h, w = faces_image.shape[:2]
                human_num = w//h
                #cv2.imwrite("./io_image/test00.png",FaceAPIimage[0])
                #face_images = cv2.imread(facesFile)
                
                if human_num == 1:
                    black_space = np.zeros((faces_image.shape), np.uint8)
                    faces_image = cv2.hconcat([faces_image, black_space])
                    cv2.imwrite("./io_image/faces.png", faces_image)
                else:
                    cv2.imwrite("./io_image/faces.png", faces_image)

                #time.sleep(1)
                FaceAPIimage_output_list[0] = FaceAPIimage_input_list[0]
                #print(FaceAPIimage_output_list[0])
                #FaceAPIdata["human"] = faceAPIrequest.main(dType='AgeGender', imgFile=facesFile, portNum=3030)
                data = faceAPIrequest.main(dType='AgeGender', imgFile=facesFile, portNum=3030)
                faces_update['flug'] = False
                FaceAPIdata["human"] = faceAPIutil.json_adjust(data, human_num, h)
                #FaceAPIdata["human"] = None  #face APIに送信
                #FaceAPIdata["human"] = multiFaceAPI.faceAPI(FaceAPIimage[0])  #face APIに送信
                frame_count += 1
                #emotion_dict(faceAPIの返り値)が正しいか確認
                #if type(emotion_dict) is dict:
                #    emo1 = max(emotion_dict.items(), key=lambda x:x[1])[0]
                #    emo1val = emotion_dict.pop(emo1)
                #    FaceAPIdata[0] = emo1
                #    if emo1 == 'neutral':
                #        emo2 = max(emotion_dict.items(), key=lambda x:x[1])[0]
                #        emo2val = emotion_dict.pop(emo2)
                #        if emo2val * emotion_coef > emo1val:
                #            FaceAPIdata[0] = emo2
                #else:
                #    FaceAPIdata[0] = 'error faceAPI_pose-detector' 
            if 'Process' in FD_Process:
                if FD_Process['Process'] == False:
                    print('stop FaceAPI ','Average FPS: ', frame_count / (time.time() - start))
                    break

    def stop(self):
        """スレッドを停止させる"""
        self.stop_event.set()
        self.thread.join()    #スレッドが停止するのを待つ

            

                

 
# サーバを作成して動かす関数
def socket_work():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        print('Connected by', addr)
        data = conn.recv(2048).decode()
        if not data:
            break
        if ("get pose" in data):
            conn.send(json.dumps(Persons).encode())
        elif("calibration:" in data):
            conn.send("done".encode())
        else:
            conn.send("error".encode())
        conn.close()
    
    return False
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='digi-sig-demo')
    
    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=1280)
    parser.add_argument('--cam_height', type=int, default=720)
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--view', type=bool, default=False)
    parser.add_argument('--dummy', type=bool, default=False, help='load dummy data')
    parser.add_argument('--disp_size_w', type=int, default=55, help="unit cm. default 55")
    parser.add_argument('--disp_size_h', type=int, default=32, help="unit cm. default 32")
    parser.add_argument('--disp_resolution_w', type=int, default=1920, help="unit pix. default 1920")
    parser.add_argument('--disp_resolution_h', type=int, default=1080, help="unit pix. default 1080")
    parser.add_argument('--detect_dist_min', type=float, default=0.55, help="unit m. default 0.55")
    parser.add_argument('--detect_dist_max', type=float, default=3.0, help="unit m. default 3.0")
    parser.add_argument('-n', type=int, default=3, help="Average number of eyetracking data. default 3")
    parser.add_argument('--noexec-faceAPI', type=bool, default = False, help='noexec-faceAPI. type bool')
    parser.add_argument('--set-faceAPI-SPF', type=float, default = 0.0, help='set second per faceAPI. type float')

    parser.add_argument('--emotion-coef', type=float, default=20,
                        help='default=20. multiply the second evaluated expression by -e. And it will be compare the first evaluated again.')

    parser.add_argument('--save-inptImg', type=bool, default = False, help='to save overwrite webcam images(debug.png). it helps to solve bugs.')

    args = parser.parse_args()
    model = args.model
    cam_id = args.cam_id
    cam_width = args.cam_width
    cam_height = args.cam_height
    scale_factor = args.scale_factor
    view = args.view
    dummy = args.dummy
    save_inptImg = args.save_inptImg
    
    disp_size_w = args.disp_size_w
    disp_size_h = args.disp_size_h
    disp_resolution_w = args.disp_resolution_w
    disp_resolution_h = args.disp_resolution_h
    detect_dist_min = args.detect_dist_min
    detect_dist_max = args.detect_dist_max
    avg_num = args.n
    # face API args
    noexec_faceAPI = args.noexec_faceAPI
    faceAPI_SPF = args.set_faceAPI_SPF

    args = parser.parse_args()

    #main(cam_id, model, resize, resize_out_ratio)
    view = False,
    # スレッドの作成と開始
    FD_Process['Process'] = True
    faces_update['flug'] = True
    mythread1 = MyThread(Persons,
                        FaceAPIimage,
                        FaceAPIimage_input_list ,
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
    mythread1.start()
    if noexec_faceAPI == False:
        mythread2 = MyThread2(FaceAPIimage,
                                FaceAPIimage_input_list,
                                FaceAPIimage_output_list,
                                FaceAPIdata,
                                FD_Process,
                                faces_update,
                                faceAPI_SPF)
        mythread2.start()

    # サーバを作成して動かす
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)

        while True:
            conn, addr = s.accept()
            print('Connected by', addr)
            data = conn.recv(2048).decode()
            if not data:
                print('not data')
                break
            if ("get pose" in data):
                conn.send(json.dumps(Persons).encode())
            elif("calibration:" in data):
                conn.send("done".encode())
            else:
                conn.send("error".encode())
            conn.close()

            if mythread1.isAlive() is False:
                print('faceDirction ERROR')
                break
            elif mythread2.isAlive() is False:
                print('multiFaceAPI ERROR')
                break
            
    except KeyboardInterrupt:
        print('faceDirction', mythread1.isAlive())
        print('multiFaceAPI', mythread2.isAlive())
    
    while(mythread1.isAlive() is True or mythread2.isAlive() is True):
        FD_Process['Process'] = False
        time.sleep(0.5)
    time.sleep(1)
    print('faceDirction', mythread1.isAlive())
    print('multiFaceAPI', mythread2.isAlive())
