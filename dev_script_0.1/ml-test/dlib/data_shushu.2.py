
import dlib
import cv2
import numpy as np
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras.layers import Dense,Activation


import faceAPIrequest


predictor_path = "./shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()


cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) # カメラ画像の横幅を1280に設定
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0) #明るさ補正 デフォルト128
cap.set(cv2.CAP_PROP_SATURATION, 100) #彩度補正 デフォルト128
cap.set(cv2.CAP_PROP_CONTRAST, 100) #コントラスト補正 デフォルト128
cap.set(cv2.CAP_PROP_GAIN, 0) #ゲイン補正 デフォルト128
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) #
#cap.set(cv2.CAP_PROP_EXPOSURE, 2)

data_count = 0
mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成

x = []
y = []
y_const = -100

white_bord_w = 1920
white_bord_h = 1080

model_w = keras.models.load_model('../20190722/model_w.h5', compile=False)
model_h = keras.models.load_model('../20190722/model_h.h5', compile=False)

vector_dict = {
                0:[ 0, 0],
                1:[ 0, 0.5],
                2:[ 0, 1],
                3:[ 0.5, 0],
                4:[ 0.5, 0.5],
                5:[ 0.5, 1],
                6:[ 1, 0],
                7:[ 1, 0.5],
                8:[ 1, 1]
                }

faceFile = '/home/ds-980/eye-track/dev_script_0.1/ml-test/dlib/face.png'
photoFile = '/home/ds-980/eye-track/dev_script_0.1/ml-test/dlib/photo.png'
while True:
    key = cv2.waitKey(1)
    ret_val, image = cap.read()
    if ret_val != True:
        continue
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (image.shape[1], image.shape[0]))
    #print(image.shape)
    face = image
    dets, scores, idx = detector.run(image, 0)
    cv2.imwrite(photoFile,image)
    
    if len(dets) == 1:
        d = dets[0]
        left = d.left()
        top  = d.top()
        right= d.right()
        bottom=d.bottom()
        face = image[top:bottom,left:right]
        if face.all:
            face = cv2.resize(face, (255, 255))
        else:
            continue
    else:
        continue
    """
    faceAPI_data2 = faceAPIrequest.main(dType='68points', imgFile=photoFile, portNum=3030)#ssd
    #print(faceAPI_data2 == [])
    if faceAPI_data2 != []:
        data = faceAPI_data2[0]['detection']['_box']
        x, y = int(data['_x']), int(data['_y'])
        width, height = int(data['_width']), int(data['_height'])
        face = image[y:y+height,x:x+width]
        face = cv2.resize(face, (255, 255))
    else:
        continue
    """
    cv2.imwrite(faceFile,face)
    white_bord = np.full((white_bord_h,white_bord_w,3),255,np.uint8)
    
    faceAPI_data = faceAPIrequest.main(dType='single68points', imgFile=faceFile, portNum=3031)#TYNY
    #faceAPI_d = faceAPIrequest.main(dType='AgeGender', imgFile=facesFile, portNum=3030)
    image_onPoint = image
    if len(faceAPI_data) != 0 and type(faceAPI_data) is dict and 'err' not in faceAPI_data:    #顔を認識できたら認識できた顔の数 以下を実行
        #______顔角度______
        #特徴点を格納
        parts = []
        #print(faceAPI_data)
        if '_positions' in faceAPI_data:
            point_data = faceAPI_data['_positions']
        elif 'landmarks' in faceAPI_data:
            point_data = faceAPI_data['landmarks']['_positions']
        for point in point_data:
            points = [point['_x'], point['_y']]
            image_onPoint = cv2.circle(face, (int(points[0]),int(points[1])), 3, (0,0,255), -1)
            parts.append(points)
            data_68point = np.array(parts)
            nomal_data = mmscaler.fit_transform(data_68point)
        x_sample = np.reshape(nomal_data,(1,136))

        if key == ord('s'):
            x.append(nomal_data)
            y.append(y_const)
            print(data_count)
            data_count += 1
        #print(x_sample.shape)
        r_w = model_w.predict([x_sample])[0]
        r_h = model_h.predict([x_sample])[0]
        head_x, head_y = 0, 0

        #for i, p in enumerate(r):
        #    head_x += vector_dict[i][1] * p
        #    head_y += vector_dict[i][0] * p
        
        for i in range(0, len(r_w)):
            head_x += r_w[i]*white_bord_w*i/len(r_w)
        for i in range(0, len(r_h)):
            head_y += r_h[i]*white_bord_h*i/len(r_h)
        
        r_w_ini_val = -40
        r_w_interval = 10
        r_h_ini_val = 20
        r_h_interval = 10
        r_w_deg = 0
        r_h_deg = 0
        for i in range(0, len(r_w)):
           r_w_deg += (r_w_ini_val+i*r_w_interval)*r_w[i] 
        for i in range(0, len(r_h)):
           r_h_deg += (r_h_ini_val-i*r_h_interval)*r_h[i] 
        print(r_w_deg, r_h_deg)
        
        #head_y = 500

        #print(r)
        white_bord = cv2.circle(white_bord, (int(head_x), int(head_y)), 50, (255,0,0), -1)
        #white_bord = cv2.circle(white_bord, (int(head_x*white_bord_w), int(head_y*white_bord_h)), 50, (255,0,0), -1)
    else:
        white_bord = np.full((white_bord_h,white_bord_w,3),0,np.uint8)

    if data_count > 100:
        break




    cv2.imshow('show',image_onPoint)
    cv2.imshow('white_bord',white_bord)
    
    if key == ord('q'):
        break

np.savez('./'+str(y_const), x=x, y=y)