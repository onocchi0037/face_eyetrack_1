
import dlib
import cv2
import numpy as np
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras.layers import Dense,Activation

predictor_path = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()


cap = cv2.VideoCapture(0)

data_count = 0
mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成

x = []
y = []
y_const = 1

white_bord_w = 1280
white_bord_h = 700

model = keras.models.load_model('../20190722/model.h5', compile=False)

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


while True:
    key = cv2.waitKey(1)
    ret_val, image = cap.read()
    if ret_val != True:
        continue
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

    white_bord = np.full((white_bord_h,white_bord_w,3),255,np.uint8)

    dets, scores, idx = detector.run(image, 0)
    if len(dets) == 1:
        d = dets[0]
        shape = predictor(image, d)
        shape_68point = []
        for i in range(68):
            shape_68point.append([shape.part(i).x, shape.part(i).y])
            data_68point = np.array(shape_68point)
            nomal_data = mmscaler.fit_transform(data_68point)
        x_sample = np.reshape(nomal_data,(1,136))

        if key == ord('s'):
            x.append(nomal_data)
            y.append(y_const)
            print(data_count)
            data_count += 1
        print(x_sample.shape)
        r = model.predict([x_sample])[0]
        head_x, head_y = 0, 0
        for i, p in enumerate(r):
            head_x += vector_dict[i][1] * p
            head_y += vector_dict[i][0] * p
        #head = r[1]*white_bord_w/2 + r[2]*white_bord_w
        print(r)
        white_bord = cv2.circle(white_bord, (int(head_x*white_bord_w), int(head_y*white_bord_h)), 50, (255,0,0), -1)
    if data_count > 100:
        break
    



    cv2.imshow('show',image)
    cv2.imshow('white_bord',white_bord)
    
    if key == ord('q'):
        break

np.savez('./'+str(y_const), x=x, y=y)