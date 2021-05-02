# -*- coding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import sys
import os
import argparse
import time

import cv2
import numpy as np

def face_leftsort(image, size, data):
    image_height, image_width = image.shape[:2]
    face_sort_num_array = []    #json_num, leftsort_num
    for json_num in range(len(data)):
        face = data[json_num]
        faceRectangle_top = int(face['faceRectangle']['top'])
        faceRectangle_left = int(face['faceRectangle']['left'])
        left_num = faceRectangle_left//size
        face_sort_num_array.append(left_num)
    
    new_data = {}
    for i in range(len(face_sort_num_array)):
        left_num = face_sort_num_array[i]
        new_data[left_num] = data[i]

    for left_num in range(image_width//size):
        if left_num not in new_data:
            new_data[left_num] = None
    
    sorted(new_data.items())

    return new_data



def faceAPI(image, face_image_size = 100, resizeRate = 1, flag = False):
    headers = {
        #    'Content-Type': 'application/json',
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': '5bc9b987a7e448e0b20c4631cc79378e',    #APIキーを入力
    }

    params = urllib.parse.urlencode({
                                    #'returnFaceId': 'true',
                                    #'returnFaceLandmarks': 'true',
                                    #'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
                                    'returnFaceAttributes': 'age,gender,headPose,emotion'
                                    })

    if flag:
        image = cv2.imread(image)
    image = cv2.resize(image, (int(image.shape[1]/resizeRate), int(image.shape[0]//resizeRate)))
    retval, buf = cv2.imencode('.png', image)
    func(retval)
    body = buf.tobytes()
    
    if len(body) > 4000000:
        print("over 4MB image size")
        return 'over 4MB image size'

    #### API request
    #try:
    #conn = http.client.HTTPSConnection('japaneast.api.cognitive.microsoft.com')   #リージョンコードを入力
    conn = http.client.HTTPSConnection('eastasia.api.cognitive.microsoft.com')   #例)東アジアリージョン
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
    response = conn.getresponse()
    data = json.loads(response.read())
    if not data:
        return 'can not detect face faceAPI'
    #print(json.dumps(data, indent=4))
    if 'faceAttributes' not in data[0]:
        return 'can not detect face faceAPI'
    if 'emotion' not in data[0]['faceAttributes']:
        return 'faceAPI_error not emotion data'
    headPose_pitch = data[0]['faceAttributes']['headPose']
    headPose_roll = data[0]['faceAttributes']['headPose']['roll']
    headPose_yaw = data[0]['faceAttributes']['headPose']['yaw']
    conn.close()
    #emotion_dict = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
    #return dict(emotion_dict)
    #return headPose_roll, headPose_yaw
    data = face_leftsort(image, face_image_size, data)
    return data
#    except Exception as e:
#        print("[Errno {0}]".format(e))
#        return 'error faceAPI.py'



def func(k):
    pass

def webcam2FaceAPI(resizeRate, camera, emotion_coef = 20, faceAPI_SPF = 0.0):
    cap = cv2.VideoCapture(0)
    spf = 0.0
    spf_time = time.time()
    faceAPI_count = 0
    time1 = time.time()

    while True:
        ret, image = cap.read()
        func(ret)
        image = cv2.resize(image, (int(image.shape[1]/resizeRate), int(image.shape[0]//resizeRate)))
        emotion_dict = faceAPI(image)
        faceAPI_count += 1
        if type(emotion_dict) is dict:
            emo1 = max(emotion_dict.items(), key=lambda x:x[1])[0]
            emo1val = emotion_dict.pop(emo1)
            emotion_cher = emo1
            if emo1 == 'neutral':
                emo2 = max(emotion_dict.items(), key=lambda x:x[1])[0]
                emo2val = emotion_dict.pop(emo2)
                if emo2val * emotion_coef > emo1val:
                    emotion_cher = emo2
        else:
            emotion_cher = emotion_dict 

        spf = (time.time() - spf_time)
        spf_time = time.time()
        if faceAPI_SPF > spf:
            time.sleep(faceAPI_SPF - spf)
            spf_time = time.time()
            spf += faceAPI_SPF - spf
        print('%d   emotion : %s  faceAPIspf : %f   time : %f' %(faceAPI_count, emotion_cher, spf, time.time() - time1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='faceAPI.py emotion')
    parser.add_argument('-i', '--Image', type=str, default=None)
    parser.add_argument('-w', '--WebcamRead', type=bool, default=False)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize-rate', type=float, default=1.0,
                        help='default=1.0. if resize-rate=2.0 => half size image (sent to FaceAPI')
    parser.add_argument('-e', '--emotion-coefficient', type=float, default=20,
                        help='default=10. multiply the second evaluated expression by -e. And it will be compare the first evaluated again.')

    args = parser.parse_args()

    image = args.Image
    webcamRead = args.WebcamRead
    camera = args.camera
    resizeRate = args.resize_rate
    emotion_coef = args.emotion_coefficient

    if image:
        print(faceAPI(image, 100, resizeRate, flag=True))

    if webcamRead:
        webcam2FaceAPI(resizeRate, camera, emotion_coef)

    
    #print(max(faceAPI(image, flag = True).items(), key=lambda x:x[1])[0])