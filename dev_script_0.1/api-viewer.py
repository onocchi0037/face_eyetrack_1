# -*- coding: utf-8 -*-

import requests
import json
import cv2
import numpy as np
import time 
import argparse

def main():
    start = time.time()

    id = {}
    requestsURL = 'http://127.0.0.1:5000/mcs'
    response = requests.get(requestsURL,params=id)
    if len(response.text) == 0:
        res = 'ERROR: return is nan'
    else:
        res = json.loads(response.text)
    #ret = json.dumps(res, indent=2)
    #print(res)
    return res

def view():
    bord_size_w = 1920
    bord_size_h = 1080
    fps = 3
    access_time = 1/5
    
    bord_size = (bord_size_h, bord_size_w, 3)
    while(1):
        try:
            white_bord = np.full(bord_size, 255, np.uint8)
            ret = main()
            persons = ret["Persons"]
            for person in persons:
                eye = person["eye"].split(",")
                eye_x, eye_y = eye[:2]
                eye_x, eye_y = float(eye_x), float(eye_y)
                eye_x, eye_y = (eye_x/(-2.0))+0.5, (eye_y/(-2.0))+0.5
                disp_eye_x, disp_eye_y = int(bord_size_w*eye_x), int(bord_size_h*eye_y)
                cv2.circle(white_bord, (disp_eye_x, disp_eye_y), bord_size_h//5, (0,0,255), -1)

                gender = person["gender"]
                age = str(person["age"])[0:4]
                distance = str(person["distance"])[0:4]
                disp_gender_x, disp_gender_y = disp_eye_x - 100, disp_eye_y - 100
                disp_age_x, disp_age_y = disp_gender_x, disp_gender_y + 100
                disp_distance_x, disp_distance_y = disp_age_x, disp_age_y + 100
                font_collor = (0,0,0)
                font_size = 100//30
                font_thickness = 10

                cv2.putText(white_bord, gender, (disp_gender_x, disp_gender_y ), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_collor, font_thickness)
                cv2.putText(white_bord, 'age:' + age, (disp_age_x, disp_age_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_collor, font_thickness)
                cv2.putText(white_bord, 'dst:' + distance, (disp_distance_x, disp_distance_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_collor, font_thickness)

                print(eye_x,eye_y)
            cv2.imshow('show',white_bord)
            print()
            time.sleep(access_time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except:
            white_bord = np.full(bord_size, 255, np.uint8)
            cv2.imshow('show',white_bord)
            time.sleep(access_time)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dType', type=str, default='Custom', help=
        'you can select \'68points\', \'single68points\',\n \
        \'AgeGender\', \'Expressions\',\n \
        \'Descriptors\', and \'Custom\'.\n \
        Default is \'Custom\''
    )
    parser.add_argument('--imgFile', type=str, default='../images/bbt1.jpg', help='Default is \'../images/bbt1.jpg\'')
    parser.add_argument('--portNum', type=int, default=50007, help='Default is 3030')
    args = parser.parse_args()

    dType = args.dType
    imgFile = args.imgFile
    portNum = args.portNum

    view()
    #ret = main()
    #print(type(ret))

