# -*- coding: utf-8 -*-
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import sys
import os
import argparse
import time

import cv2
import numpy as np

def map(size_w, size_h, data):
    map_field_array = []
    for bodey_point_deta in data:
        bbox = [min(bodey_point_deta.x),min(bodey_point_deta.y),max(bodey_point_deta.x),max(bodey_point_deta.y)]
        map_size_w, map_size_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        map_field = np.zeros((size_h, size_w))
        map_field_array



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