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

import os

import random
import numpy as np


def json_adjust(data, face_num, height):
    face_list = []
    new_data = {}

    for i in range(face_num):
        face_count = 0
        min_x = i * height
        max_x = min_x + height
        bbox_size_old = 0.0
        face_chk_flag = False
        for face_data in data:
            bbox_point = face_data['detection']['_box']
            bbox_x = float(bbox_point['_x'])
            bbox_size = float(bbox_point['_width']) + float(bbox_point['_height'])
            #print(min_x,max_x,bbox_point)
            if min_x <= bbox_x < max_x:
                if bbox_size > bbox_size_old:
                    face_data_output = face_data
                else:
                    face_data_output = face_data_old
                face_data_old = face_data
                bbox_size_old = bbox_size
                face_count += 1
                face_chk_flag = True
        if face_chk_flag:
            face_list.append(face_data_output)
            new_data[i] = face_data_output
        else:
            face_list.append({})
            new_data[i] = None

    #print(face_list)
    return new_data
                

                    




