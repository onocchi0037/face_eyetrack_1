# -*- coding: utf-8 -*-

import numpy as np
import cv2


if __name__ == "__main__":

    disp_resolution_h = 2160
    disp_resolution_w = 3840 

    #white_bord = cv2.imread('./all.png')
    white_bord = np.full((disp_resolution_h*2, disp_resolution_w*2, 3), 255, np.uint8)
    for i in range(0, 11):
        csv_file_name_sig = str(i).zfill(4)

        csv_file_name = './' + csv_file_name_sig + '.csv'
        new_csv_file_name = './new_' + csv_file_name_sig + '.csv'
        
        
        data_array = np.loadtxt(new_csv_file_name, delimiter = ",", dtype = "unicode")
        new_data_array = []

        #white_bord = np.full((disp_resolution_h*2, disp_resolution_w*2, 3), 255, np.uint8)
        for data in data_array:
            point_x, point_y = int(data[4])+(disp_resolution_w//2), int(data[5])+(disp_resolution_h//2)
            if data[1] == '0':
                white_bord = cv2.circle(white_bord, (point_x, point_y), 20, (255,0,0), -1)
                print(0, point_x, point_y)
            elif data[1] == '1':
                white_bord = cv2.circle(white_bord, (point_x, point_y), 20, (0,255,0), -1)
                print(1, point_x, point_y)
            elif data[1] == '2':
                white_bord = cv2.circle(white_bord, (point_x, point_y), 20, (0,0,255), -1)
                print(2, point_x, point_y)
            elif data[1] == '3':
                white_bord = cv2.circle(white_bord, (point_x, point_y), 20, (0,0,0), 5)
        white_bord = cv2.rectangle(white_bord, (disp_resolution_w//2,disp_resolution_h//2), (disp_resolution_w*3//2,disp_resolution_h*3//2),(0,0,0), 3)
        #cv2.imwrite(csv_file_name_sig+'.png', white_bord)
        cv2.imwrite('./all.png', white_bord)
        print(data_array)
        #np.savetxt(new_csv_file_name, np.array(new_data_array), delimiter = ",", fmt = "%s")
    pass