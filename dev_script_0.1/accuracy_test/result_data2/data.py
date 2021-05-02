# -*- coding: utf-8 -*-

import tensorflow as tf
import datetime
import json
import numpy as np



if __name__ == "__main__":
    for i in range(0, 11):
        csv_file_name_sig = str(i).zfill(4)
        first_H_m1_cache = ''
        first_H_m1_count = 0
        first_H_m1_flug = False
        #csv_file_name_sig = '0108'
        csv_file_name = './' + csv_file_name_sig + '.csv'
        new_csv_file_name = './new_' + csv_file_name_sig + '.csv'
        data_array = np.loadtxt(csv_file_name, delimiter = ",", dtype = "unicode")
        new_data_array = []
        for data in data_array:
            
            if data[1] != first_H_m1_cache:
                if data[1] == '-1':
                    first_H_m1_count += 1
                    print(first_H_m1_count)
            first_H_m1_cache = data[1]
            
            if data[2] == 's':
                new_data_array.append(data)
        #print(data_array)
        np.savetxt(new_csv_file_name, np.array(new_data_array), delimiter = ",", fmt = "%s")
        pass