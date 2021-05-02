# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import time

def load_test_number(test_number_file='./test_number.txt'):
    f = open(test_number_file)
    number = int(f.read())
    f.close()
    f = open(test_number_file, mode='w')
    f.write(str(number+1))
    
    return number

def area2number(image, n, m, X):
    '''
    画像をエリアごとに分割し、指定された座標をエリア番号に変換してreturnします
    n: 縦, m: 横: エリアの分割数
    X: どのエリア番号に当たるか知りたい座標
    '''
    x, y = int(X[0]), int(X[1])
    h, w = image.shape[:2]
    if x <= 0 or x >= w:
        return -1
    if y <= 0 or y >= h:
        return -1
    nl, ml = h//n, w//m
    ny, nx = y//nl, x//ml
    areaN = m * ny + nx
    return areaN


def number2image(image, n, m, areaN, color = (255, 0, 0), margin = 0.25):
    '''
    画像をエリアごとに分割し、指定されたエリアを色で塗ります
    色塗りされたimageを出力します
    n: 縦, m: 横: エリアの分割数
    areaN: エリアの番号
    '''
    h, w = image.shape[:2]

    if areaN == -1:
        image = cv2.rectangle(image, (0,0), (w,h), (0,255,255), -1)
        return image

    if areaN == -2:
        image = np.full((image.shape),255,np.uint8)
        #image = cv2.circle(image, (w//2, h//2), h//20,(0,255,255), -1)
        return image

    nl, ml = h/n, w/m     # エリアの大きさ
    areaNn, areaNm = areaN//m, areaN%m  # エリアの番号がどのエリアに当たるか
    margin_h, margin_w = nl*(1.0 - margin), ml*(1.0 - margin) 
    p0_x, p0_y = int(areaNm*ml + (margin_w/2)), int(areaNn*nl + (margin_h/2))    # 色を塗るエリアの座標 左上
    p0 = (p0_x, p0_y)
    p1 = (p0_x + int(ml*margin), p0_y + int(nl*margin))  #右下

    image = cv2.rectangle(image, p0, p1, color, -1)

    return image

def rand(num, u):
    '''
    num = n * m * areaN
    '''
    while(1):
        ret = int(random.uniform(0,num+1))-1
        if ret != u:
            break
    return ret


if __name__ == "__main__":
    load_test_nuber()
    '''
    areaN = -1
    n = 10
    m = 20
    while(1):
        image = np.full((100, 200, 3), 255, np.uint8)
        x, y = int(random.uniform(0, 200)), int(random.uniform(0, 100))
        #areaN = rand(4, areaN)
        areaN = area2number(image, n, m, [x, y])
        print(areaN)
        image = number2image(image, n, m, areaN)
        image = cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.imwrite('./test.png', image)
        time.sleep(0.1)
    '''

