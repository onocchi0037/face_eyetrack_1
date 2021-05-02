# -*- coding: utf-8 -*-

import requests
import json
import cv2
import time 
import argparse

def main(dType, imgFile, portNum = 3030):
    start = time.time()

    id = {'dType':dType, 'imgFile':imgFile}
    requestsURL = 'http://localhost:'+str(portNum)
    response = requests.get(requestsURL,params=id)
    if len(response.text) == 0:
        res = 'ERROR: return is nan'
    else:
        res = json.loads(response.text)
    ret = json.dumps(res, indent=2)
    #print(res)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dType', type=str, default='Custom', help=
        'you can select \'68points\', \'single68points\',\n \
        \'AgeGender\', \'Expressions\',\n \
        \'Descriptors\', and \'Custom\'.\n \
        Default is \'Custom\''
    )
    parser.add_argument('--imgFile', type=str, default='../images/bbt1.jpg', help='Default is \'../images/bbt1.jpg\'')
    parser.add_argument('--portNum', type=int, default=3030, help='Default is 3030')
    args = parser.parse_args()

    dType = args.dType
    imgFile = args.imgFile
    portNum = args.portNum

    main(dType, imgFile, portNum)

