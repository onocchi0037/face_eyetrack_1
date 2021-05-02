# -*- coding: utf-8 -*-

import dlib
import cv2
import sys
import numpy as np

def func(k):
    pass

def lOS_main(detector, predictor, inputImage, outputImage, m, rxk, lxk):
    #img = cv2.imread(inputImage)
    img = inputImage
    #img = m_scale(img, m)
    lOSmap = [-1,-1]

    dets = detector(img[:, :, ::-1]) # bgr2rgb
    if len(dets) == 1:
        parts = predictor(img, dets[0]).parts()
        pA = []
        for pa in parts:
            pA.append([pa.x, pa.y])
        face_yaw, face_roll = pupil_faceangle(parts)

        if abs(face_roll) > 1.5:
            #print("roll error")
            return img, lOSmap

        eye_scale = 0

        #print("\tL eye ::", end="")
        # 左目瞳の処理
        eyearea, eyeRelpoint = pupil_eyeArea(parts, True)
        if eyearea == None:
            draw_monitor(img)
            return img, lOSmap
        center = pupilPoint(img, eyearea, eyeRelpoint)
        if center == None:
            draw_monitor(img)
            return img, lOSmap
        w = eyearea[3] - eyearea[2]
        centerxper_L = (center[0] - eyearea[2] - w/2)*2/w
        #print("\tangle : %f" %(centerxper_L), end="")
        center_L = center
        eye_scale += abs((eyearea[1]-eyearea[0])*(eyearea[3]-eyearea[2]))

        #print("\tR eye ::", end="")
        # 右目瞳の処理
        eyearea, eyeRelpoint = pupil_eyeArea(parts, False)
        if eyearea == None:
            draw_monitor(img)
            return img, lOSmap
        center = pupilPoint(img, eyearea, eyeRelpoint)
        if center == None:
            draw_monitor(img)
            return img, lOSmap
        w = eyearea[3] - eyearea[2]
        centerxper_R = (center[0] - eyearea[2] - w/2)*2/w
        #print("\tangle : %f" %(centerxper_R), end="")
        center_R = center
        eye_scale += abs((eyearea[1]-eyearea[0])*(eyearea[3]-eyearea[2]))

        
        #print("\tface angle : %f" %(face_roll), end="")

        img2 = pupil_plot(img, parts, (center_L, center_R), outputImage)
        
        #print("")
        img2 = draw_headCenter(img2, parts)
        img2 = draw_monitor(img2)
        img2, lOSmap = lineOfSight(img2, centerxper_L, centerxper_R, face_yaw, face_roll, eye_scale, rxk, lxk)

        return img2, lOSmap, pA

    else:
        #print("skip : Detected %d faces. " %(len(dets)))
        draw_monitor(img)
        return img, lOSmap, []

def lineOfSight(img, centerxper_L, centerxper_R, face_yaw, face_roll, eye_scale, rxk, lxk):
    esk = 900 #my eyescale distance550mm
    exa = 36.0
    exk = 0.85
    fxk = 0.6


    exa = exa/np.sqrt(eye_scale/esk)
    fxk = fxk/np.sqrt(eye_scale/esk)

    h = img.shape[0]
    w = img.shape[1]


    #eye_x
    ##fx__face yaw
    fx = face_yaw - 1/2
    fxabsfun = np.sqrt(abs(fx))
    if fx > 0:
        fx = fxabsfun
    else:
        fx = -1*fxabsfun

    ##fr__face roll
    cv2.line(img, (w//2 + int((0 - h//2)//face_roll), 0), (w//2 + int((h - h//2)//face_roll), h), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    fr = abs(face_roll)

    ##ex with fx
    ex = exk*(centerxper_L+centerxper_R)/2    #-1<ex<1 *exk
    ex = ex + (fx * fxk) * (1 - fr)
 
    exabsfun = exa * (abs(ex) ** 4) + (-2*exa) * (abs(ex) ** 3) + exa * (abs(ex) ** 2)
    if ex > 0:#right
        ex = exabsfun * rxk
    else:#left
        ex = -1*exabsfun * lxk
    ###ex disp
    ex = ex + 1/2
    fx_disp = int((face_yaw) * w)
    ex_disp = int(ex * w)
    #exff_disp = int(ex * w) + (fx_disp - (w//2))
    #else:
    #    img = cv2.circle(img, (ex_disp, h//2), 8, (0, 0, 255), -1)
    
    img = cv2.circle(img, (fx_disp, h*2//3), 8, (0, 255, 0), -1)

    lOSmap = [ex, h//2]

    return img, lOSmap


def pupilPoint(img, eyearea, eyeRelpoint):
    try:
        eyeimg = img[eyearea[0]:eyearea[1], eyearea[2]:eyearea[3]]
        if eyeimg.shape[0] * eyeimg.shape[1] == 0:
            return None 

        eyeimggray = cv2.cvtColor(eyeimg, cv2.COLOR_BGR2GRAY)
        t1mask = cropWarpTriangle(eyeimggray, [eyeRelpoint[0],eyeRelpoint[1],eyeRelpoint[5]], [eyeRelpoint[0],eyeRelpoint[1],eyeRelpoint[5]], True)
        t2mask = cropWarpTriangle(eyeimggray, [eyeRelpoint[1],eyeRelpoint[4],eyeRelpoint[5]], [eyeRelpoint[1],eyeRelpoint[4],eyeRelpoint[5]], True)
        t3mask = cropWarpTriangle(eyeimggray, [eyeRelpoint[1],eyeRelpoint[2],eyeRelpoint[4]], [eyeRelpoint[1],eyeRelpoint[2],eyeRelpoint[4]], True)
        t4mask = cropWarpTriangle(eyeimggray, [eyeRelpoint[2],eyeRelpoint[3],eyeRelpoint[4]], [eyeRelpoint[2],eyeRelpoint[3],eyeRelpoint[4]], True)
        if t1mask is None or t2mask is None or t3mask is None or t4mask is None:
            return None
        tmask = cv2.bitwise_or(t1mask, t2mask)
        tmask = cv2.bitwise_or(tmask, t3mask)
        tmask = cv2.bitwise_or(tmask, t4mask)
        t1gray = cv2.bitwise_and(eyeimggray, eyeimggray, mask=t1mask )
        t2gray = cv2.bitwise_and(eyeimggray, eyeimggray, mask=t2mask )
        t3gray = cv2.bitwise_and(eyeimggray, eyeimggray, mask=t3mask )
        t4gray = cv2.bitwise_and(eyeimggray, eyeimggray, mask=t4mask )
        crpdeye = cv2.bitwise_and(eyeimggray, eyeimggray, mask=tmask )
        
        #grayscale avg and 2chi
        t1grayavg = int(np.sum(t1gray)//np.sum(t1gray != 0))
        t2grayavg = int(np.sum(t2gray)//np.sum(t2gray != 0))
        t3grayavg = int(np.sum(t3gray)//np.sum(t3gray != 0))
        t4grayavg = int(np.sum(t4gray)//np.sum(t4gray != 0))

        crpdeye_graymap = np.array([t1grayavg,t2grayavg,t3grayavg,t4grayavg])
        white = crpdeye_graymap[crpdeye_graymap.argsort()[3]]

        crpdeye = np.where(crpdeye == 0, white, crpdeye)

        _, eye2l = cv2.threshold(crpdeye, crpdeye_graymap[crpdeye_graymap.argsort()[1]], 255, cv2.THRESH_BINARY_INV)
        e1 = blackeye_Round(eye2l)

        #e2 = (blackeye_Round((np.array(eye2l)).T)).T
        #cv2.imshow("re2", e2)
        #cv2.waitKey(0)

        #cv2.imshow("imege", eye2l)
        #cv2.waitKey(0)

        ##center
        center_e1 = get_center(e1)
        e2 = widthMask(eye2l, center_e1, 0.5)
        eye2l3 = cv2.bitwise_and(eye2l, e2, mask=eye2l)
        #cv2.imshow("ime3", eye2l3)
        #cv2.waitKey(0)

        ##white black area menseki
        whtS = np.sum(tmask)
        blkS = np.sum(eye2l3//255)
        #print ("\tblk/wht : %f" %(blkS/whtS), end="")

        center = get_center(eye2l3)
        if center:

            return center[0] + eyearea[2], center[1] + eyearea[0]
        else:
            #print("not black eye")
            pass
    except:
        return None


def blackeye_Round(grayimg):
    w = grayimg.shape[1]
    returnimg = []
    topnum = 0
    topnum_old = 0
    func(topnum_old)
    for top in grayimg:
        topstate = top[0]
        tcount = 0
        topdist = 0
        topsign = []
        topsign1 = []
        
        for t in top:
            t = t//255
            if t != topstate:
                topsign.append([tcount,topstate,topdist])
                topstate = t
                topdist = 0
                if topstate == 1:
                    topsign1.append([tcount,topstate,topdist])

            tcount += 1
            topdist += 1

        topnum = len(topsign)
        if topnum == 2 and topsign[0][1] == 0 and topsign[1][2] < w//3:
            top = np.where(top != 0, 0, top)
            
        elif topnum > 2 and topsign[1][2] < w//3:
            top = np.where(top != 0, 0, top)
            #returnimg.append(np.array(topzero))
        #if topnum == 4 and topnum_old ==2:
        #    top = np.where(top != 1, 155, top)
        returnimg.append(np.array(top))
        topnum_old = topnum

    returnimg = np.array(returnimg)
        
    return returnimg


def widthMask(img, center, blceyesize_rate = 0.66):
    img = img.T
    returnimg = []
    h = img.shape[0]
    count = 0
    if center == None:
        center = [h//2]
    blceye_up   = center[0] - int(h*blceyesize_rate)//2
    blceye_down = center[0] + int(h*blceyesize_rate)//2

    for im in img:
        if count > blceye_up and count < blceye_down:
            im = np.where(im != 255, 255, im)
        else:
            im = np.where(im != 0, 0, im)
        returnimg.append(np.array(im))
        count += 1
    returnimg = np.array(returnimg).T   
    return returnimg        


def is_close(y0, y1):
    if abs(y0 - y1) < 10:
        return True
    return False

def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None

def cropWarpTriangle(inputimage, tri1Point, tri2Point, masksign = False):
    graysign = 0
    if inputimage is None:
        return inputimage
    if inputimage.shape == False:
        return inputimage
    if len(inputimage.shape) == 2:
        graysign = 1
    
    # Read input image and convert to float
    img1 = inputimage
    if masksign == True:
        img1 = np.ones(img1.shape, dtype = img1.dtype)

    # Output image is set to white
    img2 = 0 * np.ones(img1.shape, dtype = img1.dtype)
    # Define input and output triangles 
    tri1 = np.float32([tri1Point])
    tri2 = np.float32([tri2Point])
    # Find bounding box. 
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)
    # Offset points by left top corner of the 
    # respective rectangles
    tri1Cropped = []
    tri2Cropped = []
    for i in range(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))
    # Apply warpImage to small rectangular patches
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    # error check singyrarity
    if img1Cropped.shape[1] < 1 or img1Cropped.shape[0] < 1:
        return inputimage
    if r2[2] < 1 or r2[3] < 1:
        return inputimage
    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    # Get mask by filling triangle
    if graysign == 0:
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        oneArray = (1.0, 1.0, 1.0)
    else:
        mask = np.zeros((r2[3], r2[2]), dtype = np.float32)
        oneArray = (1.0)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), oneArray, 16, 0)

    # Apply mask to cropped region
    img2Cropped = img2Cropped * mask
    
    # Copy triangular region of the rectangular patch to the output image
    mask = np.resize(mask, img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].shape)
    img2Cropped = np.resize(img2Cropped, img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].shape)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( oneArray - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

    return img2


def m_scale(img, m):
    imgArry = np.array(img)
    w = imgArry.shape[1]
    h = imgArry.shape[0]
    if m != False:
        if h > w:
            r = float(m/w)
        else:
            r = float(m/h)
        img = cv2.resize(img, (int(r*w), int(r*h)))
    return img

def pupil_faceangle(parts):
    pA = []
    for pa in parts:
        pA.append([pa.x, pa.y])
    
    da = ((pA[22][0] - pA[21][0]) - (pA[22][0] - pA[27][0]))/(pA[22][0] - pA[21][0])
    db = ((pA[42][0] - pA[39][0]) - (pA[42][0] - pA[27][0]))/(pA[42][0] - pA[39][0])
    dc = ((pA[35][0] - pA[31][0]) - (pA[35][0] - pA[30][0]))/(pA[35][0] - pA[31][0])

    face_yaw = (da+db+dc)/3

    #roll for saishoujijouhou
    X1 = pA[27][0]
    Y1 = pA[27][1]
    X = np.array([0, pA[28][0] - X1, pA[29][0] - X1, pA[30][0] - X1])
    Y = np.array([0, pA[28][1] - Y1, pA[29][1] - Y1, pA[30][1] - Y1])
    Xr = []
    Yr = []
    alpha = -3.14/2
    for i in range(0, 4):
        Xr.append( X[i] * np.cos(alpha) - Y[i] * np.sin(alpha))
        Yr.append( X[i] * np.sin(alpha) + Y[i] * np.cos(alpha))
    Xr = np.array(Xr)
    Yr = np.array(Yr)
    A = np.array([[np.dot(Xr,Xr),Xr.sum()],[Xr.sum(),len(X)]])
    b = np.array([np.dot(Xr,Yr),Yr.sum()])
    B = np.linalg.solve(A,b)
    face_roll = B[0]

    return face_yaw, face_roll

def pupil_eyeArea(parts, left=True):

    if left:
        eyes = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x.y),
                max(parts[40], parts[41], key=lambda x: x.y),
                parts[39],
                ]
        eyepoint6 = [parts[36],parts[37],parts[38],parts[39],parts[40],parts[41]]
    else:
        eyes = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x.y),
                max(parts[46], parts[47], key=lambda x: x.y),
                parts[45],
                ]
        eyepoint6 = [parts[42],parts[43],parts[44],parts[45],parts[46],parts[47]]
    org_x = eyes[0].x
    org_y = eyes[1].y

    eyeRelpoint = []
    eyeRelpoint.append([eyepoint6[0].x-org_x + 1, eyepoint6[0].y-org_y])
    eyeRelpoint.append([eyepoint6[1].x-org_x, eyepoint6[1].y-org_y + 1])
    eyeRelpoint.append([eyepoint6[2].x-org_x, eyepoint6[2].y-org_y + 1])
    eyeRelpoint.append([eyepoint6[3].x-org_x - 1, eyepoint6[3].y-org_y])
    eyeRelpoint.append([eyepoint6[4].x-org_x, eyepoint6[4].y-org_y - 1])
    eyeRelpoint.append([eyepoint6[5].x-org_x, eyepoint6[5].y-org_y - 1])

    if is_close(org_y, eyes[2].y):
        return None, None

    eyearea = [org_y, eyes[2].y, org_x, eyes[-1].x]
    return eyearea, eyeRelpoint

def pupil_plot(img, parts, eye, outputImage):
    if eye[0]:
        cv2.circle(img, eye[0], 20, (255, 255, 0), -1)
    if eye[1]:
        cv2.circle(img, eye[1], 20, (255, 255, 0), -1)

    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    #cv2.imshow("imege", img)
    #cv2.waitKey(0)
    #cv2.imwrite(outputImage, img)
    return img

def draw_monitor(img):
    h = img.shape[0]
    w = img.shape[1]
    cv2.line(img, (w//3, 0), (w//3, h), (0, 255, 255), thickness=4, lineType=cv2.LINE_AA)
    cv2.line(img, (w*2//3, 0), (w*2//3, h), (0, 255, 255), thickness=4, lineType=cv2.LINE_AA)
    return img

def draw_headCenter(img, parts):
    h = img.shape[0]
    w = img.shape[1]
    xavg = 0
    yavg = 0
    count = 0 
    for i in range(36,48):
        xavg += parts[i].x
        yavg += parts[i].y
        count += 1
    xavg = xavg//count
    yavg = yavg//count

    cv2.line(img, (xavg, 0), (xavg, h), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(img, (0, yavg), (w, yavg), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return img
