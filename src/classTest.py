""" Test the function
    
    Author: Lyu Yaopengfei
    Date: 13-May-2016
"""

import cv2
import imgPreprocessing as imgprep
from PIL import Image
# import numpy as np
from ocr import tesserOcr

if __name__ == '__main__':
    settingInfo = open('./data/setting','r')
    settingInfo.readline()
    PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    DATAPATH = settingInfo.readline().strip().lstrip().rstrip(',')
    FEATURE_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    MATERIAL_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    BOX_DATA_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    LOG_PATH = settingInfo.readline().strip().lstrip().rstrip(',')

    logoTp = cv2.imread(MATERIAL_IMG_FOLDER+'purelogo256.png')
    logoAffinePos = imgprep.LogoAffinePos(logoTp)
#     logoAffinePos = imgprep.LogoAffinePos(logoTp,featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(), \
#                                                     matchMethod = 'knnMatch')

    VWIDTH = 1600
    VHIGH = 900
#     VWIDTH = 1280
#     VHIGH = 720
#     VWIDTH = 960
#     VHIGH = 540
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    ret,img = cap.read()
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()
    showNarrowScale = 0.4
    startPosx = 50
    startPosy = 50
#     cnt = 0
    while(1):
        ret,imgRes = cap.read()
        img = imgRes.copy()
        imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         imgR = imgRes.copy()
#         imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        logoContourPts, cPts, affinedcPts, affinedImg, affinedCropedImg, rtnFlag = logoAffinePos.rcvAffinedAll(img)
        if (logoContourPts is not None):
            # draw contour we finding
            cv2.drawContours(img, [logoContourPts], -1, (0,255,0), 2)
            if (cPts is not None):
                # draw corner points we finding
                for idx, cPt in enumerate(cPts):
                    cPt = cPt.flatten()
                    ptsize = int(logoAffinePos.estLength/20)
                    img[cPt[1]-ptsize:cPt[1]+ptsize,cPt[0]-ptsize:cPt[0]+ptsize,:] = [255,255,0]
        
        if (rtnFlag is True):
            affinedImgNarrow = cv2.resize(affinedImg,(0,0),fx=showNarrowScale,fy=showNarrowScale)
            cv2.imshow('croped',affinedCropedImg)
            cv2.moveWindow('croped',startPosx+int(showNarrowScale*VWIDTH),startPosy)
#             filtedCroped = imgprep.imgFilter2(affinedCropedImg)
            filtedCroped = imgprep.imgFilter(affinedCropedImg)
            cv2.imshow('filtedCroped',filtedCroped)
            cv2.moveWindow('filtedCroped',startPosx+int(showNarrowScale*VWIDTH)+filtedCroped.shape[1],startPosy) 
            filtedCroped = cv2.cvtColor(filtedCroped,cv2.COLOR_GRAY2RGB)
            filtedCroped = Image.fromarray(filtedCroped)
            numStr = tesserOcr(filtedCroped)
            print(numStr)

        imgNarrow = cv2.resize(img,(0,0),fx=showNarrowScale,fy=showNarrowScale)
        redmaskNarrow = cv2.resize(logoAffinePos.redmask,(0,0),fx=showNarrowScale,fy=showNarrowScale)
        cv2.imshow('frame',imgNarrow)
        cv2.moveWindow('frame',startPosx,startPosy)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()