""" main function
    
    Author: Lyu Yaopengfei
    Date: 24-May-2016
"""

import cv2
from PIL import Image
import threading
import Lego.dsOperation as dso
import Lego.imgPreprocessing as imgprep
# import time
from Lego.ocr import tesserOcr
import Levenshtein

capImg = None
resImg = None
stopFlat = 0
lock = threading.Lock()

def capFrame(cap):
    global capImg
    global stopFlat
    while(1):
        lock.acquire()
        try:
            _,capImg = cap.read()
        finally:
            lock.release()
        if (stopFlat > 0):
            break

def numMatch(boxesds,num):
    matchRst = None
    if (num is None):
        return matchRst
    tempSim = 0
    maxSim = 0
    print('+------------------+')
    for item in boxesds:
        tempSim = Levenshtein.jaro_winkler(str(item.number),num)
        print(item.boxname+': '+ str(tempSim))
        if(tempSim > maxSim):
            maxSim = tempSim
            matchRst = item.boxname
    print('+------------------+\n')
    
    return matchRst
if __name__ == '__main__':
    settingInfo = open('../data/setting','r')
    settingInfo.readline()
    PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    DATAPATH = settingInfo.readline().strip().lstrip().rstrip(',')
    FEATURE_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    MATERIAL_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    BOX_DATA_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    LOG_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    
    boxesds = dso.dsRead(BOX_DATA_PATH)
#     print(boxdes)
    logoTp = cv2.imread(MATERIAL_IMG_FOLDER+'purelogo256.png')
    logoAffinePos = imgprep.LogoAffinePos(logoTp)
    
    VWIDTH = 1280
    VHIGH = 720    
    cv2.namedWindow('capFrame')
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,VWIDTH)
    ret = cap.set(4,VHIGH)
    cap.read();cap.read();cap.read();cap.read();
    ret,capImg = cap.read()
    
    tCapFrame = threading.Thread(target=capFrame, args=(cap,))
    tCapFrame.start()
    # flush the frame
    while(capImg is None):
        pass
    
    showNarrowScale = 0.4
    startPosx = 50
    startPosy = 50
    
    while(1):
        if ((cv2.waitKey(1) & 0xFF == 27)):
            stopFlat = 1
            break

        resImg = capImg.copy()
        showImg = resImg.copy()
#         print(resImg)
        logoContourPts, cPts, affinedcPts, affinedImg, affinedCropedImg, rtnFlag = logoAffinePos.rcvAffinedAll(resImg)
        if (logoContourPts is not None):
            # draw contour we finding
            cv2.drawContours(showImg, [logoContourPts], -1, (0,255,0), 2)
            if (cPts is not None):
                # draw corner points we finding
                for idx, cPt in enumerate(cPts):
                    cPt = cPt.flatten()
                    ptsize = int(logoAffinePos.estLength/20)
                    showImg[cPt[1]-ptsize:cPt[1]+ptsize,cPt[0]-ptsize:cPt[0]+ptsize,:] = [255,255,0]
                    
        if (rtnFlag is True):
            affinedImgNarrow = cv2.resize(affinedImg,(0,0),fx=showNarrowScale,fy=showNarrowScale)
            cv2.imshow('croped',affinedCropedImg)
            cv2.moveWindow('croped',startPosx+int(showNarrowScale*VWIDTH),startPosy)
#             filtedCroped = imgprep.imgFilter2(affinedCropedImg)
            filtratedCroped = imgprep.imgFilter(affinedCropedImg)
            cv2.imshow('filtratedCroped',filtratedCroped)
            cv2.moveWindow('filtratedCroped',startPosx+int(showNarrowScale*VWIDTH)+filtratedCroped.shape[1],startPosy) 
            filtratedCroped = cv2.cvtColor(filtratedCroped,cv2.COLOR_GRAY2RGB)
            filtratedCroped = Image.fromarray(filtratedCroped)
            numStr = tesserOcr(filtratedCroped)
            numMatch(boxesds,numStr)
#             print(numStr)

        showImg = cv2.resize(showImg,(0,0),fx=showNarrowScale,fy=showNarrowScale)
        redmaskNarrow = cv2.resize(logoAffinePos.redmask,(0,0),fx=showNarrowScale,fy=showNarrowScale)
        cv2.imshow('frame',showImg)
        cv2.moveWindow('frame',startPosx,startPosy)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()