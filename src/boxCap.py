""" Capturing and analyzing the box information
    
    Author: Lyu Yaopengfei
    Date: 23-May-2016
"""

import cv2
from PIL import Image
import dsOperation as dso
import imgPreprocessing as imgprep
import threading
import time
from ocr import tesserOcr

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

clickCnt = 0
clickFlag = 0
def detect_circle(event,x,y,flags,param):
    global clickFlag

    if event==cv2.EVENT_LBUTTONUP:
        clickFlag = clickFlag+1
    elif event==cv2.EVENT_RBUTTONUP:
        clickFlag = -1
#         lock.acquire()
#         try:
#             cv2.imwrite('cap.png',capImg)
#         finally:
#             clickCnt = clickCnt+1
#             lock.release()

# detect the useful information from the selected image
def detectImg(logoAffinePos,img,idx):
    _, _, _, _, affinedCropedImg, rtnFlag = logoAffinePos.rcvAffinedAll(img)
    if (rtnFlag is False):
        return None,None,None,False
    filtedCroped = imgprep.imgFilter(affinedCropedImg)
    filtedCroped = cv2.cvtColor(filtedCroped,cv2.COLOR_GRAY2RGB)
    filtedCropedPIL = Image.fromarray(filtedCroped)
    numStr = tesserOcr(filtedCropedPIL)
    return affinedCropedImg,filtedCroped,numStr,True
    
def analyseBoxInfo(bds,imgfolder):
    maxCnt = 0
    tempCnt = 0
    tempNumSet = set(bds.tempNumList)
    bds.setImgFolder(imgfolder)
    for item in tempNumSet:
        tempCnt = bds.tempNumList.count(item)
        if(tempCnt > maxCnt):
            maxCnt = tempCnt
            bds.number = item

def exportLog(lf, expStr):
    print(expStr)
    expStr = expStr+'\n'
    lf.writelines(expStr)

if __name__ == '__main__':
    bxnm = input('Input the box name: ')
#     time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))
    bx1 = dso.boxds(bxnm)
    settingInfo = open('./data/setting','r')
    settingInfo.readline()
    PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    DATAPATH = settingInfo.readline().strip().lstrip().rstrip(',')
    FEATURE_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    MATERIAL_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    BOX_DATA_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    LOG_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    
    curTime = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))
    LOG_PATH = LOG_PATH+curTime+bx1.boxname+'.log'
    logFile = open(LOG_PATH,'w+')
    
    boxData = open(BOX_DATA_PATH,'r')
    
    logoTp = cv2.imread(MATERIAL_IMG_FOLDER + 'purelogo256.png')
    logoAffinePos = imgprep.LogoAffinePos(logoTp)
    
    cv2.namedWindow('capFrame')
    cv2.setMouseCallback('capFrame',detect_circle)
    
    VWIDTH = 1280
    VHIGH = 720
    cap = cv2.VideoCapture(0)
    cap.set(3,VWIDTH)
    cap.set(4,VHIGH)
    cap.read();cap.read();cap.read()
    tCapFrame = threading.Thread(target=capFrame, args=(cap,))
    tCapFrame.start()
    while(capImg is None):
        pass
    dtrtnFlag = False
    showFlag = 0
    while(1):
        if ((cv2.waitKey(1) & 0xFF == 27) | (clickCnt>=6) ):
            stopFlat = 1
            break

        resImg = capImg.copy()
        showImg = resImg.copy()
        logoContourPts,logoContour,rtnFlag = logoAffinePos.extLegoLogo(resImg, minArea=5000)
        if (rtnFlag is True):
            # draw contour we finding
            cv2.drawContours(showImg, [logoContourPts], -1, (0,255,0), 2)
            cPts,rtnFlag = logoAffinePos.extQuadrangleCpts(logoContourPts, logoContour)
            if (rtnFlag is True):
                # draw corner points we finding
                for idx, cPt in enumerate(cPts):
                    cPt = cPt.flatten()
                    ptsize = int(logoAffinePos.estLength/20)
                    showImg[cPt[1]-ptsize:cPt[1]+ptsize,cPt[0]-ptsize:cPt[0]+ptsize,:] = [255,255,0]
        showImg = cv2.resize(showImg,(0,0),fx=0.4,fy=0.4)
        
        # right click, discard the data and re-capturing
        if(clickFlag < 0):
            clickFlag = 0
            exportLog(logFile, 'Data was discarded')
            cv2.destroyWindow('filted')
        # capturing image
        if(clickFlag is 0):
            dtrtnFlag = False
            showFlag = 0
            cv2.putText(showImg,'Capturing '+bx1.boxname+'_'+dso.SUF_DEF[clickCnt]+' picture',(10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)
        
        # fisrt time left click, detect the image and output the result
        elif(clickFlag is 1):
            if(dtrtnFlag is False):
                affinedCropedImg,filtedCroped,numStr,dtrtnFlag = detectImg(logoAffinePos,resImg,clickCnt)
                if(dtrtnFlag is False):
                    # if detect result is False, set clickFlag 0, re-capturing
                    clickFlag = 0
                    exportLog(logFile, 'Detecting fault, re-capturing')
            elif(dtrtnFlag is True):
                cv2.imshow('filted',filtedCroped)
                cv2.moveWindow('filted',50+int(0.4*VWIDTH),50)
                exportLog(logFile, bx1.boxname+'_'+dso.SUF_DEF[clickCnt]+' OCR: '+str(numStr))
                dtrtnFlag = None
            else:
                cv2.putText(showImg,'Do you save this result? Lclick Save, Rclick Discard',(10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)
        elif(clickFlag is 2):
            exportLog(logFile, 'Saving '+bx1.boxname+'_'+dso.SUF_DEF[clickCnt]+' data')
            imgName = bx1.boxname+'_'+str(clickCnt)+'.tiff'
            savingPath = FEATURE_IMG_FOLDER + imgName
            savingPath2 = FEATURE_IMG_FOLDER + 'color/c' + imgName
            cv2.imwrite(savingPath, filtedCroped)
            cv2.imwrite(savingPath2, affinedCropedImg)
            bx1.setSingleFeatureImgsName(dso.SUF_DEF[clickCnt], imgName)
            exportLog(logFile, '--------Finish capturing--------\n')
            if(numStr is not None):
                bx1.appendTempNumList(numStr)
            clickCnt = clickCnt + 1
            clickFlag = 0
            cv2.destroyWindow('filted')
        else:
            clickFlag = 0
            cv2.destroyWindow('filted')
        cv2.imshow('capFrame',showImg)
        
    analyseBoxInfo(bx1,FEATURE_IMG_FOLDER)
    dso.dsWrite(BOX_DATA_PATH,bx1)
    print('\n')
    logFile.close()
    boxData.close()
    cap.release()
    cv2.destroyAllWindows()