""" This lib defined the data structure for the box.
    
    Author: Lyu Yaopengfei
    Date: 23-May-2016
"""

import cv2
import numpy as np

# surface define
SUF_DEF = ('Front', 'Up', 'Back', 'Down', 'Left', 'Right')
        
class boxds(object):
    def __init__(self, name):
#         if(type(name) is not str):
        self.boxname = str(name)
        self.number = None
        self.imgFolder = None
        self.featureImgs = {SUF_DEF[0]:None, SUF_DEF[1]:None, SUF_DEF[2]:None, \
                            SUF_DEF[3]:None, SUF_DEF[4]:None, SUF_DEF[5]:None}
        self.featureImgsName = {SUF_DEF[0]:None, SUF_DEF[1]:None, SUF_DEF[2]:None, \
                                SUF_DEF[3]:None, SUF_DEF[4]:None, SUF_DEF[5]:None}
        self.imgsNum = 0
        self.tempNumList = None
        self.barcode = None
        
    def changeName(self, name):
        self.boxname = str(name)
    
    def setNumber(self,number):
        self.number = number
    
    def setImgFolder(self,imgfolder):
        self.imgFolder = str(imgfolder)
    
    def setSingleFeatureImg(self, surface, img):
        if(isinstance(img, np.ndarray)):
            self.featureImgs[surface] = img
        elif(isinstance(img, str)):
            self.featureImgs[surface] = cv2.imread(img, 0)
        elif(img is None):
            self.featureImgs[surface] = None
        else:
            raise ValueError('invalied img input')
        
    def setFeatureImgs(self, imgs):
        for (surface,img) in list(imgs):
            self.setSingleFeatureImg(surface, img)
    
    def setSingleFeatureImgsName(self, surface, imgpath):
        self.featureImgsName[surface] = imgpath
        self.imgsNum = self.imgsNum+1
    
    def setFeatureImgsName(self, imgnames):
        for (surface,imgpath) in list(imgnames):
            self.setSingleFeatureImgsName(surface,imgpath)
        
    def setBarcode(self,barcode):
        self.barcode = barcode
        
    def appendTempNumList(self,numList):
        if (self.tempNumList is None):
            self.tempNumList = []
        self.tempNumList.append(numList)
        
#     def getBarcode():

#     def __repr__(self):
#         print('boxname: ' + str(self.boxname))
#         print('number: ' + str(self.number))
#         print('imgsNum: ' + str(self.imgsNum))
#         print('barCode:' + str(self.barcode))     
           
#     def __str__(self):
#         self.__repr__()

# write box dataset information to file
def dsWrite(filePath, bds):
    boxData = open(filePath,'a+')
    writeStr = bds.boxname+' '+str(bds.number)+' '+bds.imgFolder+' '
    for i in range(6):
        writeStr = writeStr+str(bds.featureImgsName[SUF_DEF[i]])+' '
    writeStr = writeStr + str(bds.barcode) + ' ;\n'
#     print(writeStr)
    boxData.writelines(writeStr)
    boxData.close()

# read box information from file and set box dataset
def dsRead(filePath):
    boxesData = open(filePath,'r')
    bxes = []
    while(1):
#         cnt = cnt+1
        boxStr = boxesData.readline()
        if(len(boxStr) is 0):
            break
        bxes.append(boxds(None))
#         print(bxes)
        boxStrSplit = boxStr.split(' ')
        bxes[-1].changeName(boxStrSplit[0])
        bxes[-1].setNumber(boxStrSplit[1])
        bxes[-1].setImgFolder(boxStrSplit[2])
#         print(bxes)
        imgNames = []
        imgs = []
        for i in range(6):
            imgpath = bxes[-1].imgFolder+boxStrSplit[i+3]
            if (boxStrSplit[i+3] is 'None'):
                boxStrSplit[i+3] = None
                imgpath = None
            imgNames.append([SUF_DEF[i],boxStrSplit[i+3]])
            imgs.append([SUF_DEF[i],imgpath])
        bxes[-1].setFeatureImgsName(imgNames)
        bxes[-1].setFeatureImgs(imgs)
#         print(bxes[-1])
#     print(bxes[0].boxname,bxes[1].boxname,bxes[2].boxname,bxes[3].boxname,bxes[4].boxname)
    return bxes
