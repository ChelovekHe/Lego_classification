""" This is image Optical Character Recognition(OCR) module. It's using for 
    extract the number from the image
    
    Author: Lei Zhang & Lyu Yaopengfei
    Date: 24-May-2016
"""

import cv2
from PIL import Image
import numpy as np
from pytesseract import image_to_string
from tesserwrap import Tesseract

def tesserOcr(img):
    numStr = None
    if(isinstance(img, np.ndarray)):
        if(img.ndim < 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgPIL = Image.fromarray(img)
    else:
        imgPIL = img
        
    try:
        numList = image_to_string(imgPIL,lang='eng')
        # tr = Tesseract(datadir='../data', lang='eng')
        # numList = tr.ocr_image(imgPIL)
    except:
        numList = None
    else:
        # print(numList)
        if(len(numList) < 2):
            numList = None
        else:
            numList = numList.split('\n')
            tempLen = 0
            maxLen = 0
            for item in numList:
                eachNumList = list(filter(str.isdigit,item))
                tempLen = len(eachNumList)
                if(tempLen > maxLen):
                    maxLen = tempLen
                    numStr = ''.join(eachNumList)
    return numStr