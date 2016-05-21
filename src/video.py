import os

import cv2
from PIL import Image
from tesserwrap import Tesseract
from Lego.Lego import Lego
from Lego.imgPreprocessing import LogoAffinePos
from Lego.extend import *

FRAME_SIZE_FACTOR = 0.4
temp_count = 0
logo_box = None

def resize(image, factor=0.5):
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return image

def ocr(info):
    cv2.imwrite('info.jpg', info)
    img = Image.open('info.jpg')
    tr = Tesseract(datadir='../tessdata', lang='eng')
    text = tr.ocr_image(img)
    return text

def initial_lyu_class():
    imgPATH = '../fig/'
    logoTp = cv2.imread(imgPATH + 'purelogo256.png')
    lyu = LogoAffinePos(logoTp, featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(),
                        matchMethod='knnMatch')
    return lyu

def initial_li_class(image):
     li = Lego(image)
     return li

def get_affined_image(lyu, image):
    logoContourPts, cPts, affinedcPts, affinedImg, croped, rtnFlag = lyu.rcvAffinedAll(image)
    if (rtnFlag is True):
        affined = affinedImg
        affined = resize(affined, FRAME_SIZE_FACTOR)
        # cv2.imshow('affined', affined)
        # cv2.moveWindow('affined',int(1280*FRAME_SIZE_FACTOR),int(720*FRAME_SIZE_FACTOR))
        lyu_info = croped
        # global logo_box
        # logo_box = cPts
        if lyu_info is not None:
            lyu_info = denoise_info(lyu_info)
            lyu_info = resize(lyu_info)
            cv2.imshow('lyu_info', lyu_info)
            cv2.moveWindow('lyu_info',int(1280*FRAME_SIZE_FACTOR),int(720*FRAME_SIZE_FACTOR))
        return lyu_info

def get_rotated_image(li):
    if li._has_rotated_image:
        rotated = li.get_rotated_image()
        rotated = resize(rotated, FRAME_SIZE_FACTOR)
        cv2.imshow('rotate', rotated)
        cv2.moveWindow('rotate',int(1280*FRAME_SIZE_FACTOR),0)

def get_info_part(li):
    li_info = li.get_information_part()
    if li_info is not None:
        li_info = resize(li_info)
        cv2.imshow('li_info', li_info)
        cv2.moveWindow('li_info',int(1280*FRAME_SIZE_FACTOR),0)
    return li_info

def save_info(img, s):
    global temp_count
    if (s < 0.8) & (temp_count<=100):
        cv2.imwrite('../info/info'+ str(temp_count) +'.jpg', img)
        temp_count += 1
        print(temp_count)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    lyu = initial_lyu_class()
    while 1:
        _, frame = cap.read()

        li = initial_li_class(frame.copy())
        # get_rotated_image(li)

        li_info = get_info_part(li)
        # if li_info is not None:
            # li_info = cv2.resize(li_info, (80, 80))
            # print(compare_image(li_info))
            # save_info(li_info, compare_image(li_info))

        lyu_info = get_affined_image(lyu, frame.copy())
        # if lyu_info is not None:
        #     lyu_info = cv2.resize(lyu_info, (80, 80))
            # save_info(lyu_info, compare_image(lyu_info))
            # text = ocr(lyu_info)
            # print(text)

        # logo_box = li.get_logo_box()

        cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)
        frame = resize(frame, FRAME_SIZE_FACTOR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
