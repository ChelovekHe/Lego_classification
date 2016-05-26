import os
import cv2
from Lego.Lego import Lego
from Lego.imgPreprocessing import LogoAffinePos
from Lego.extend import *

FRAME_SIZE_FACTOR = 0.4
info_size = 30
logo_box = None
lyu_info = None


def initial_lyu_class():
    img_path = '../fig/'
    logoTp = cv2.imread(img_path + 'purelogo256.png')
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
        global logo_box
        logo_box = cPts
        if lyu_info is not None:
            gray = gray_image(lyu_info)
            lyu_info = cv2.resize(gray, (info_size, info_size))
            cv2.imshow('lyu_info', lyu_info.copy())
            cv2.moveWindow('lyu_info', int(1280*FRAME_SIZE_FACTOR), int(720*FRAME_SIZE_FACTOR))
        return lyu_info

def get_rotated_image(li):
    if li._has_rotated_image:
        rotated = li.get_rotated_image()
        rotated = resize(rotated, FRAME_SIZE_FACTOR)
        cv2.imshow('rotate', rotated)
        cv2.moveWindow('rotate', int(1280*FRAME_SIZE_FACTOR), 0)

def get_info_part(li):
    li_info = li.get_information_part()
    if li_info is not None:
        li_info = cv2.resize(li_info, (info_size, info_size))
        cv2.imshow('li_info', li_info)
        cv2.moveWindow('li_info', int(1280*FRAME_SIZE_FACTOR), 0)
    return li_info

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()

        if ret is True:
            # li = initial_li_class(frame.copy())
            lyu = initial_lyu_class()
            # logo_box = li.get_logo_box()
            # get_rotated_image(li)

            # li_info = get_info_part(li)
            # if li_info is not None:
                # print(compare_image(li_info))
                # save_info_image(li_info, compare_image(li_info))
            try:
                lyu_info = get_affined_image(lyu, frame.copy())
            except:
                pass
            if lyu_info is not None:
                # save_training_info_image(lyu_info)
                text = ocr(lyu_info)

            cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)
            frame = resize(frame, FRAME_SIZE_FACTOR)
            frame = put_text(frame.copy())
            cv2.imshow('frame', frame)

            if get_keyboard():
                break
    cap.release()
    cv2.destroyAllWindows()
