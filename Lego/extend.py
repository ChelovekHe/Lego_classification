import os

import cv2
import numpy as np
from PIL import Image
from tesserwrap import Tesseract
from skimage.measure import compare_ssim as ssim

temp_img = None
temp_count = 0
train_box = 1
train_box_logo = 1
ssim_list = []

def denoise_info(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # kernel = np.ones((2, 2), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = gray
    return thresh


def compare_image(img):
    global temp_img
    if temp_img is not None:
        s = ssim(img, temp_img)
    else:
        s = 0
    temp_img = img
    return s


def resize(image, factor=0.5):
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return image


def ocr(info):
    cv2.imwrite('../fig/info.jpg', info)
    img = Image.open('../fig/info.jpg')
    tr = Tesseract(datadir='../tessdata', lang='eng')
    text = tr.ocr_image(img)
    print(text)


def save_info_image(img, s):
    global temp_count, train_box, train_box_logo, ssim_list
    path = '../info/box' + str(train_box) + '.' + str(train_box_logo)
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.exists(path):
        os.rmdir(path)
        os.mkdir(path)

    if (s < 0.8) & (temp_count <= 100):
        cv2.imwrite(path + '/' + str(train_box_logo) + str(temp_count) + '.jpg', img)
        temp_count += 1
        ssim_list.append(s)
        print(temp_count)


def get_train_index():
    flag = False
    global train_box, train_box_logo, temp_count
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        flag = True
        return flag
    elif k == ord('b'):
        train_box += 1
        train_box_logo = 1
        temp_count = 0
        return flag
    elif k == ord('='):
        train_box_logo += 1
        temp_count = 0
        return flag
    return flag

def put_text(img):
    global train_box, train_box_logo
    string = 'box: ' + str(train_box) + ' orient: ' + str(train_box_logo)
    cv2.putText(img, string, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    return img

def write_ssim():
    global ssim_list, train_box
    path = '../info/box' + str(train_box)
    with open(path + '/ssim_value.txt','w+') as f:
        for i in range(1,len(ssim_list)-1):
            f.writelines(["%s\n" % str(ssim_list[i])])

def get_box_list():
    path = '../info/info.txt'
    with open(path) as f:
        list = f.read()
        line = list.split('\n')
    return line