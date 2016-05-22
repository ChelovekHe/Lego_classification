import cv2
import numpy as np
from PIL import Image
from tesserwrap import Tesseract
from skimage.measure import compare_ssim as ssim

temp_img = None
temp_count = 0


def denoise_info(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
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


def save_info(img, s):
    global temp_count
    if (s < 0.8) & (temp_count <= 100):
        cv2.imwrite('../info/info' + str(temp_count) + '.jpg', img)
        temp_count += 1
        print(temp_count)
