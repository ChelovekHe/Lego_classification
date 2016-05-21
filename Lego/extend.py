import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

temp_img = None

def denoise_info(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    return thresh

def compare_image(img):
    global temp_img
    if temp_img is not None:
        s = ssim(img, temp_img)
    else:
        s = 0
    temp_img = img
    return s