import os

import cv2
from PIL import Image
import numpy as np


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1

if __name__ == '__main__':
    picture = np.array([])
    picture1 = np.array([])
    path = '../info/'
    list1 = listdir_no_hidden(path)
    temp1 = None
    for i in range(0, len(list1)-1):
        list2 = listdir_no_hidden(path+list1[i])
        temp = None
        for j in range(0, 299, 10):
            img = Image.open(path + list1[i] + '/' + list2[j])
            arr = np.asarray(img, dtype='uint8')
            if temp is not None:
                picture = np.hstack((picture, arr))
            elif temp is None:
                picture = arr
                temp = arr
        if temp1 is not None:
            picture1 = np.vstack((picture1, picture))
        elif temp1 is None:
            picture1 = picture
            temp1 = picture
    cv2.imwrite('../fig_sample/present_pic.jpg', picture1)
