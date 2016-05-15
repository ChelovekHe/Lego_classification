import cv2
import numpy as np
from Lego import Lego, imgPreprocessing
from PIL import Image
from tesserwrap import Tesseract


def show_info(l):

    info = l.get_information_part()

    if l._has_information:
        cv2.imshow('info', info)
        cv2.imwrite('info.jpg',info)
        img = Image.open('info.jpg')
        tr = Tesseract(datadir='../tessdata', lang='eng')
        text = tr.ocr_image(img)
        print(text)

def draw_potential_logo_box(l):
    logo_box = l.get_logo_box()
    cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)

def draw_corner_points(l):
    img = l._image
    if l._has_corner_points:
        points = l._corner_points
        for i in points:
            x = i[0][0]
            y = i[0][1]
            img[y:y+5, x:x+5, :] = [255, 255, 255]
    return img

def resize(img):
    frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while 1:
        _, frame = cap.read()

        l = Lego.Lego(frame)

        # draw_potential_logo_box(l)
        # corner_points = draw_corner_points(l)
        rotated = resize(l.get_rotated_image())
        affined = resize(l.get_affined_image())
        # show_info(l)

        # cv2.imshow('frame', corner_points)
        cv2.imshow('ratated', rotated)
        cv2.imshow('affined', affined)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     image = cv2.imread('/Users/harrysocool/Github/package_identification/fig/1.pic_hd.jpg')
#     image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
#     l = Lego.Lego(image)
#     draw_corner_points(l)
#     cv2.waitKey(0)