import cv2
from lib import Lego, imgPreprocessing
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
    if l._has_potential_logo:
        logo_box = l.get_logo_box()
        cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)

def show_potential_logo_image(l):
    if l._has_potential_logo:
        cv2.imshow('logo', l.get_logo_image())

def show_rotated_image(l):
    if l._has_rotated_image:
        img = l._rotated_image
        img[l._logo_center_y:l._logo_center_y+3, l._logo_center_x:l._logo_center_x+3, :] = [255, 255, 255]
        cv2.imshow('rotate', img)

if __name__ == '__main__':
    VWIDTH = 640
    VHIGH = 480
    cap = cv2.VideoCapture(0)
    ret = cap.set(3, VWIDTH)
    ret = cap.set(4, VHIGH)

    while 1:
        _, frame = cap.read()

        l = Lego.Lego(frame)
        # ll = imgPreprocessing.LogoAffinePos(frame, l._pureLogo)
        # ll.extLegoLogo(l._image)
        draw_potential_logo_box(l)


        cv2.imshow('frame', l._mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()