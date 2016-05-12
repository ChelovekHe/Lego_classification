import cv2
from Lego import Lego
from PIL import Image
from tesserwrap import Tesseract


VWIDTH = 640
VHIGH = 480
cap = cv2.VideoCapture(0)
ret = cap.set(3, VWIDTH)
ret = cap.set(4, VHIGH)

while 1:
    _, frame = cap.read()

    imgH,imgW, _ = frame.shape

    l = Lego(frame)
    logo_box = l.get_logo_box()
    info = l.get_information_part()

    cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)

        # cv2.imwrite('info.jpg',info)
        # img = Image.open('info.jpg')
        # tr = Tesseract(datadir='../tessdata', lang='eng')
        # text = tr.ocr_image(img)
        # print(text)


    if l._has_information:
        cv2.imshow('info', info)
    # if l._has_potential_logo:
    #     cv2.imshow('logo', l.get_logo_image())

    # print(l._logo.shape)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
