import cv2
from PIL import Image
from tesserwrap import Tesseract
from Lego.Lego import Lego
from Lego.imgPreprocessing import LogoAffinePos


def resize(image, factor=0.5):
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    return image


def ocr(info):
    info = resize(info, 0.5)
    cv2.imshow('info', info)
    cv2.imwrite('info.jpg', info)
    img = Image.open('info.jpg')
    tr = Tesseract(datadir='../tessdata', lang='eng')
    text = tr.ocr_image(img)
    return text


def compare(image):
    li = Lego(image)
    rotated = li.get_logo_image()
    print(type(rotated))
    lyu = LogoAffinePos(li._pureLogo, featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(),
                        matchMethod='knnMatch')
    try:
        logoContourPts, cPts, affinedcPts, affinedImg, rtnFlag = lyu.rcvAffinedAll(image)
        cv2.drawContours(image, [logoContourPts], -1, (0,255,0), 2)
        if (rtnFlag is True):
            image = affinedImg
    except:
        pass
    return rotated, image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()

        # info = l.get_information_part()
        # text = ocr(info)

        rotated, affined = compare(frame)

        # rotated = resize(rotated, 0.5)
        # cv2.imshow('rotate', rotated)

        affined = resize(affined, 0.5)
        cv2.imshow('frame', affined)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
