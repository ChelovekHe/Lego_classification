from Lego.imgPreprocessing import LogoAffinePos, imgFilter
from Lego.extend import *
import cv2
from PIL import Image
import numpy as np
import cnn
from Lego.ocr import tesserOcr
import Lego.dsOperation as dso
from Lego.RandomForestOCR import RandomFtestOCR

FRAME_SIZE_FACTOR = 0.4
info_size = 30
logo_box = None
lyu_info = None


def initial_lyu_class():
    img_path = '../fig/'
    logo = cv2.imread(img_path + 'purelogo256.png')
    lyu = LogoAffinePos(logo, featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(),
                        matchMethod='knnMatch')
    return lyu


def get_affined_image(lyu, image):
    logoContourPts, cPts, affinedcPts, affinedImg, croped, rtnFlag = lyu.rcvAffinedAll(image)
    if (rtnFlag is True):
        lyu_info = croped
        global logo_box
        logo_box = cPts
        return lyu_info

def read_data():
    settingInfo = open('../data/setting', 'r')
    settingInfo.readline()
    PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    DATAPATH = settingInfo.readline().strip().lstrip().rstrip(',')
    FEATURE_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    MATERIAL_IMG_FOLDER = settingInfo.readline().strip().lstrip().rstrip(',')
    BOX_DATA_PATH = settingInfo.readline().strip().lstrip().rstrip(',')
    LOG_PATH = settingInfo.readline().strip().lstrip().rstrip(',')

    boxesds = dso.dsRead(BOX_DATA_PATH)
    return  boxesds


if __name__ == '__main__':
    model = cnn.initial_cnn_model(5)
    model.load_weights('../data/lego_identify.h5')
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()

        if ret is True:
            lyu = initial_lyu_class()
            try:
                lyu_info = get_affined_image(lyu, frame.copy())
            except:
                pass
            if lyu_info is not None:
                filtratedCroped = imgFilter(lyu_info)
                filtratedCroped = cv2.cvtColor(filtratedCroped, cv2.COLOR_GRAY2RGB)
                filtratedCroped = Image.fromarray(filtratedCroped)
                numStr = tesserOcr(filtratedCroped)
                boxesds = read_data()
                matched = numMatch(boxesds, numStr)

                str = RandomFtestOCR(lyu_info)

                lyu_info = gray_image(lyu_info)
                cv2.imshow('info', lyu_info)
                cv2.moveWindow('info', int(1280 * FRAME_SIZE_FACTOR), 0)
                gray = cv2.resize(lyu_info, (info_size, info_size))
                arr = np.asarray(gray, dtype='float32').reshape((1, 1, 30, 30))
                arr /= np.max(arr)
                arr -= np.std(arr)
                predict = model.predict(arr, batch_size=1)
                box_serials = get_box_serials()
                if (predict is not None) & (matched is not None):
                    predict_class1, predict_class2 = combine_results(predict, matched)
                    print(box_serials[predict_class1], box_serials[predict_class2], str)


            cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)
            frame = resize(frame, FRAME_SIZE_FACTOR)
            cv2.imshow('frame', frame)

            if get_keyboard():
                break
    cap.release()
    cv2.destroyAllWindows()

