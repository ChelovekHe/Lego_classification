from Lego.imgPreprocessing import LogoAffinePos
from Lego.extend import *
import numpy as np
import cnn

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

def get_affined_image(lyu, image):
    logoContourPts, cPts, affinedcPts, affinedImg, croped, rtnFlag = lyu.rcvAffinedAll(image)
    if (rtnFlag is True):
        lyu_info = croped
        global logo_box
        logo_box = cPts
        if lyu_info is not None:
            gray = gray_image(lyu_info)
        return lyu_info

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
                gray = cv2.resize(lyu_info, (info_size, info_size))
                arr = np.asarray(gray, dtype='float32').reshape((1, 1, 30, 30))
                arr /= np.max(arr)
                arr -= np.std(arr)
                predict = model.predict(arr, batch_size=1)
                box_serials = get_box_serials()

            cv2.drawContours(frame, [logo_box], -1, (0, 255, 0), 2)
            frame = resize(frame, FRAME_SIZE_FACTOR)
            cv2.imshow('frame', frame)

            if get_keyboard():
                break
    cap.release()
    cv2.destroyAllWindows()

