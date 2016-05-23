from imgPreprocessing import LogoAffinePos
from extend import get_box_serials, cv2, os, gray_image

info_size = 80


def initial_lyu_class():
    img_path = '../fig/'
    logoTp = cv2.imread(img_path + 'purelogo256.png')
    lyu = LogoAffinePos(logoTp, featureObject=cv2.AKAZE_create(), matcherObject=cv2.BFMatcher(),
                        matchMethod='knnMatch')
    return lyu


def get_affined_info(lyu, image):
    logoContourPts, cPts, affinedcPts, affinedImg, croped, rtnFlag = lyu.rcvAffinedAll(image)
    if (rtnFlag is True):
        lyu_info = croped
        if croped is not None:
            gray = gray_image(croped)
            lyu_info = cv2.resize(gray, (info_size, info_size))
        return lyu_info


def get_label_info_from_picture():
    box_serial_list = get_box_serials()
    lyu = initial_lyu_class()

    list1 = os.listdir('../fig')
    for i in range(1, len(list1)-1):
        list2 = os.listdir('../fig/' + list1[i])
        count = 1
        for j in range(0, len(list2)):
            img = cv2.imread('../fig/' +list1[i]+'/'+list2[j])
            info = get_affined_info(lyu, img)
            if info is not None:
                box_serial = box_serial_list[i-1]
                path = '../fig_sample/box_info/'
                if not os.path.exists(path):
                    os.mkdir(path)
                # elif os.path.exists(path):
                #     os.rmdir(path)
                #     os.mkdir(path)
                cv2.imwrite(path+box_serial+'.'+str(count)+'.jpg', info)
                count += 1

if __name__ == '__main__':
    get_label_info_from_picture()

