# import the necessary packages
import numpy as np
import cv2
import extLogo


class Lego(object):
    def __init__(self, image):
        # initialize attributes
        self._pureLogo = cv2.imread('../fig/purelogo128.png')
        self._image = image
        self._rotated_image = None
        self._rotate_angle = None
        self._information = None
        self._logo = None
        self._logo_box = None
        self._hasValidLogo = False

        self._logo_detect(image)
        if self._hasValidLogo:
            try:
                self._get_rotate_angle()
                self._get_rotated_image()
            except:
                pass

    def _logo_detect(self,image):
        self._hasValidLogo = False

        rows, cols, _ = image.shape

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # define range of red color in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])

        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([179, 255, 255])

        # Threshold the HSV image to get only red colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        new_contours = []
        for idx, contour in enumerate(contours):
            if idx >= 5:
                break
            # moment = cv2.moments(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if (np.sqrt(area) * 4 <= perimeter * 1.2) & (np.sqrt(area) * 4 >= perimeter * 0.8):
                new_contours.append(contour)

        if len(new_contours) >= 1:
            cnt = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]

            # compute the rotated bounding box of the contour
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))

            xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
            yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])
            cropst = np.array([yaxis.min()-10, xaxis.min()-10])
            croped = np.array([yaxis.max()+10, xaxis.max()+10])

            self._logo = self._image[cropst[0]:croped[0], cropst[1]:croped[1]]
            self._logo_box = box

            # when the detect logo is square, set flag true
            height, width, _ = self._logo.shape
            if abs(height - width) < 0.1*np.mean([height,width]):
                self._hasValidLogo = True



    def _get_rotate_angle(self):
        akaze = cv2.AKAZE_create()

        gray_image1 = cv2.cvtColor(self._pureLogo,cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(self._logo,cv2.COLOR_BGR2GRAY)

        kp1, des1 = akaze.detectAndCompute(gray_image1, None)
        kp2, des2 = akaze.detectAndCompute(gray_image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        MIN_MATCH_COUNT = 6
        MAX_MATCH_COUNT = 10
        if len(good_matches) >= MIN_MATCH_COUNT & len(good_matches) <= MAX_MATCH_COUNT :
            src_pts = np.float64([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float64([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            Ang = extLogo.__calcuAngle__(src_pts,dst_pts)
            if np.isnan(Ang):
                self._hasValidLogo = False
            else:
                self._rotate_angle = Ang/np.pi*180

        # im3 = cv2.drawMatchesKnn(self._pureLogo, kp1, self._logo, kp2, good_matches, None, flags=2)
        # cv2.imshow("AKAZE matching", im3)
        # cv2.waitKey(0)

    def get_information_part(self):
        self._logo_detect(self._rotated_image)
        height, _, _ = self._logo.shape
        box = self._logo_box
        xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
        yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])
        cropst = np.array([yaxis.min()-10, xaxis.min()-10])
        croped = np.array([yaxis.max()+height+10, xaxis.max()+10])

        self._information = self._rotated_image[cropst[0]:croped[0], cropst[1]:croped[1]]

    def _get_rotated_image(self):
        if self._hasValidLogo:
            imgH,imgW,_ = self._image.shape
            M = cv2.getRotationMatrix2D((imgW/2, imgH/2), self._rotate_angle, 1)
            self._rotated_image = cv2.warpAffine(self._image,M,(imgW,imgH))

    def get_logo_box(self):
        if self._hasValidLogo:
            return self._logo_box
        else:
            return None

    def get_logo_image(self):
        if self._hasValidLogo:
            return self._logo
        else:
            return None

    # def getBarcodeBox(self):
    #     if hasattr(self, '_barcode_box'):
    #         return self._barcode_box
    #     else:
    #         return None
    #         print('No barcode detected')
