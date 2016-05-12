# import the necessary packages
import numpy as np
import cv2
import extLogo


class Lego(object):
    def __init__(self, image):
        # initialize attributes
        self._pureLogo = cv2.imread('../fig/purelogo256.png')
        self._image = image
        self._rotated_image = None
        self._rotate_angle = None
        self._information = None
        self._logo = None
        self._logo_box = None

        # Parameters
        self.MIN_MATCH_COUNT = 6
        self.MAX_MATCH_COUNT = 10

        # Flags
        self._has_rotated_image = False
        self._has_rotate_angle = False
        self._has_potential_logo = False
        self._has_valid_logo = False
        self._has_information = False

        self._logo_detect(image)
        if self._has_potential_logo:
            self._get_rotate_angle()
            if self._has_valid_logo:
                self._get_rotated_image()
            # self._get_information()

    def _logo_detect(self,image):
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

            self._has_potential_logo = True

    def _get_rotate_angle(self):
        akaze = cv2.AKAZE_create()

        gray_image1 = cv2.cvtColor(self._pureLogo, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(self._logo, cv2.COLOR_BGR2GRAY)

        kp1, des1 = akaze.detectAndCompute(gray_image1, None)
        kp2, des2 = akaze.detectAndCompute(gray_image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        if des2 != None:
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            matches = []

        good_matches = []
        if len(matches) > 1:
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)


        if (len(good_matches) >= self.MIN_MATCH_COUNT) & (len(good_matches) <= self.MAX_MATCH_COUNT):
            src_pts = np.float64([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float64([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            Ang = extLogo.__calcuAngle__(src_pts, dst_pts)
            if np.isnan(Ang):
                self._has_valid_logo = False
            else:
                self._rotate_angle = Ang/np.pi*180
                self._has_rotate_angle = True
                self._has_valid_logo = True

        # im3 = cv2.drawMatchesKnn(self._pureLogo, kp1, self._logo, kp2, good_matches, None, flags=2)
        # cv2.imshow("AKAZE matching", im3)
        # cv2.waitKey(0)

    def _get_information(self):
        self._logo_detect(self._rotated_image)
        if self._has_valid_logo & self._has_rotated_image:
            height, _, _ = self._logo.shape
            box = self._logo_box
            xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
            yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])
            cropst = np.array([box[0, 0], xaxis.min()-10])
            croped = np.array([yaxis.max()+height+10, xaxis.max()+10])

            img = self._rotated_image[cropst[0]:croped[0], cropst[1]:croped[1]]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, threshold_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self._information = threshold_image
            self._has_information = True

    def _get_rotated_image(self):
        if self._has_rotate_angle:
            imgH, imgW, _ = self._image.shape
            box = self._logo_box
            xaxis = np.array([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])
            yaxis = np.array([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])
            center_x = int(round(np.mean(xaxis),0))
            center_y = int(round(np.mean(yaxis),0))
            M = cv2.getRotationMatrix2D((center_x, center_y), self._rotate_angle, 1)
            self._rotated_image = cv2.warpAffine(self._image, M, (imgW, imgH))
            self._has_rotated_image = True

    def get_logo_box(self):
        if self._has_valid_logo:
            return self._logo_box
        else:
            return None

    def get_logo_image(self):
        if self._has_valid_logo:
            return self._logo
        else:
            return None

    def get_information_part(self):
        if self._has_information:
            return self._information
        else:
            return None

    # def getBarcodeBox(self):
    #     if hasattr(self, '_barcode_box'):
    #         return self._barcode_box
    #     else:
    #         return None
    #         print('No barcode detected')
