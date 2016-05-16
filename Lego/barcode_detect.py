# import the necessary packages
import numpy as np
import cv2


def barcode_detect(img):
    image = img
    # load the image and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelgx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sobelgy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

    # subtract the y-gradient from the x-gradient
    gradient = sobelgx - sobelgy
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (3, 3))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (_, contours, _) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    return box
