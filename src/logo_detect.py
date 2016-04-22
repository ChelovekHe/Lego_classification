# import the necessary packages
import numpy as np
import cv2


def logo_detect(image):
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
        if idx > 5:
            break
        # moment = cv2.moments(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if ((np.sqrt(area) * 4 <= perimeter * 1.1) & (np.sqrt(area) * 4 >= perimeter * 0.9)):
            new_contours.append(contour)

    cnt = sorted(new_contours, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the contour
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))

    print(rect[2])

    if (rect[2] < -1):
        rotate_angle = rect[2] + 90
    else:
        rotate_angle = rect[2]

    # rotate the image
    M = cv2.getRotationMatrix2D(tuple(box[2]), rotate_angle, 1)
    global rot_img
    rot_img = cv2.warpAffine(image, M, (cols, rows))

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    # todo still confusing about the coordination
    cols_st = box[2][1]
    cols_ed = cols_st + rect[1][1]
    rows_st = box[2][0]
    rows_ed = rows_st + rect[1][0]
    # print([cols_st,cols_ed, rows_st,rows_ed])

    global logo
    logo = rot_img[cols_st:cols_ed, rows_st:rows_ed]
    return box


def getLogo():
    return logo


def getRotatedImage():
    return rot_img
