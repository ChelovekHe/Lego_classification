# import the necessary packages
import os
import numpy as np
import cv2

p1 = os.listdir('../fig')
p2 = os.listdir('../fig/' + p1[1] +'/')

# load the image and convert it to grayscale
image = cv2.imread('../fig/'+ p1[1] + '/' + p2[5])
image = cv2.resize(image,(0,0),fx = 0.2,fy = 0.2)

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red1 = np.array([0,50,50])
upper_red1 = np.array([15,255,255])

lower_red2 = np.array([165,50,50])
upper_red2 = np.array([179,255,255])


# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

_, contours, _ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
new_contours = []
for idx, contour in enumerate(contours):
    if idx > 5:
        break
    # moment = cv2.moments(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    if ((np.sqrt(area)*4 <= perimeter*1.1) & (np.sqrt(area)*4 >= perimeter*0.9)):
        new_contours.append(contour)

cnt = sorted(new_contours, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the contour
rect = cv2.minAreaRect(cnt)
box = np.int0(cv2.boxPoints(rect))
print(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

cv2.imshow("Image", image)
cv2.waitKey(0)
#

#
# # draw a bounding box arounded the detected barcode and display the
# # image
# cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
