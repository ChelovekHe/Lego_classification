
# import the necessary packages
import os
import barcode_detect as bd
import logo_detect as ld
import numpy as np
import cv2

p1 = os.listdir('../fig')
p2 = os.listdir('../fig/' + p1[3] +'/')

# load the image and convert it to grayscale
image = cv2.imread('../fig/'+ p1[3] + '/' + p2[3])
# image = cv2.imread('/Users/harrysocool/Github/package_identification/1.pic_hd.jpg')
image = cv2.resize(image,(0,0),fx = 0.1,fy = 0.1)

bar_box = bd.barcode_detect(image)
logo_box = ld.logo_detect(image)
logo = ld.getLogo()

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [bar_box,logo_box], -1, (0, 255, 0), 3)
cv2.imshow("Image1",image)
cv2.imshow("Image2",logo)
cv2.waitKey(0)