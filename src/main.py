# import the necessary packages
import os

import cv2
from Lego import Lego
from PIL import Image
from tesserwrap import Tesseract
import barcode_detect

p1 = os.listdir('../fig')
p2 = os.listdir('../fig/' + p1[3] + '/')

# load the image and convert it to grayscale
image = cv2.imread('../fig/' + p1[3] + '/' + p2[1])
image = cv2.imread('/Users/harrysocool/Github/package_identification/fig/1.pic_hd.jpg')
image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

l = Lego(image)

# rot_image = l.getRotatedImage()
logo_image = l.get_logo_image()

# barcode_box = l.getBarcodeBox()
logo_box = l.get_logo_box()
# logo_box = barcode.barcode(image)

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [logo_box], -1, (0, 255, 0), 2)

cv2.imshow("Image1", image)
cv2.imshow("Image2", logo_image)

# img = Image.open('/Users/harrysocool/Github/package_identification/fig/1.png')
# tr = Tesseract(datadir='../tessdata', lang='eng')
# text = tr.ocr_image(img)
# print(text)
cv2.waitKey(0)
