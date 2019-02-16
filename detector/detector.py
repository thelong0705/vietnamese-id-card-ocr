import numpy as np
import cv2
import imutils
from transform import four_point_transform

image_path = 'test_images/image5.jpg'

img = cv2.imread(image_path)
left, right, top, bottom = 364, 1371, 259, 886
img = img[top:bottom, left:right]
height, width, channels = img.shape
print(height,width)
if height > 500:
    ratio = img.shape[0] / 500.0
    img = imutils.resize(img, height=500)
orig = img.copy()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)


cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)[:5]
c = cnts[0]
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)
screenCnt = approx
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", img)
warped = four_point_transform(orig, screenCnt.reshape(4, 2))
cv2.imshow("Scanned", warped)
cv2.waitKey(0)
