import cv2
import numpy as np
import imutils
#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path
image_path = 'test_images/image4.jpg'

img = cv2.imread(image_path)
# left, right, top, bottom = 364, 1371, 259, 886
left, right, top, bottom = 414, 1233, 300, 879

img = img[top:bottom, left:right]
height, width, channels = img.shape

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)
