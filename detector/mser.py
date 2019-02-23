import cv2
import numpy as np
import imutils
#Create MSER object
mser = cv2.MSER_create(_delta=4, _max_area=1000)

#Your image path i-e receipt path
image_path = 'cropped.png'

img = cv2.imread(image_path)
# left, right, top, bottom = 364, 1371, 259, 886
# left, right, top, bottom = 414, 1233, 300, 879

# img = img[top:bottom, left:right]

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()


regions, bboxs = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
index = 0 
for contour in hulls:
    index += 1
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

im2, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

