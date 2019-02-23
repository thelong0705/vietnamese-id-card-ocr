import numpy as np
import cv2
import imutils
from transform import four_point_transform
# image_path = 'test_images/image5.jpg'
image_path = 'test_images/8.png'
img = cv2.imread(image_path)
# left, right, top, bottom = 364, 1371, 259, 886
# img = img[top:bottom, left:right]
height, width, channels = img.shape
if height > 500:
    img = imutils.resize(img, height=500)

orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)[:5]
c = cnts[0]

peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)
screenCnt = approx
warped = four_point_transform(orig, screenCnt.reshape(4, 2))
cv2.imwrite("test2.png",warped)



























# number_img = warped[int(0.25*height):int(0.37*height), int(0.46*width):int(0.86*width)]
# name_img = warped[int(0.4*height):int(0.5*height), int(0.46*width):int(width)]
# dob = warped[int(0.52*height):int(0.6*height), int(0.61*width):int(0.77*width)]
# sex = warped[int(0.58*height):int(0.68*height), int(0.45*width):int(0.55*width)]
# nationality = warped[int(0.58*height):int(0.68*height), int(0.72*width):int(width)]
# country = warped[int(0.7*height):int(0.8*height), int(0.55*width):int(width)]
# address1 = warped[int(0.8*height):int(0.87*height), int(0.55*width):int(width)]
# address2 = warped[int(0.87*height):int(0.95*height), int(0.4*width):int(width)]


# cv2.imwrite('number.png', number_img)
# cv2.imwrite('name.png', name_img)
# cv2.imwrite('dob.png', dob)
# cv2.imwrite('sex.png', sex)
# cv2.imwrite('nationality.png', nationality)
# cv2.imwrite('country.png', country)
# cv2.imwrite('address1.png', address1)
# cv2.imwrite('address2.png', address2)

