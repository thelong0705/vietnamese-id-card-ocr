import numpy as np
import cv2
import imutils


def get_important_part(img):
    h, w, _ = img.shape
    img = img[int(0.25*h):h, int(0.35*w):w]
    return img

def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def draw_rec(list_rec_tuple, img):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = rec_tuple
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

image_path = 'test/not-normal.png'
img = cv2.imread(image_path)
h, w, _ = img.shape
if h > 500:
    img = imutils.resize(img, height=500)
img = get_important_part(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
_, contours, _ = cv2.findContours(
    edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in contours]
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
for contour in hulls:
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

_, contours, _ = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

locs = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    locs.append((x, y, w, h))

draw_rec(locs, img)
show_img(img)
