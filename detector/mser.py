import cv2
import numpy as np
import imutils
from itertools import groupby

def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def grouper(iterable):
    prev = ()
    group = []
    for item in iterable:
        if not prev or item[1] - prev[1] <= 15:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

mser = cv2.MSER_create(_delta=3, _max_area=400)
image_path = 'cropped.png'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


regions, bboxs = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

im2, contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

locs = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    locs.append((x,y,w,h))
    # print(x,y,w,h)
    # crop = img[y:y+h, x:x+w]
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
locs.sort(key=lambda tup: tup[1])
l = dict(enumerate(grouper(locs), 1))[1]
l.sort(key=lambda tup: tup[0])
l.pop(0)
xmin,ymin,_,_ = l[0]
x,y,w,h = l[-1]
xmax = x+w
ymax = y+h
cv2.rectangle(img, (xmin-5, ymin-5), (xmax+5, ymax+5), (255, 0, 0), 2)
show_img(img)
