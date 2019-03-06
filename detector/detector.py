import numpy as np
import cv2
import imutils
import argparse


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


def grouper(iterable, index, threshold):
    prev = ()
    group = []
    for item in iterable:
        if not prev or item[index] - prev[index] <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
image_path = args['image']
img = cv2.imread(image_path)
img = get_important_part(img)
h, w, _ = img.shape
if h > 500:
    ratio = h/500
    img = imutils.resize(img, height=500)

kernel = np.ones((25, 25), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_OTSU)[1]
kernel = np.ones((1, w), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# show_img(closing)
_, cnts, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
locs = []
for contour in cnts:
    x, y, w, h = cv2.boundingRect(contour)
    locs.append((x, y, w, h))

locs.sort(key=lambda tup: tup[1])
id_group = locs[0]
name_group = locs[1]
dob_group = locs[2]
gender_and_nation_group = locs[3]
country_group = locs[4]
address_first = locs[5]
address_second = locs[6]
