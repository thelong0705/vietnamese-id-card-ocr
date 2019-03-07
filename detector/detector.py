import numpy as np
import cv2
import imutils
import statistics
import copy


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def get_max_box(group):
    xmin = min(group, key=lambda tup: tup[0])[0]
    xmax, _, w, _ = max(group, key=lambda tup: tup[0]+tup[3])
    xmax = xmax + w
    return xmin, xmax


def cropout_unimportant_part(img):
    h, w, _ = img.shape
    img = img[int(0.25*h):h, int(0.35*w):w]
    return img


def remove_shorter_than_med(group, med):
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[-1] <= 0.75*med:
            group.remove(element)
    return group


def process_id_part(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 20), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    _, cnts, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        locs.append((x, y, w, h))
    med = statistics.median(map(lambda t: t[-1], locs))
    number_locs = remove_shorter_than_med(locs, med)
    xmin, xmax = get_max_box(number_locs)
    id_img = img[0: img.shape[0], xmin-5:xmax]
    return id_img


def get_information(img):
    img = cropout_unimportant_part(img)
    orig = img.copy()
    h, w, _ = img.shape
    ratio = 1
    if h > 500:
        ratio = h/500
        img = imutils.resize(img, height=500)
    kernel = np.ones((25, 25), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, w), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
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
    x, y, w, h = tuple(int(ratio * element) for element in id_group)
    id_img = orig[y-5: y+h+5, x:x+w]
    id_img = process_id_part(id_img)
    return id_img
