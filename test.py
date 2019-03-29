import argparse
import cv2
import sys
import numpy as np
import imutils
import tensorflow as tf
import statistics
import copy
from matplotlib import pyplot as plt
from cropper.transform import four_point_transform
import pytesseract
from PIL import Image
from skimage.filters import threshold_local
from itertools import combinations


def get_information(img):
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
    id_img = get_part(id_group, orig, ratio)


def draw_rec(list_rec_tuple, img, ratio=1):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = tuple(int(ratio * l) for l in rec_tuple)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


def test_draw_rec(list_rec_tuple, img):
    list_rec_tuple.sort(key=lambda tup: tup[1])
    height, width, _ = img.shape
    list_info = []
    for index, l in enumerate(list_rec_tuple):
        x, y, w, h = l
        y = y - 20
        if index != len(list_rec_tuple) - 1:
            x1, y1, _, _ = list_rec_tuple[index+1]
            list_info.append((x, y, width, y1))
        else:
            list_info.append((x, y, width, height))
            # cv2.rectangle(img, (x, y), (width, height-5), (255, 0, 0), 2)
    return list_info


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def plot_img(img):
    plt.imshow(img)
    plt.show()


def cropout_unimportant_part(img):
    h, w, _ = img.shape
    img = img[int(0.25*h):h, int(0.35*w):w]
    return img


def get_part(locs, img, ratio):
    x, y, w, h = tuple(int(ratio * l) for l in locs)
    img = img[y-5: y+h+5, x:x+w]
    return img


def resize_img(img):
    h, w, _ = img.shape
    max_dim = min(h, w)
    ratio = h/500
    img = imutils.resize(img, height=500)
    return (img, ratio)


def crop_label(img):
    h, w, _ = img.shape
    img = img[0:h, 0:int(0.05 * w)]
    return img


def process_name(img, lozs, height):
    x0, y0, x1, y1 = lozs
    img = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, height))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    _, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    max_locs = max(locs, key=lambda tup: tup[2] * tup[3])
    x, y, w, h = max_locs
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def cut_name(img, lul):
    x0, y0, x1, y1 = lul
    img = img[y0:y1, x0:x1]
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs = remove_name_label(locs, width)
    locs = remove_smaller_area(locs, width)
    x, y, w, h = find_max_box(locs)
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def remove_smaller_area(group, width):
    avg = statistics.median(map(lambda t: t[-1] * t[-2], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[-1] * element[-2] < 0.5 * avg and element[0] < width/4:
            group.remove(element)
    return group


def get_max_box(group):
    xmin, ymin, _, _ = min(group, key=lambda tup: tup[0])
    xmax, _, w, _ = max(group, key=lambda tup: tup[0]+tup[2])
    xmax = xmax + w
    return xmin, ymin, xmax


def remove_name_label(group, width):
    avg = statistics.mean(map(lambda t: t[1], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[1] < avg and element[0] < width/4:
            group.remove(element)
    return group


def main(i):
    img = cv2.imread("result/{}_n.jpg".format(i))
    img = cropout_unimportant_part(img)
    orig = img.copy()
    img, ratio = resize_img(img)
    label_img = crop_label(img)
    h, w, _ = label_img.shape
    gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, w), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=2)
    _, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs.sort(key=lambda tup: tup[2] * tup[3], reverse=True)
    locs = locs[:5]
    list_info = test_draw_rec(locs, img)
    # x0, y0, x1, y1 = process_name(img, list_info[0], 5)
    # x0, y0, x1, y1 = cut_name(img, (x0, y0, x1, y1))
    # x0, y0, x1, y1 = tuple(int(ratio * l) for l in (x0, y0, x1, y1))
    # show_img(orig[y0:y1, x0:x1])
    # to_text(orig[y0:y1, x0:x1])

    # x0, y0, x1, y1 = process_name(img, list_info[1], 5)
    # x0, y0, x1, y1 = tuple(int(ratio * l) for l in (x0, y0, x1, y1))
    # show_img(orig[y0:y1, x0:x1])
    # to_text(orig[y0:y1, x0:x1])

    # x0, y0, x1, y1 = process_name(img, list_info[2], 5)
    # x0, y0, x1, y1 = tuple(int(ratio * l) for l in (x0, y0, x1, y1))
    # gender = orig[y0-5:y1, x0:x1]
    # h, w, _ = gender.shape
    # gender_part = gender[0:h, 0:int(w/3)]
    # to_text(gender_part)
    # nation_part = gender[0:h, int(w/3):int(0.85*w)]
    # to_text(nation_part)
    result = brand_new_country(img, list_info[3])
    if type(result) is tuple:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result)
        show_img(orig[y0:y1, x0:x1])

    if type(result) is list and len(result) == 2:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[0])
        show_img(orig[y0:y1, x0:x1])
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[1])
        show_img(orig[y0:y1, x0:x1])

    if type(result) is list and len(result) == 1:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[0])
        show_img(orig[y0:y1, x0:x1])


def process_country(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = thresh.shape
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))

    locs.sort(key=lambda t: t[0]*2+t[1])
    x, y, w, h = locs[0]
    # if y + h > img.shape[0]*0.5:
    #     print('pass')
    #     show_img(img)
    #     return
    group_orig = copy.deepcopy(locs)
    line2 = []
    for element in group_orig:
        if element[1] > y+h:
            locs.remove(element)
            line2.append(element)
    wew = max(locs, key=lambda t: t[1]+t[3])

    _, y, _, h = wew
    wew = y+h
    xmin, ymin, xmax = get_max_box(locs)
    first_line = img[0: wew, xmin:img.shape[1]]
    show_img(first_line)
    if wew < img.shape[0]*2/3:
        print('here')
        second_line = img[wew:img.shape[0], 0:img.shape[1]]
        show_img(second_line)


def to_text(img, filename='lul.png', config='--psm 7'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite(filename, thresh)
    text = pytesseract.image_to_string(Image.open(
        filename), lang='vie', config=config)
    print(text)


def process_address(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    show_img(thresh)
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs.sort(key=lambda t: t[1])
    for l in locs:
        print(l)
    location_groups = list(grouper(locs, 1, 10))
    locs = []
    for group in location_groups:
        locs.append(find_max_box(group))
    locs.sort(key=lambda t: t[2] * t[3], reverse=True)
    locs = locs[:2]
    draw_rec(locs, img)
    show_img(img)


def new_process_img(img):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    _, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    show_img(dilation)
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs.sort(key=lambda t: t[2] * t[3], reverse=True)
    locs = locs[:2]
    draw_rec(locs, img)
    show_img(img)


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


def find_max_box(group):
    xmin = min(group, key=lambda t: t[0])[0]
    ymin = min(group, key=lambda t: t[1])[1]
    xmax_box = max(group, key=lambda t: t[0] + t[2])
    xmax = xmax_box[0] + xmax_box[2]
    ymax_box = max(group, key=lambda t: t[1] + t[3])
    ymax = ymax_box[1] + ymax_box[3]
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def brand_new_country(img, lul):
    x0, y0, x1, y1 = lul
    img = img[y0:y1, x0:x1]
    show_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    height, width = thresh.shape
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs_copy = copy.deepcopy(locs)
    for l in locs_copy:
        box_height = l[1] + l[3]
        height_lim = 0.9 * height
        if l[1] == 0 or l[1] > height_lim or box_height == height:
            locs.remove(l)
    x, y, w, h = find_max_box(locs)
    if h < 0.6 * height:
        return (x0+x, y0+y, x0+x+w, y0+y+h)
    else:
        x0, y0 = x0+x, y0+y
        crop_img = thresh[y:y+h, x:x+w]
        img = img[y:y+h, x:x+w]
        height, width = crop_img.shape
        kernel = np.ones((1, width), np.uint8)
        dilation = cv2.dilate(crop_img, kernel, iterations=1)
        _, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        locs = []
        for contour in cnts:
            locs.append(cv2.boundingRect(contour))
        if len(locs) > 1:
            locs.sort(key=lambda t: t[2]*t[3], reverse=True)
            locs = locs[:2]
            show_img(img)
            x, y, w, h = locs[0]
            first_line = (x0+x, y0+y, x0+x+w, y0+y+h)
            x, y, w, h = locs[1]
            second_line = (x0+x, y0+y, x0+x+w, y0+y+h)
            return [first_line, second_line]
        if len(locs) == 1:
            return [(x0+x, y0+y, x0+x+w, y0+y+h)]


# for i in range(1, 5):
#     print(i)
#     main(i)
main(5)
