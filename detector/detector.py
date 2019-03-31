import cv2
import imutils
import numpy as np
import statistics
import copy
import pytesseract
from PIL import Image


# def show_img(img):
#     cv2.imshow('', img)
#     cv2.waitKey(0)


def cropout_unimportant_part(img):
    h, w, _ = img.shape
    img = img[int(0.25*h):h, int(0.35*w):w]
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


def get_threshold_img(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def get_contour_boxes(img):
    _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes = []
    for cnt in cnts:
        contour_boxes.append(cv2.boundingRect(cnt))
    return contour_boxes


def get_info_list(img, contour_boxes):
    contour_boxes.sort(key=lambda tup: tup[1])
    height, width, _ = img.shape
    list_info = []
    for index, l in enumerate(contour_boxes):
        x, y, w, h = l
        y = y - 20
        if index != len(contour_boxes) - 1:
            x1, y1, _, _ = contour_boxes[index+1]
            list_info.append((x, y, width, y1))
        else:
            list_info.append((x, y, width, height))
    return list_info


def get_main_text(img, box, kernel_height):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh = get_threshold_img(img, kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (thresh.shape[1], kernel_height))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    contour_boxes = get_contour_boxes(dilation)
    max_box = max(contour_boxes, key=lambda tup: tup[2] * tup[3])
    x, y, w, h = max_box
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def remove_name_label(group, width):
    avg = statistics.mean(map(lambda t: t[1], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[1] < avg and element[0] < width/4:
            group.remove(element)
    return group


def remove_smaller_area(group, width):
    avg = statistics.median(map(lambda t: t[-1] * t[-2], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[-1] * element[-2] < 0.5 * avg and element[0] < width/4:
            group.remove(element)
    return group


def find_max_box(group):
    xmin = min(group, key=lambda t: t[0])[0]
    ymin = min(group, key=lambda t: t[1])[1]
    xmax_box = max(group, key=lambda t: t[0] + t[2])
    xmax = xmax_box[0] + xmax_box[2]
    ymax_box = max(group, key=lambda t: t[1] + t[3])
    ymax = ymax_box[1] + ymax_box[3]
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def get_name(img, box):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    height, width, _ = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh_img = get_threshold_img(img, kernel)
    contour_boxes = get_contour_boxes(thresh_img)
    contour_boxes = remove_name_label(contour_boxes, width)
    contour_boxes = remove_smaller_area(contour_boxes, width)
    x, y, w, h = find_max_box(contour_boxes)
    return (x0+x, y0+y, x0+x+w, y0+y+h)


def get_img_from_box(orig, ratio, box):
    x0, y0, x1, y1 = tuple(int(ratio * element) for element in box)
    return orig[y0:y1, x0:x1]


def get_text_from_two_lines(img, box):
    x0, y0, x1, y1 = box
    img = img[y0:y1, x0:x1]
    kernel = np.ones((25, 25), np.uint8)
    thresh = get_threshold_img(img, kernel)
    height, width = thresh.shape
    contour_boxes = get_contour_boxes(thresh)
    boxes_copy = copy.deepcopy(contour_boxes)
    for box in boxes_copy:
        box_height = box[1] + box[3]
        height_lim = 0.9 * height
        if box[1] == 0 or box[1] > height_lim or box_height == height:
            contour_boxes.remove(box)
    x, y, w, h = find_max_box(contour_boxes)
    if h < 0.6 * height:
        return (x0+x, y0+y, x0+x+w, y0+y+h)
    else:
        crop_img = thresh[y:y+h, x:x+w]
        img = img[y:y+h, x:x+w]
        height, width = crop_img.shape
        kernel = np.ones((1, width), np.uint8)
        dilation = cv2.dilate(crop_img, kernel, iterations=1)
        locs = get_contour_boxes(dilation)
        if len(locs) > 1:
            x0, y0 = x0+x, y0+y
            locs.sort(key=lambda t: t[2]*t[3], reverse=True)
            locs = locs[:2]
            locs.sort(key=lambda t: t[1])
            x, y, w, h = locs[0]
            first_line = (x0+x, y0+y, x0+x+w, y0+y+h)
            x, y, w, h = locs[1]
            second_line = (x0+x, y0+y, x0+x+w, y0+y+h)
            return [first_line, second_line]
        if len(locs) == 1:
            return [(x0+x, y0+y, x0+x+w, y0+y+h)]


def process_result(orig, ratio, result):
    if type(result) is tuple:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result)
        return [orig[y0:y1, x0:x1]]
    if type(result) is list and len(result) == 2:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[0])
        first_line = orig[y0:y1, x0:x1]
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[1])
        second_line = orig[y0:y1, x0:x1]
        return [first_line, second_line]
    if type(result) is list and len(result) == 1:
        x0, y0, x1, y1 = tuple(int(ratio * l) for l in result[0])
        return [orig[y0:y1, x0:x1], None]


def detect_info(img):
    img = cropout_unimportant_part(img)
    orig = img.copy()
    img, ratio = resize_img(img)
    label_img = crop_label(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    threshold_img = get_threshold_img(label_img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (label_img.shape[1], 5))
    dilation = cv2.dilate(threshold_img, kernel, iterations=2)
    contour_boxes = get_contour_boxes(dilation)
    contour_boxes.sort(key=lambda t: t[2] * t[3], reverse=True)
    contour_boxes = contour_boxes[:5]
    info_list = get_info_list(img, contour_boxes)
    # get number part
    x, y, _, _ = info_list[0]
    number_box = (0, 0, img.shape[1], info_list[0][1])
    number_box = get_main_text(img, number_box, 5)
    number_img = get_img_from_box(orig, ratio, number_box)
    # show_img(number_img)
    # get name part
    name_box = info_list[0]
    name_box = get_name(img, get_main_text(img, name_box, 5))
    name_img = get_img_from_box(orig, ratio, name_box)
    # show_img(name_img)
    # get dob part
    dob_box = info_list[1]
    dob_box = get_main_text(img, dob_box, 5)
    dob_img = get_img_from_box(orig, ratio, dob_box)
    # show_img(dob_img)
    # get gender_and national part
    gender_and_nationality_box = info_list[2]
    gender_and_nationality_box = get_main_text(
        img, gender_and_nationality_box, 5)
    gender_n_nation_img = get_img_from_box(
        orig, ratio, gender_and_nationality_box)
    h, w, _ = gender_n_nation_img.shape
    gender_img = gender_n_nation_img[0:h, 0:int(w/3)]
    nation_img = gender_n_nation_img[0:h, int(w/3):int(0.85*w)]
    # show_img(gender_img)
    # show_img(nation_img)
    # get country part
    country_box = info_list[3]
    result = get_text_from_two_lines(img, country_box)
    country_img_list = process_result(orig, ratio, result)
    address_box = info_list[4]
    result = get_text_from_two_lines(img, address_box)
    address_img_list = process_result(orig, ratio, result)
    return number_img, name_img, dob_img, gender_img, nation_img, country_img_list, address_img_list
