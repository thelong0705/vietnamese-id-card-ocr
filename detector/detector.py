import numpy as np
import cv2
import imutils
import statistics
import copy


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def draw_rec(list_rec_tuple, img, ratio=1):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = tuple(int(ratio * l) for l in rec_tuple)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


def resize_img(img):
    h, w, _ = img.shape
    max_dim = min(h, w)
    ratio = h/500
    img = imutils.resize(img, height=500)
    return (img, ratio)


def find_color_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    _, contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    mask = np.zeros(img.shape[:2], np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return cv2.mean(img_hsv, mask)[2]


def get_max_box(group):
    xmin, ymin, _, _ = min(group, key=lambda tup: tup[0])
    xmax, _, w, _ = max(group, key=lambda tup: tup[0]+tup[2])
    xmax = xmax + w
    return xmin, ymin, xmax


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


# def remove_smaller_area(group, avg):
#     group_orig = copy.deepcopy(group)
#     for element in group_orig:
#         if element[-1] * element[-2] < avg:
#             group.remove(element)
#     return group


def process_id_part(img):
    show_img(img)
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


def get_part(locs, img, ratio):
    x, y, w, h = tuple(int(ratio * l) for l in locs)
    img = img[y: y+h, x:x+w]
    return img


def split_half_by_distance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    # show_img(thresh)
    _, contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        locs.append([x, y, w, h])
    locs.sort(key=lambda tup: tup[0])
    max_distance = 0
    index = 0
    for idx, current_location in enumerate(locs):
        if idx >= len(locs) - 1:
            break
        next_location = locs[idx+1]
        end_box_loc = current_location[0] + current_location[2]
        diff = next_location[0] - end_box_loc
        if diff > max_distance:
            max_distance = diff
            index = idx
    first_half = locs[0:index + 1]
    second_half = locs[index + 1:]
    return first_half, second_half


def process_name(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((25, 25), np.uint8)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = thresh.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 5))
    dilation = cv2.dilate(thresh, kernel, iterations=2)
    _, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    max_locs = max(locs, key=lambda tup: tup[2] * tup[3])
    x, y, w, h = max_locs
    return img[y:y+h, x:x+w]


def process_gender(img):
    locs = split_half_by_distance(img)[0]
    xmin, xmax = get_max_box(locs)
    gender_img = img[0: img.shape[0], xmin:xmax+5]
    return gender_img


def process_country(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_threshhold = find_color_threshold(img)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([180, 255, int(color_threshhold-20)], np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    _, contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    show_img(mask)
    locs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        locs.append([x, y, w, h])
    avg = statistics.mean(map(lambda t: t[-1] * t[-2], locs))
    locs = remove_smaller_area(locs, avg)
    # draw_rec(locs,img)
    show_img(img)
    xmin, xmax = get_max_box(locs)
    xmin = xmin - 5 if xmin - 5 > 0 else xmin
    country_img = img[0: img.shape[0], xmin:xmax+30]
    return country_img


def process_first_address(img):
    return process_country(img)


def get_information(img):
    img = cropout_unimportant_part(img)
    orig = img.copy()
    h, w, _ = img.shape
    img, ratio = resize_img(img)
    kernel = np.ones((25, 25), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, w), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, cnts, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs.sort(key=lambda tup: tup[2] * tup[3], reverse=True)
    locs = locs[:7]
    draw_rec(locs, orig, ratio)
    return orig
    locs.sort(key=lambda tup: tup[1])
    id_group = locs[0]
    name_group = locs[1]
    dob_group = locs[2]
    gender_and_nation_group = locs[3]
    country_group = locs[4]
    address_first = locs[5]
    address_second = locs[6]
    id_img = get_part(id_group, orig, ratio)
    id_img = process_id_part(id_img)
    name_img = get_part(name_group, orig, ratio)
    # name_img = process_name(name_img)
    dob_img = get_part(dob_group, orig, ratio)
    gender_and_nation_img = get_part(gender_and_nation_group, orig, ratio)
    gender_img = process_gender(gender_and_nation_img)
    country_img = get_part(country_group, orig, ratio)
    # country_img = process_country(country_img)
    address_first_img = get_part(address_first, orig, ratio)
    # address_first_img = process_first_address(address_first_img)
    address_second_img = get_part(address_second, orig, ratio)
    return id_img, name_img, dob_img, gender_img, country_img, address_first_img, address_second_img


def detector(img):
    img = cropout_unimportant_part(img)
    img, ratio = resize_img(img)
    label_img = crop_label(img)
    h, w, _ = label_img.shape
    gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, w), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, cnts, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        locs.append((x, y, w, h))
    locs.sort(key=lambda tup: tup[2] * tup[3], reverse=True)
    locs = locs[:5]
    info_parts = get_info_parts(locs, img)
    name_part = info_parts[0]
    name = process_name(name_part)
    name = cut_name(name)


def crop_label(img):
    h, w, _ = img.shape
    img = img[0:h, 0:int(0.05 * w)]
    return img


def get_info_parts(list_rec_tuple, img):
    list_rec_tuple.sort(key=lambda tup: tup[1])
    height, width, _ = img.shape
    list_info = []
    for index, l in enumerate(list_rec_tuple):
        x, y, w, h = l
        y = y - 20
        if index != len(list_rec_tuple) - 1:
            x1, y1, _, _ = list_rec_tuple[index+1]
            list_info.append(img[y:y1, x:width])
        else:
            list_info.append(img[y:height, x:width])
    return list_info


def cut_name(img):
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
    locs = remove_name_label(locs)
    # draw_rec(locs,img)
    # show_img(img)
    locs = remove_smaller_area(locs)
    xmin, ymin, xmax = get_max_box(locs)
    name_img = img[ymin-10: img.shape[0], xmin:img.shape[1]]
    show_img(name_img)
    return name_img


def remove_smaller_area(group):
    avg = statistics.median(map(lambda t: t[-1] * t[-2], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[-1] * element[-2] < 0.5 * avg:
            group.remove(element)
    return group


def remove_name_label(group):
    avg = statistics.mean(map(lambda t: t[1], group))
    group_orig = copy.deepcopy(group)
    for element in group_orig:
        if element[1] < avg:
            group.remove(element)
    return group



