import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import subprocess
import copy
import statistics
import math
from unidecode import unidecode
from util.util import get_threshold_img, get_contour_boxes, run_item, gather_results


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def get_each_number(img, normal_img):
    config = '--oem 0  --psm 10 -c tessedit_char_whitelist=1234567890'
    lang = 'eng'
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    if not text:
        text = pytesseract.image_to_string(
            normal_img, lang=lang, config=config)
        if not text:
            text = '?'
    return text


def get_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'vie'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    if not text:
        subprocess.call(["./resources/textcleaner", "-g", "-e", "normalize",
                         "-o", "11", "-t", "5", "temp.png", "temp.png"])
        text = pytesseract.image_to_string(Image.open(
            filename), lang=lang, config=config)
    text = text.strip('. :')
    return text


def get_dob_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'eng'
    h, w, _ = img.shape
    if h < 25:
        ratio = math.ceil(25/h)
        img = cv2.resize(img, None, fx=ratio, fy=ratio,
                         interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    date = re.findall(r'\d{2}/\d{2}/\d{4}', text)
    if date:
        numbers = re.findall(r'\d', date[0])
    else:
        numbers = re.findall(r'\d', text)
    numbers = numbers[-8:]
    if len(numbers) < 8:
        numbers.extend(['?' for i in range(8-len(numbers))])
    if numbers[4] != '2':
        numbers[4] = '1'
        numbers[5] = '9'
        if numbers[6] == '0':
            numbers[6] = '9'
    numbers[2:2] = ['/']
    numbers[5:5] = ['/']
    numbers = ''.join(numbers)
    return numbers


def get_gender_text(img):
    text = get_text(img)
    capital_n_index = text.find('N')
    if capital_n_index != -1:
        gender_text = text[capital_n_index:]
        a_character = ['a', 'A', 'ă', 'â']
        if (gender_text[1] and gender_text[1] in a_character) or gender_text[-1] == 'm':
            gender_text = 'Nam'
            return gender_text
    gender_text = 'Nữ'
    return gender_text


def get_nation_text(img):
    text = get_text(img)
    colon_index = text.find(':')
    if colon_index != -1 and colon_index < len(text)/2:
        text = text[colon_index+1:]
    else:
        words = text.split()
        for index, word in enumerate(words):
            if index != 0 and word[0].isupper():
                text = words[index:]
                break
        text = ' '.join(text)
    text = text.strip('. :')
    return text


def get_id_numbers_text(img):
    height_img, width_img, _ = img.shape
    if height_img < 20:
        img = cv2.resize(
            img, (2*width_img, 2*height_img), interpolation=cv2.INTER_CUBIC)
    height, width, _ = img.shape
    kernel = np.ones((height//2, height//2), np.uint8)
    thresh = get_threshold_img(img, kernel)
    boxes = get_contour_boxes(thresh)
    boxes_copy = copy.deepcopy(boxes)
    for box in boxes_copy:
        if box[3] < 0.4 * height:
            boxes.remove(box)
    boxes.sort(key=lambda t: t[0])
    list_number = []
    for box in boxes:
        x, y, w, h = box
        if x < 2 or x+w+2 > width:
            continue
        thresh_number = thresh[0:height, x-2:x+w+2]
        normal_number = img[0:height, x-2:x+w+2]
        list_number.append((thresh_number, normal_number))
    numbers = gather_results([run_item(get_each_number, item)
                              for item in list_number])
    text = ''.join(numbers)
    return text[-12:]


def strip_label_and_get_text(img, is_country, config='--psm 7'):
    text = get_text(img)
    colon_index = text.find(':')
    if colon_index != -1 and colon_index < len(text)/2:
        text = text[colon_index+1:]
        text = text.strip()
    else:
        for index, letter in enumerate(text):
            if is_country:
                condition = letter.isupper()
            else:
                condition = letter.isupper() or letter.isdigit()
            if index > 1 and condition:
                text = text[index:]
                break
    return text


def process_list_img(img_list, is_country):
    if len(img_list) == 1:
        return process_first_line(img_list[0], is_country)
    if len(img_list) == 2:
        line1 = process_first_line(img_list[0], is_country)
        line2 = get_text(img_list[1])
        return line1 + '\n' + line2


def process_first_line(img, is_country):
    img_h, img_w, _ = img.shape
    kernel = np.ones((25, 25), np.uint8)
    thresh = get_threshold_img(img, kernel)
    contour_boxes = get_contour_boxes(thresh)
    avg = statistics.mean(map(lambda t: t[-1] * t[-2], contour_boxes))
    boxes_copy = copy.deepcopy(contour_boxes)
    for box in boxes_copy:
        if box[-1] * box[-2] < avg/3:
            contour_boxes.remove(box)
    contour_boxes.sort(key=lambda t: t[0])
    list_distance = []
    for index, box in enumerate(contour_boxes):
        current_x = box[0]+box[2]
        if index < len(contour_boxes) - 1:
            next_x = contour_boxes[index+1][0]
            list_distance.append(next_x-current_x)
    avg = statistics.mean(list_distance)
    list_copy = copy.deepcopy(list_distance)
    list_copy.sort(reverse=True)
    if len(list_copy) > 1 and list_copy[0] > 3 * list_copy[1]:
        max_index = list_distance.index(list_copy[0])
        contour_boxes = contour_boxes[max_index+1:]
        x, y, w, h = find_max_box(contour_boxes)
        img = img[0:img_h, x:img_w]
        return get_text(img)
    else:
        return strip_label_and_get_text(img, is_country)


def find_max_box(group):
    xmin = min(group, key=lambda t: t[0])[0]
    ymin = min(group, key=lambda t: t[1])[1]
    xmax_box = max(group, key=lambda t: t[0] + t[2])
    xmax = xmax_box[0] + xmax_box[2]
    ymax_box = max(group, key=lambda t: t[1] + t[3])
    ymax = ymax_box[1] + ymax_box[3]
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def fix_last_name(name):
    fname = 'resources/common_last_name.txt'
    words = name.split()
    words[0] = unidecode(words[0]).upper()
    with open(fname) as f:
        last_name_list = f.readlines()
    last_name_list = [x.strip().upper() for x in last_name_list]
    last_name_decode = [unidecode(x) for x in last_name_list]
    if words[0] in last_name_decode:
        words[0] = last_name_list[last_name_decode.index(words[0])]
    text = ' '.join(words)
    return text


def get_name_text(img):
    name = get_text(img)
    return fix_last_name(name)
