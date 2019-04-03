import cv2
import pytesseract
import numpy as np
from PIL import Image
import copy
import re


# def show_img(img):
#     cv2.imshow('', img)
#     cv2.waitKey(0)


def get_each_number(img, normal_img):
    filename = 'temp.png'
    config = '--oem 0  --psm 10 -c tessedit_char_whitelist=1234567890'
    lang = 'eng'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    if not text:
        cv2.imwrite(filename, normal_img)
        text = pytesseract.image_to_string(Image.open(
            filename), lang=lang, config=config)
        if not text:
            text = '?'
    return text


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


def draw_rec(list_rec_tuple, img, ratio=1):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = tuple(int(ratio * l) for l in rec_tuple)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


def get_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'vie'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    print(text)


def get_dob_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'eng'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
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
    print(numbers)


def get_gender_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'vie'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    capital_n_index = text.find('N')
    if capital_n_index != -1:
        gender_text = text[capital_n_index:]
        a_character = ['a', 'A', 'ă', 'â']
        if gender_text[1] and gender_text[1] in a_character:
            gender_text = 'Nam'
            print(gender_text)
            return
    gender_text = 'Nữ'
    print(gender_text)
    return


def get_nation_text(img):
    filename = 'temp.png'
    config = '--psm 7'
    lang = 'vie'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    colon_index = text.find(':')
    if colon_index != -1:
        text = text[colon_index+1:]
        text = text.strip()
    else:
        words = text.split()
        for index, word in enumerate(words):
            if index != 0 and word[0].isupper():
                text = words[index:]
                break
        text = ' '.join(text)
    print(text)


def get_id_numbers_text(img):
    height_img, width_img, _ = img.shape
    kernel = np.ones((height_img, height_img), np.uint8)
    if height_img < 20:
        img = cv2.resize(
            img, (2*width_img, 2*height_img), interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((height_img * 2, height_img*2), np.uint8)
    thresh = get_threshold_img(img, kernel)
    height, width = thresh.shape
    boxes = get_contour_boxes(thresh)
    boxes_copy = copy.deepcopy(boxes)
    for box in boxes_copy:
        if box[3] < 0.4 * height:
            boxes.remove(box)
    boxes.sort(key=lambda t: t[0])
    text = ''
    for box in boxes:
        x, y, w, h = box
        number = thresh[0:height, x-2:x+w+2]
        text = text + get_each_number(number, img[0:height, x-2:x+w+2])
    print(text[-12:])


def strip_label_and_get_text(img, config='--psm 7'):
    filename = 'temp.png'
    lang = 'vie'
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=lang, config=config)
    colon_index = text.find(':')
    if colon_index != -1:
        text = text[colon_index+1:]
        text = text.strip()
    else:
        words = text.split()
        for index, word in enumerate(words):
            if index != 0 and (word[0].isupper() or word[0].isdigit()):
                text = words[index:]
                break
        text = ' '.join(text)
    print(text)
    return text


def process_list_img(img_list):
    if len(img_list) == 1:
        strip_label_and_get_text(img_list[0])
        return
    if len(img_list) == 2 and img_list[1] is not None:
        strip_label_and_get_text(img_list[0])
        get_text(img_list[1])
    else:
        strip_label_and_get_text(img_list[0], config='')


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
