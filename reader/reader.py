import cv2
import pytesseract
import numpy as np
from PIL import Image
import copy


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


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


for i in range(1, 14):
    img = cv2.imread('id_{}.png'.format(i))
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
        if box[3] < 0.5 * height:
            boxes.remove(box)
    boxes.sort(key=lambda t: t[0])
    text = ''
    for box in boxes:
        x, y, w, h = box
        number = thresh[0:height, x-2:x+w+2]
        text = text + get_each_number(number, img[0:height, x-2:x+w+2])
    print(text[-12:])
