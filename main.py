import argparse
import cv2
import sys
from cropper.cropper import crop_card
from detector.detector import get_information
from PIL import Image
import pytesseract
import os
import re
import matplotlib.pyplot as plt


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
warped = crop_card(args['image'])
group_tuple = get_information(warped)

for index, t in enumerate(group_tuple):
    language = 'vie'
    if index == 0 or index == 2:
        language = 'eng'

    filename = "temp.png"
    cv2.imwrite(filename, t)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=language, config='--psm 7')
    if index == 2:
        numbers = re.findall(r'\d+', text)
        text = '/'.join(numbers)
    if index == 3:
        text = text.split()[-1]
    os.remove(filename)
    print(text)
