import argparse
import cv2
import sys
from cropper.cropper import crop_card
from detector.detector import detect_info
from PIL import Image
import pytesseract
import os
import re
import matplotlib.pyplot as plt


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)

warped = crop_card('test/{}.jpg'.format(3))
detect_info(warped)

# for index, t in enumerate(group_tuple):
#     language = 'vie'
#     if index == 0 or index == 2:
#         language = 'eng'

#     filename = "{}.png".format(index)
#     cv2.imwrite(filename, t)
#     text = pytesseract.image_to_string(Image.open(
#         filename), lang=language, config='--psm 7')
#     if index == 2:
#         numbers = re.findall(r'\d+', text)
#         text = '/'.join(numbers)
#     if index == 3:
#         text = text.split()[-1]
#     # os.remove(filename)
#     print(text)
