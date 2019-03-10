import argparse
import cv2
import sys
from cropper.cropper import crop_card
from detector.detector import get_information
from PIL import Image
import pytesseract
import os


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
warped = crop_card(args['image'])
group_tuple = get_information(warped)

index = 0
for t in group_tuple:
    language = 'vie'
    if index == 0 or index == 2:
        language = 'eng'
    if index != 0:
        t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    filename = "xam.png"
    cv2.imwrite(filename, t)
    text = pytesseract.image_to_string(Image.open(
        filename), lang=language, config='--psm 7')
    os.remove(filename)
    print(text)
    index = index + 1
