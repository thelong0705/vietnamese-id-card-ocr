import argparse
import cv2
import sys
from cropper.cropper import crop_card
from detector.detector import get_information


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
warped = crop_card(args['image'])
id_img = get_information(warped)
show_img(id_img)
cv2.imwrite("id.png", id_img)
