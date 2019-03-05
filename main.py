import argparse
import cv2
import sys
from cropper.cropper import crop_card
from detector.detect_location import get_each_roi


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
warped = crop_card(args['image'])
id_part, name_part = get_each_roi(warped)
show_img(id_part)
