import argparse
import cv2
from cropper.cropper import crop_card
from detector.detector import detect_info
from reader import reader
from PIL import Image
import pytesseract


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


warped = crop_card('test/{}.jpg'.format(31))
number_img, name_img, dob_img, gender_img,\
    nation_img, country_img_list, address_img_list = detect_info(warped)
reader.get_id_numbers_text(number_img)
reader.get_text(name_img)
reader.get_dob_text(dob_img)
reader.get_gender_text(gender_img)
reader.get_nation_text(nation_img)
reader.process_list_img(country_img_list)
reader.process_list_img(address_img_list)
