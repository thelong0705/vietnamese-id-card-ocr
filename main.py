import argparse
import cv2
from cropper.cropper import crop_card
from detector.detector import detect_info
from reader import reader


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


warped = crop_card('test/random.jpg')
number_img, name_img, dob_img, gender_img, nation_img, country_img_list, address_img_list = detect_info(
    warped)
number_text = reader.get_id_numbers_text(number_img)
name_text = reader.get_text(name_img)
dob_text = reader.get_dob_text(dob_img)
gender_text = reader.get_gender_text(gender_img)
nation_text = reader.get_nation_text(nation_img)
country_text = reader.process_list_img(country_img_list, is_country=True)
address_text = reader.process_list_img(address_img_list, is_country=False)
print(number_text)
print(name_text)
print(dob_text)
print(gender_text)
print(nation_text)
print(country_text)
print(address_text)
