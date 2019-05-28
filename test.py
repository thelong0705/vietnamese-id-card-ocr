import argparse
import cv2
from cropper.cropper import crop_card
from detector.detector import detect_info
from reader import reader
from Levenshtein import distance


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def get_ground_truth(filepath):
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if cnt == 1:
                number_truth = line.strip()
            if cnt == 2:
                name_truth = line.strip()
            if cnt == 3:
                dob_truth = line.strip()
            if cnt == 4:
                gender_truth = line.strip()
            if cnt == 5:
                nation_truth = line.strip()
            if cnt == 6:
                country_truth = line.strip()
            if cnt == 7 and line.strip():
                country_truth = country_truth + '\n' + line.strip()
            if cnt == 8:
                address_truth = line.strip()
            if cnt == 9 and line.strip():
                address_truth = address_truth + '\n' + line.strip()
            line = fp.readline()
            cnt += 1
    return number_truth, name_truth, dob_truth, gender_truth, nation_truth, country_truth, address_truth


id_character = 0
name_character = 0
dob_character = 0
gender_character = 0
nation_character = 0
country_character = 0
address_character = 0

number_wrong = 0
name_wrong = 0
dob_wrong = 0
gender_wrong = 0
nation_wrong = 0
country_wrong = 0
address_wrong = 0

for i in range(50):
    print(i)
    warped = cv2.imread('test/detector_test/{}.png'.format(i))
    _, number_img, name_img, dob_img, gender_img,\
        nation_img, country_img, address_img, country_img_list, address_img_list = detect_info(
            warped)
    number_text = reader.get_id_numbers_text(number_img)
    name_text = reader.get_name_text(name_img)
    dob_text = reader.get_dob_text(dob_img)
    gender_text = reader.get_gender_text(gender_img)
    nation_text = reader.get_nation_text(nation_img)
    country_text = reader.process_list_img(country_img_list, is_country=True)
    address_text = reader.process_list_img(address_img_list, is_country=False)
    if i < 20:
        filepath = 'test/detector_test/{}.txt'.format(i)
    else:
        filepath = 'test/detector_test/{}.txt'.format(i//10*10)
    number_truth, name_truth, dob_truth, gender_truth, nation_truth, country_truth, address_truth = get_ground_truth(
        filepath)
    id_character = 12
    name_character = len(name_truth)
    dob_character = len(dob_truth)
    gender_character = len(gender_truth)
    nation_character = len(nation_truth)
    country_character = len(country_truth)
    address_character = len(address_truth)

    number_wrong = (number_wrong * i + distance(number_text,
                                                number_truth)/id_character)/(i+1)
    name_wrong = (name_wrong * i + distance(name_text,
                                            name_truth)/name_character)/(i+1)
    dob_wrong = (dob_wrong * i + distance(dob_text,
                                          dob_truth)/dob_character)/(i+1)
    gender_wrong = (gender_wrong * i + distance(gender_text,
                                                gender_truth)/gender_character)/(i+1)
    nation_wrong = (nation_wrong * i + distance(nation_text,
                                                nation_truth)/nation_character)/(i+1)
    country_wrong = (country_wrong * i + distance(country_text,
                                                  country_truth)/country_character)/(i+1)
    address_wrong = (address_wrong * i + distance(address_text,
                                                  address_truth)/address_character)/(i+1)

    print('ID:      ', round(number_wrong*100, 2), distance(number_text, number_truth),
          ' ID character', id_character)
    print('Name:    ', round(name_wrong*100, 2), distance(name_text, name_truth),
          ' Name character', name_character)
    print('DOB:     ', round(dob_wrong*100, 2), distance(dob_text, dob_truth),
          ' DOB character', dob_character)
    print('GENDER:  ', round(gender_wrong*100, 2), distance(gender_text, gender_truth),
          ' GENDER character', gender_character)
    print('NATION:  ', round(nation_wrong*100, 2), distance(nation_text, nation_truth),
          'Nation character', nation_character)
    print('COUNTRY: ', round(country_wrong*100, 2), distance(country_text, country_truth),
          'COUNTRY character', country_character)
    print('ADDRESS: ', round(address_wrong*100, 2), distance(address_text, address_truth),
          'ADDRESS character', address_character)
