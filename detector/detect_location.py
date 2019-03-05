import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt



def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def get_each_roi(img):
    h, w, _ = img.shape
    id_part = img[int(0.27*h):int(0.35*h), int(0.46*w):w]
    name_part = img[int(0.4*h):int(0.5*h), int(0.46*w):w]
    dob_part = img[int(0.52*h):int(0.59*h), int(0.6*w):w]
    gender_part = img[int(0.6*h):int(0.7*h), int(0.45*w):int(0.6*w)]
    nation_part = img[int(0.6*h):int(0.7*h), int(0.73*w):w]
    country_part = img[int(0.7*h):int(0.8*h), int(0.46*w):w]
    address_part_first = img[int(0.8*h):int(0.88*h), int(0.51*w):w]
    address_part_second = img[int(0.89*h):h, int(0.4*w):w]

    return id_part,name_part


