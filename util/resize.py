import cv2
import imutils

def resize_img_by_height(img, size=500):
    h, w, _ = img.shape
    ratio = h/size
    img = imutils.resize(img, height=size)
    return (img, ratio)


def resize_img_by_width(img, size=500):
    h, w, _ = img.shape
    ratio = w/size
    img = imutils.resize(img, width=size)
    return (img, ratio)
