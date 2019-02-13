import cv2
import numpy as np
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to the video to be cut')
ap.add_argument('-o', '--output', required=True,
                help='Path to the folder will store frames')
args = vars(ap.parse_args())
vid = cv2.VideoCapture(args['image'])

if not os.path.exists(args['output']):
    os.makedirs(args['output'])

#for frame identity
index = 0
img_index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret:
        break
    # after 5 frame save 1 image
    if index % 5 == 0:
        name = '{}/img_{}.jpg'.format(args['output'],str(img_index))
        print('Creating...' + name)
        img_index += 1
    cv2.imwrite(name, frame)

    # next frame
    index += 1
