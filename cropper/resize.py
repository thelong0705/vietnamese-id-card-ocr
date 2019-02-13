import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', required=True,
                help='Number image to be resized')
ap.add_argument('-i', '--input', required=True,
                help='Path to the folder images to be resize')
ap.add_argument('-w', '--width', required=True,
                help='target width')
ap.add_argument('-he', '--height', required=True,
                help='target height')
ap.add_argument('-o', '--output', required=True,
                help='Path to the folder will store frames')
args = vars(ap.parse_args())
width = int(args['width'])
height = int(args['height'])
if not os.path.exists(args['output']):
    os.makedirs(args['output'])

for i in range(0, int(args['number'])):
    image = cv2.imread('{}/img_{}.jpg'.format(args['input'],i))
    res = cv2.resize(image, (width, height))
    height, width, channels = res.shape
    cv2.imwrite('resize_img/img_{}.jpg'.format(i), res)
