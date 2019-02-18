# import stuffs
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

# declare program 's argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-r", "--reference", required=True,
                help="path to refere")
args = vars(ap.parse_args())

# load reference picuture
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find countours in the referecene picture
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the countours to map template to numeber
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    print(x, y, w, h)
    roi = cv2.resize(roi, (14, 20))
    digits[i] = roi

# load the image to read
image = cv2.imread(args["image"])
height, width, channels = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []
# get each number in picture
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    #little padding to get number more precisely
    # cv2.rectangle(thresh, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 1)
    locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
output = []

index = 0
for l in locs:
    index += 1
    (x, y, w, h) = l
    roi = thresh[y:y+h, x:x+w]
    roi = cv2.resize(roi, (14, 20))
    scores = []
    for (digit, digitROI) in digits.items():
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
    output.append(str(np.argmax(scores)))

print(output)

