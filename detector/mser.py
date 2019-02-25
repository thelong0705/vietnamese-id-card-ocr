import cv2
import numpy as np
import imutils
from itertools import groupby


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def grouper(iterable, index, threshold):
    prev = ()
    group = []
    for item in iterable:
        if not prev or item[index] - prev[index] <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def special_grouper(iterable, index, threshold):
    prev = ()
    group = []
    for item in iterable:
        if not prev or item[index] - prev[index] - prev[2] <= threshold:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def draw_rec(list_rec_tuple, img):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = rec_tuple
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


def get_coordinate(group):
    xmin, ymin, _, _ = group[0]
    xmax, ymax, w, h = group[-1]  # get the last location of group
    # get the bottom right location of id group
    xmax = xmax + w
    ymax = ymax + h
    return (xmin, ymin, xmax, ymax)


mser = cv2.MSER_create(_delta=4, _max_area=500)
image_path = 'cropped1.png'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


regions, bboxs = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

_, contours, _ = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create a list of character locations
locs = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    locs.append((x, y, w, h))
# devide locations list into group base on y axis
locs.sort(key=lambda tup: tup[1])
location_groups = list(grouper(locs, 1, 15))
# group structure: id->name->

# get id part
# id is the first group
id_group = location_groups[0]
id_group.sort(key=lambda tup: tup[0])  # sort element in id group by x-axis
# split label part and information part by height. Because the label part is shorter than the information
average_height = sum(element[3]for element in id_group)/len(id_group)
for element in id_group:
    if element[3] < average_height:
        id_group.remove(element)
xmin, ymin, xmax, ymax = get_coordinate(id_group)
id_group_image = img[ymin-5: ymax+5, xmin-5:xmax+5]  # some padding
cv2.imwrite('trash/id.png', id_group_image)


# get name part
name_group = location_groups[1]
name_group.sort(key=lambda tup: tup[0])
# split label part and imformation part
name_group_split_list = list(special_grouper(name_group, 0, 20))
# the information 's x-axis location is higher
name_group = name_group_split_list[-1]
# get the first location (top_left) of id_group
xmin, ymin, xmax, ymax = get_coordinate(name_group)
# crop the name part:
name_group_image = img[ymin-10: ymax+10, xmin-5:xmax+5]  # some padding
cv2.imwrite('trash/name.png', name_group_image)

# get birthday part:
birthday_group = location_groups[2]
