import cv2
import numpy as np
import imutils
from itertools import groupby
import copy


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


def remove_shorter_than_average(group):
    group_orig = copy.deepcopy(group)
    average_height = sum(element[3]
                         for element in group)/len(group)
    for element in group_orig:
        if element[3] <= average_height:
            group.remove(element)
    return group


mser = cv2.MSER_create(_delta=4, _max_area=500)
image_path = 'test/cropped1.png'
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
draw_rec(locs,img)
show_img(img)
location_groups = list(grouper(locs, 1, 10))  # TODO hard-code
# group structure: id->name->

# get id part
# id is the first group
id_group = location_groups[0]
id_group.sort(key=lambda tup: tup[0])  # sort element in id group by x-axis
# split label part and information part by height. Because the label part is shorter than the information
id_group = remove_shorter_than_average(id_group)
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
name_group_image = img[ymin-5: ymax+5, xmin-5:xmax+5]  # some padding
cv2.imwrite('trash/name.png', name_group_image)

# get birthday part:
birthday_group = location_groups[2]
birthday_group.sort(key=lambda tup: tup[0])
birthday_group = remove_shorter_than_average(birthday_group)
birthday_group_split_list = list(special_grouper(birthday_group, 0, 10))
birthday_group = birthday_group_split_list[-1]
xmin, ymin, xmax, ymax = get_coordinate(birthday_group)
birthday_group_image = img[ymin-5: ymax+5, xmin-5:xmax+5]  # some padding
cv2.imwrite('trash/dob.png', birthday_group_image)

# get gender part
gender_group = location_groups[3]
gender_group.sort(key=lambda tup: tup[0])
gender_group_split_list = list(special_grouper(gender_group, 0, 20))
gender_group = gender_group_split_list[0]
xmin, ymin, w, h = gender_group[-1]
xmax = xmin + w
ymax = ymin + h
# some padding
gender_image = img[ymin-5: ymax+5, xmin-5:xmax+5]
cv2.imwrite('trash/gender.png', gender_image)

# get nationality part
# get gender part
national_group = location_groups[3]
national_group.sort(key=lambda tup: tup[0])
national_group_split_list = list(special_grouper(national_group, 0, 20))
national_group = national_group_split_list[-1]
national_group = remove_shorter_than_average(national_group)
xmin, ymin, xmax, ymax = get_coordinate(national_group)
gender_image = img[ymin-5: ymax+5, xmin-5:xmax+5]
cv2.imwrite('trash/nation.png', gender_image)

# get country side part
country_group = location_groups[4]
country_group.sort(key=lambda tup: tup[0])
country_group_split_list = list(special_grouper(country_group, 0, 20))
country_group = country_group_split_list[-1]
xmin, ymin, xmax, ymax = get_coordinate(country_group)
country_image = img[ymin-5: ymax+5, xmin-5:xmax+5]
cv2.imwrite('trash/country.png', country_image)

# get address part
address_group = location_groups[5]
average_y_axis = sum(element[1]
                     for element in address_group)/len(address_group)
address_group_1 = [c for c in address_group if c[1] <= average_y_axis]
address_group_2 = [c for c in address_group if c[1] > average_y_axis]
# get first address line
address_group_1.sort(key=lambda tup: tup[0])
address_group_1 = list(special_grouper(address_group_1, 0, 20))[-1]
xmin, ymin, xmax, ymax = get_coordinate(address_group_1)
first_address_image = img[ymin-5: ymax+5, xmin-5:xmax+5]
cv2.imwrite('trash/address1.png', first_address_image)
# get second address line
address_group_2.sort(key=lambda tup: tup[0])
xmin, ymin, xmax, ymax = get_coordinate(address_group_2)
second_address_image = img[ymin-5: ymax+5, xmin-5:xmax+5]
cv2.imwrite('trash/address2.png', second_address_image)
