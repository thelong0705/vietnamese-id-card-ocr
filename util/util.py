import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading

def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def plot_img(img):
    plt.imshow(img)
    plt.show()


def plot_img_bin(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def draw_rec(list_rec_tuple, img, ratio=1):
    for rec_tuple in list_rec_tuple:
        x, y, w, h = tuple(int(ratio * l) for l in rec_tuple)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


def get_threshold_img(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thresh = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def get_contour_boxes(img):
    _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes = []
    for cnt in cnts:
        contour_boxes.append(cv2.boundingRect(cnt))
    return contour_boxes


def find_max_box(group):
    xmin = min(group, key=lambda t: t[0])[0]
    ymin = min(group, key=lambda t: t[1])[1]
    xmax_box = max(group, key=lambda t: t[0] + t[2])
    xmax = xmax_box[0] + xmax_box[2]
    ymax_box = max(group, key=lambda t: t[1] + t[3])
    ymax = ymax_box[1] + ymax_box[3]
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def get_img_from_box(orig, ratio, box, padding=0):
    x0, y0, x1, y1 = tuple(int(ratio * element) for element in box)
    height, width, _ = orig.shape
    if y0 - padding > 0:
        y0 = y0 - padding
    if y1 + padding < height:
        y1 = y1 + padding
    return orig[y0:y1, x0:x1]


def four_point_transform(image, rect):
    (tl, tr, bl, br) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_NEAREST)
    return warped


def run_item(f, item):
    result_info = [threading.Event(), None]

    def runit():
        result_info[1] = f(item[0], item[1])
        result_info[0].set()
    threading.Thread(target=runit).start()
    return result_info


def gather_results(result_infos):
    results = []
    for i in range(len(result_infos)):
        result_infos[i][0].wait()
        results.append(result_infos[i][1])
    return results
