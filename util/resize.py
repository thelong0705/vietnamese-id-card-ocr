import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def resize_img_by_height(img, size=500):
    h, w, _ = img.shape
    ratio = h/size
    img = resize(img, height=size)
    return (img, ratio)


def resize_img_by_width(img, size=500):
    h, w, _ = img.shape
    ratio = w/size
    img = resize(img, width=size)
    return (img, ratio)


def resize_to_range(img, min_dim, max_dim):
    orig_height, orig_width, _ = img.shape
    img_min_dim = min(orig_height, orig_width)
    large_scale_factor = min_dim/img_min_dim
    large_height = round(orig_height * large_scale_factor)
    large_width = round(orig_width * large_scale_factor)
    large_size = (large_height, large_width)
    img_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dim / img_max_dim
    small_height = round(orig_height * small_scale_factor)
    small_width = round(orig_width * small_scale_factor)
    small_size = (small_height, small_width)
    if max(large_size) > max_dim:
        new_size = small_size
        factor = small_scale_factor
    else:
        new_size = large_size
        factor = large_scale_factor
    if factor < 1:
        return resize(img, width=new_size[1], height=new_size[0], inter=cv2.INTER_AREA), factor
    else:
        return resize(img, width=new_size[1], height=new_size[0], inter=cv2.INTER_LINEAR), factor


def resize_by_max(img, max_value):
    h, w, _ = img.shape
    max_dim = max(h, w)
    ratio = 1
    if max_dim <= max_value:
        return (img, ratio)
    if max_dim == h:
        ratio = img.shape[0] / max_value
        img = resize(img, height=max_value)
    if max_dim == w:
        ratio = img.shape[1] / max_value
        img = resize(img, width=max_value)
    return (img, ratio)
