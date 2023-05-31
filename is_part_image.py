import cv2
import numpy as np


def is_part(img, template, thresh):
    """
    Check if a template is present in an image using template matching.

    Args:
        img (numpy.ndarray): The input image.
        template (numpy.ndarray): The template to match.
        thresh (float): The threshold for matching.

    Returns:
        tuple: A tuple containing a boolean value indicating if the template is present,
               and the center coordinates of the best match.
    """
    w_templ, h_templ = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res[::-1] >= thresh)
    is_there = False
    center_loc = None
    if loc[0].size > 0:
        is_there = True
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        center_loc = (max_loc[0] + w_templ // 2, max_loc[1] + h_templ // 2)
    return is_there, center_loc


def is_part_color(img, template, thresh):
    """
    Check if a colored template is present in a colored image using template matching in HSV color space.

    Args:
        img (numpy.ndarray): The input image.
        template (numpy.ndarray): The template to match.
        thresh (float): The threshold for matching.

    Returns:
        tuple: A tuple containing a boolean value indicating if the template is present,
               and the center coordinates of the best match.
    """
    h, w = template.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = cv2.split(img_hsv)
    template_h, template_s, template_v = cv2.split(template_hsv)
    res_h = cv2.matchTemplate(img_h, template_h, cv2.TM_CCOEFF_NORMED)
    res_s = cv2.matchTemplate(img_s, template_s, cv2.TM_CCOEFF_NORMED)
    res_v = cv2.matchTemplate(img_v, template_v, cv2.TM_CCOEFF_NORMED)
    res = (res_h + res_s + res_v) / 3
    loc = np.where(res >= thresh)
    is_there = False
    center_loc = None
    if loc[0].size > 0:
        is_there = True
        max_loc = np.unravel_index(np.argmax(res), res.shape)
        center_loc = (max_loc[1] + w // 2, max_loc[0] + h // 2)
    return is_there, center_loc
