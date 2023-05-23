import pyautogui as pg
import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_part(img, template, thresh):
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res[::-1] >= thresh)
    is_there = False
    center_loc = None
    if loc[0].size > 0:
        is_there = True
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        center_loc = (max_loc[0] + w//2, max_loc[1] + h//2)
        # print(center_loc)
    return is_there, center_loc

def is_part_color(img, template, thresh):
    h, w = template.shape[:2]
    # Convert the image and template to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    img_h, img_s, img_v = cv2.split(img_hsv)
    template_h, template_s, template_v = cv2.split(template_hsv)

    # Apply template matching in each channel
    res_h = cv2.matchTemplate(img_h, template_h, cv2.TM_CCOEFF_NORMED)
    res_s = cv2.matchTemplate(img_s, template_s, cv2.TM_CCOEFF_NORMED)
    res_v = cv2.matchTemplate(img_v, template_v, cv2.TM_CCOEFF_NORMED)

    # Combine the results
    res = (res_h + res_s + res_v) / 3

    # Find the location of the best match
    loc = np.where(res >= thresh)

    is_there = False
    center_loc = None
    if loc[0].size > 0:
        is_there = True
        max_loc = np.unravel_index(np.argmax(res), res.shape)
        center_loc = (max_loc[1] + w // 2, max_loc[0] + h // 2)

    return is_there, center_loc
