import pyautogui as pg
import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_part(img, template, thresh):
    # img = cv2.imread(r'screenshots\full\main_menu.png',0)

    # template = cv2.imread(r'screenshots\cut\start_game.png',0)

    w, h = template.shape[::-1]


    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # print('res', res)
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

    # print(res.shape)
    # print(loc)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow('Detected', img)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res[::-1])
    # print(min_val, max_val, min_loc, max_loc)
    # # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # center_loc = (min_loc[0] - w//2, min_loc[1] + h//2)
    # print(min_loc[0] - w//2, min_loc[1] + h//2)
    # return True, center_loc
#
# is_part(cv2.imread(r'screenshots\full\main_menu.png',0),
#         cv2.imread(r'screenshots\cut\start_game.png',0),
#         0.99)

# is_part(cv2.imread(r'screenshots\full\test.png',0),
#         cv2.imread(r'screenshots\cut\start_game.png',0),
#         0.99)


# pg.moveTo(200,300, 0.5)

# print(pg.position())

# pg.leftClick(146,108)

# imageObj = pg.screenshot()
# cv_imageObj = cv2.cvtColor(np.array(imageObj), cv2.COLOR_RGB2BGR)
# plt.imshow(cv_imageObj)
# plt.show()