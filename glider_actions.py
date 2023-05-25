from time import sleep
import pyautogui as pg
import cv2
import numpy as np
import math
from statistics import mean
from configs.config import maps_to_glide, TAKE_LAST_N_MEASUERS
from my_utils import take_screnshot, press_for_long, find_tag_new, rescale_w, rescale_h
from fly_over import fly_over
from search_targets import main_search
from collections import deque
from configs.config import  maps_destinations, scrolled_coordinates


def mark_glider(map_name, ms):
    print('mark glider')
    x, y = maps_to_glide[map_name]
    if map_name in scrolled_coordinates.keys():
        new_x, new_y = scrolled_coordinates[map_name]
        ms.move(x, y)
        sleep(0.5)
        for _ in range(4):
            ms.scroll_up(12)
        sleep(0.5)
        pg.rightClick(new_x, new_y)
        sleep(0.5)
        for _ in range(4):
            ms.scroll_down(12)

    else:
        pg.rightClick(x, y)
    return x,y


def wait_and_jump(x, y, map_name,  screenshots):
    print('wait and jump start')
    mean_dest = 0
    if map_name == 'taego':
        sleep(15)
        print('need to jump')
        press_for_long('m')
        press_for_long('f')
    else:
        too_far = True
        destinations = deque(TAKE_LAST_N_MEASUERS*[0], TAKE_LAST_N_MEASUERS)
        appended_cnt = 0
        mean_dest = 0
        short = maps_destinations[map_name]['short']
        while too_far:
            print('wait and jump')
            _, cv_imageObj1 = take_screnshot(region=(rescale_w(565),
                                                     rescale_h(0),
                                                     rescale_w(1425),
                                                     rescale_h(1440)))
            sleep(0.2)
            _, cv_imageObj2 = take_screnshot(region=(rescale_w(565),
                                                     rescale_h(0),
                                                     rescale_w(1425),
                                                     rescale_h(1440)))

            diff = cv2.absdiff(cv_imageObj1, cv_imageObj2)
            threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            # Применяем эрозию и дилатацию, чтобы убрать шум
            kernel = np.ones((4, 4), np.uint8)
            threshold = cv2.erode(threshold, kernel, iterations=2)
            threshold = cv2.dilate(threshold, kernel, iterations=2)
            cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            xses = []
            yses = []
            for c in cnts:
                # Вычисляем моменты и центр контура
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"]) + rescale_w(564)
                cY = int(M["m01"] / M["m00"])
                xses.append(cX)
                yses.append(cY)


            if len(xses) > 0 and len(yses) > 0:
                diff_x = abs(max(xses) - min(xses))
                diff_y = abs(max(yses) - min(yses))
                if diff_x < rescale_w(50) and diff_y < rescale_h(50):
                    destination = math.sqrt((x - mean(xses)) ** 2 + (y - mean(yses)) ** 2)
                    destinations.append(destination)
                    appended_cnt += 1
                    print('desintaions', destinations, mean(destinations), x, y, xses, yses)
                    mean_dest = mean(destinations)
                    if (((destination - mean_dest) > rescale_w(10)) or (mean_dest < short)) and appended_cnt > TAKE_LAST_N_MEASUERS:
                        print('need to jump')
                        press_for_long('m')
                        press_for_long('f')
                        break
    find_tag_new(screenshots)
    pg.keyDown('w')
    print('wait and jump finish')
    return mean_dest



def glider_actions(map_name, button_event, screenshots, ms):

    print('start glider actions')
    x, y = mark_glider(map_name, ms)

    mean_dist = wait_and_jump(x, y, map_name, screenshots)

    fly_over(screenshots, mean_dist, map_name, button_event)
    main_search(button_event, screenshots, map_name, ms)
