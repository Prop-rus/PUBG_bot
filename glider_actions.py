from time import sleep
import pyautogui as pg
import cv2
import numpy as np
import math
from statistics import mean
from configs.config import maps_to_glide, TAKE_LAST_N_MEASUERS
from my_utils import take_screenshot, press_for_long, find_tag_new, rescale_w, rescale_h
from fly_over import fly_over
from collections import deque
from configs.config import maps_destinations
from mouse_control import MouseControls

ms = MouseControls()


def mark_land_point(map_name):
    """
    Mark the target vehicle on the map.

    Args:
        map_name (str): Name of the map.

    Returns:
        Tuple[int, int]: The coordinates of the glider marker.
    """
    print('mark glider')
    x, y = maps_to_glide[map_name]
    x = rescale_w(x)
    y = rescale_h(y)
    pg.rightClick(x, y)

    return x, y


def wait_and_jump(x, y, map_name, screenshots):
    """
    Wait for the target coordinates to reach the desired distance and initiate the jump.

    Args:
        x (int): X-coordinate of the marker.
        y (int): Y-coordinate of the marker.
        map_name (str): Name of the map.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots

    Returns:
        float: Mean distance to the destination.
    """
    print('wait and jump start')
    mean_dest = 0
    if map_name == 'taego':
        sleep(15)
        print('need to jump')
        press_for_long('m')
        press_for_long('f')
    else:
        too_far = True
        destinations = deque(TAKE_LAST_N_MEASUERS * [0], TAKE_LAST_N_MEASUERS)
        appended_cnt = 0
        mean_dest = 0
        short = maps_destinations[map_name]['short']
        while too_far:
            print('wait and jump')
            _, cv_imageObj1 = take_screenshot(region=(rescale_w(565),
                                                      rescale_h(0),
                                                      rescale_w(1425),
                                                      rescale_h(1440)))
            sleep(0.2)
            _, cv_imageObj2 = take_screenshot(region=(rescale_w(565),
                                                      rescale_h(0),
                                                      rescale_w(1425),
                                                      rescale_h(1440)))

            diff = cv2.absdiff(cv_imageObj1, cv_imageObj2)
            threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((4, 4), np.uint8)
            threshold = cv2.erode(threshold, kernel, iterations=2)
            threshold = cv2.dilate(threshold, kernel, iterations=2)
            cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            xses = []
            yses = []
            for c in cnts:
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


def loot_from_box():
    pg.press('i')
    sleep(0.4)
    pg.rightClick(rescale_w(642), rescale_h(195), duration=1)
    sleep(0.4)
    pg.leftClick(rescale_w(1276), rescale_h(730), duration=1)
    sleep(0.4)
    pg.leftClick(rescale_w(1276), rescale_h(1157), duration=1)
    sleep(5)
    pg.press('x')



def loot_actions(map_name, button_event, screenshots):
    """
    Perform glider actions, including marking the glider, waiting, jumping, flying over, and searching for targets.

    Args:
        map_name (str): Name of the map.
        button_event (multiprocessing.Event): Event to signal the detection of a button.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots

    Returns:
        None
    """
    print('start glider actions')
    x, y = mark_land_point(map_name)

    mean_dist = wait_and_jump(x, y, map_name, screenshots)

    fly_over(screenshots, mean_dist, map_name, button_event)
    loot_from_box()
