import pyautogui as pg
import cv2
from is_part_image import is_part
import numpy as np
from time import sleep
from configs.config import map_list

def take_screnshot(region=None):
    imageObj = pg.screenshot(region=region)
    color = cv2.cvtColor(np.array(imageObj), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(np.array(imageObj), cv2.COLOR_RGB2GRAY)
    return color, gray

def press_for_long(button, dalay=0.3):
    pg.keyDown(button)
    sleep(dalay)
    pg.keyUp(button)


def define_map(screenshots):
    press_for_long('m', 0.5)
    scr =  screenshots['gray']
    for m in map_list:
        template = cv2.imread(rf'screenshots\maps\{m}.png', 0)
        is_there, _ = is_part(scr, template, 0.5)
        if is_there:
            print(f'map {m} detected')
            return m
    return None

def search_f_key(screenshots):
    cv_imageObj =  screenshots['gray']
    template = cv2.imread(r'screenshots\cut\jump_cut_mod.png', 0)
    cv_imageObj = cv_imageObj[810: 865, 1420: 1534]
    is_there, center = is_part(cv_imageObj, template, 0.91)
    if is_there:
        print('F found')
        return True
