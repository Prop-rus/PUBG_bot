import pyautogui as pg
import cv2
from is_part_image import is_part
import numpy as np
from time import sleep
from configs.config import map_list
import random

def find_tag_new(screenshots):
    print('find tag new start')
    x1, x2, y1, y2 = 28, 51, 860, 1700

    wide = 15
    segments_n = (y2 - y1) // wide
    for i in range(3):
        template = screenshots['color'][x1: x2, y1: y2, :]
        start = 0
        pos = -1
        for part in range(segments_n):
            ch = template[:, start:start + wide, :]
            res = define_color_new(ch)
            if np.array_equal(res, np.array([33, 237, 251])):
                pos = part
                to_move = ((wide * pos - 420 + 5) * 1320) // 178
                ms.move_relative(to_move, 0)
                break
            start += wide
        if pos == -1:
            ms.move_relative(2632, 0)
            sleep(0.8)
        else:
            break
    print('find tag new finish')


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


def define_color_new(image):
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def stop():
    pg.keyUp('l')
    pg.keyUp('w')


def look_around(res_dict, event_list, ms):

    moves = [20,50,100]
    for i in range(20):
        if event_list['detected'].is_set() or event_list['final'].is_set():
            return
        ms.move_relative(20, 0)
        sleep(0.1)
    pg.keyDown('w')
    sleep(4)
    pg.keyUp('w')
    if event_list['detected'].is_set()  or event_list['final'].is_set():
        return
    ms.move_relative(random.choice(moves), 0)
    pg.keyDown('a')
    sleep(1)
    pg.keyUp('a')
    if event_list['detected'].is_set()  or event_list['final'].is_set():
        return
    res_dict['tag_in'] = True
    event_list['tag_in'].clear()

def steps_on_timing():
    print('steps_on_timing')
    steps_right = 5
    steps_back = 2
    sides_list = ['a', 'd']
    pg.keyDown('s')
    sleep(steps_back)
    pg.keyUp('s')
    toMove = random.choice(sides_list)
    pg.keyDown(toMove)
    sleep(steps_right)
    pg.keyUp(toMove)

    print('steps end')