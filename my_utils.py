import pyautogui as pg
import cv2
from is_part_image import is_part
import numpy as np
from time import sleep
from configs.config import map_list
import random
from configs.config import steps_right, steps_back
from mouse_control import MouseControls
from configs.resolution_conf import w, h

ms = MouseControls()


def rescale_w(coord):
    """
    Rescale the width coordinate from a 2K resolution to another defined resolution.

    Args:
        coord (int): The coordinate to rescale.

    Returns:
        int: The rescaled coordinate.
    """
    return (w * coord) // 2560


def rescale_h(coord):
    """
    Rescale the height coordinate from a 2K resolution to another defined resolution.

    Args:
        coord (int): The coordinate to rescale.

    Returns:
        int: The rescaled coordinate.
    """
    return (h * coord) // 1440


def rescale_template(template):
    """
    Rescale the template image to match a defined resolution.

    Args:
        template (numpy.ndarray): The template image.

    Returns:
        numpy.ndarray: The rescaled template image.
    """
    w_image, h_image = template.shape
    w_image = rescale_w(w_image)
    h_image = rescale_h(h_image)
    template = cv2.resize(template, (h_image, w_image), interpolation=cv2.INTER_AREA)
    return template


def find_tag_new(screenshots):
    """
    Find the tag in the compass area at top of the image.

    Args:
        screenshots (dict): A dictionary containing color and gray screenshots.

    Returns:
        None
    """
    print('find tag new start')
    x1, x2, y1, y2 = rescale_w(28), rescale_w(51), rescale_h(860), rescale_h(1700)

    wide = 15
    segments_n = (y2 - y1) // wide
    for i in range(3):
        template = screenshots['color'][x1: x2, y1: y2, :]
        start = 0
        pos = -1
        for part in range(segments_n):
            ch = template[:, start:start + wide, :]
            res = define_color_new(ch)
            yellow_pattern = [33, 237, 251]
            if np.array_equal(res, np.array(yellow_pattern)):
                pos = part
                to_move = ((wide * pos - rescale_w(420) + 5) * rescale_w(1320)) // rescale_w(178)
                ms.move_relative(to_move, 0)
                break
            start += wide
        if pos == -1:
            ms.move_relative(rescale_w(2632), 0)
            sleep(0.8)
        else:
            break
    print('find tag new finish')


def take_screenshot(region=None):
    """
    Take a screenshot of the specified region.

    Args:
        region (tuple): The coordinates of the region to capture. Default is None to capture the entire screen.

    Returns:
        tuple: A tuple containing the color and gray versions of the screenshot.
    """
    imageObj = pg.screenshot(region=region)
    color = cv2.cvtColor(np.array(imageObj), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(np.array(imageObj), cv2.COLOR_RGB2GRAY)
    return color, gray


def press_for_long(button, delay=0.3):
    """
    Simulate pressing and holding a keyboard button with delay on keep pressing.

    Args:
        button (str): The button to press.
        delay (float): The duration to hold the button. Default is 0.3 seconds.

    Returns:
        None
    """
    pg.keyDown(button)
    sleep(delay)
    pg.keyUp(button)


def define_map(screenshots):
    """
    Find the current map in the screenshots using template matching.

    Args:
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots

    Returns:
        str: The name of the detected map, or None if no map is detected.
    """
    press_for_long('m', 0.5)
    scr = screenshots['gray']
    for m in map_list:
        template = cv2.imread(rf'screenshots\maps\{m}.png', 0)
        is_there, _ = is_part(scr, template, 0.5)
        if is_there:
            print(f'map {m} detected')
            return m
    return None


def search_f_key(screenshots):
    """
    Search for the 'F' key in a region of the screenshot using template matching.

    Args:
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots

    Returns:
        bool: True if the 'F' key is found, False otherwise.
    """
    cv_imageObj = screenshots['gray']
    template = cv2.imread(r'screenshots\cut\jump_cut_mod.png', 0)

    w_image, h_image = template.shape
    w_image = rescale_w(w_image)
    h_image = rescale_h(w_image)
    template = cv2.resize(template, (w_image, h_image), interpolation=cv2.INTER_AREA)
    cv_imageObj = cv_imageObj[rescale_w(810): rescale_w(865), rescale_h(1420): rescale_h(1534)]
    is_there, center = is_part(cv_imageObj, template, 0.6)
    if is_there:
        print('F found')
        return True


def define_color_new(image):
    """
    Define the dominant color in an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The dominant color.
    """
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def stop():
    """
    Command to stop running.

    Returns:
        None
    """
    pg.keyUp('l')
    pg.keyUp('w')


def look_around(res_dict, event_list):
    """
    Perform a sequence of movements to look around and detect the tag.

    Args:
        res_dict (dict): A dictionary to store the detection result.
        event_list (dict): A dictionary of event flags.

    Returns:
        None
    """
    moves = [20, 50, 100]
    for i in range(20):
        if event_list['detected'].is_set() or event_list['final'].is_set():
            return
        ms.move_relative(20, 0)
        sleep(0.1)
    pg.keyDown('w')
    sleep(4)
    pg.keyUp('w')
    if event_list['detected'].is_set() or event_list['final'].is_set():
        return
    ms.move_relative(random.choice(moves), 0)
    pg.keyDown('a')
    sleep(1)
    pg.keyUp('a')
    if event_list['detected'].is_set() or event_list['final'].is_set():
        return
    res_dict['tag_in'] = True
    event_list['tag_in'].clear()


def steps_on_timing():
    """
    Perform a sequence of steps when no result is reached on timeout.

    Returns:
        None
    """
    print('steps_on_timing')
    sides_list = ['a', 'd']
    pg.keyDown('s')
    sleep(steps_back)
    pg.keyUp('s')
    toMove = random.choice(sides_list)
    pg.keyDown(toMove)
    sleep(steps_right)
    pg.keyUp(toMove)

    print('steps end')
