from multiprocessing import Process, Event, Manager, current_process
from time import sleep
import pyautogui as pg
import cv2

from is_part_image import is_part
from my_utils import press_for_long, find_tag_new
from configs.config import maps_destinations


def forward_and_detect_tag(event_list, mean_dist, map_name):
    """
    Move forward in the way depending on the map and destination.

    Args:
        event_list (dict): Dictionary containing event flags.
        mean_dist (float): Mean distance to the destination.
        map_name (str): Name of the map.

    Returns:
        None
    """
    print(f"Process {current_process().name} started")
    short = maps_destinations[map_name]['short']
    long = maps_destinations[map_name]['long']
    if mean_dist <= short:
        while not event_list['ground'].is_set() and not event_list['button'].is_set():
            pg.keyDown('w')
            sleep(0.5)
    if mean_dist > short and mean_dist <= long:
        pg.keyDown('w')
        print('parachute')
        sleep(7)
        press_for_long('f')
        while not event_list['ground'].is_set() and not event_list['button'].is_set():
            pg.keyDown('w')
            sleep(0.5)
            pg.keyDown('c')
            sleep(0.5)
            pg.keyUp('c')
            sleep(0.5)
        pg.keyUp('c')
    if mean_dist > long:
        print('long jump')
        pg.keyDown('w')
        print('parachute')
        sleep(7)
        press_for_long('f')
        while not event_list['ground'].is_set() and not event_list['button'].is_set():
            pg.keyDown('w')
            pg.keyDown('c')
            sleep(2)
            pg.keyUp('c')
            sleep(1)
        pg.keyUp('c')
    pg.keyUp('w')
    print(f"Process {current_process().name} finished")


def detect_ground(event_list, screenshots):
    """
    Detect whether the character is on the ground or in the water basing on the icon.

    Args:
        event_list (dict): Dictionary containing event flags.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.

    Returns:
        None
    """
    print(f"Process {current_process().name} started")

    while True:
        cv_imageObj = screenshots['gray']
        template = cv2.imread(r'screenshots\cut\underwater_cut.png', 0)
        # template = rescale_template(template)
        is_there, center = is_part(cv_imageObj, template, 0.95)
        if is_there:
            print('water')
            pg.keyDown('space')
            sleep(5)
            pg.keyUp('space')
            event_list['ground'].set()
            break

        template = cv2.imread(r'screenshots\cut\stand_cut.png', 0)
        # template = rescale_template(template)
        is_there, center = is_part(cv_imageObj, template, 0.95)
        if is_there:
            print('ground')
            event_list['ground'].set()
            break
    print(f"Process {current_process().name} finished")


def fly_over(screenshots, mean_dist, map_name, button_event):
    """
    Perform flying actions from plane to the ground.

    Args:
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        mean_dist (float): Mean distance to the destination.
        map_name (str): Name of the map.
        button_event (multiprocessing.Event): Event to signal the detection of a button.

    Returns:
        None
    """
    print('fly over start')

    processes_fly = []
    grounded_event = Event()
    keep_tag_event = Event()
    keep_tag_event.set()
    manager = Manager()
    res_dict = manager.dict()
    res_dict['keep_tag'] = True

    find_tag_new(screenshots)
    event_list = {'ground': grounded_event, 'keep_tag': keep_tag_event, 'button': button_event}
    detect_tag_proc = Process(target=forward_and_detect_tag, args=(event_list, mean_dist, map_name,), name='forw and detect')
    detect_ground_proc = Process(target=detect_ground, args=(event_list, screenshots,), name='ground')
    processes_fly.append(detect_tag_proc)
    processes_fly.append(detect_ground_proc)
    for p in processes_fly:
        p.start()

    grounded_event.wait()  # <- blocks until condition met

    for p in processes_fly:
        p.join()
    pg.keyUp('w')
    print('fly over finish')
