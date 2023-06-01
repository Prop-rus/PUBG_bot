from multiprocessing import current_process
import torch
import numpy as np
from is_part_image import is_part
from my_utils import find_tag_new, define_color_new, rescale_w, rescale_h
from configs.config import gliders_or_cars, w
from mouse_control import MouseControls
from multiprocessing import Process, Manager, Event
import cv2
import pyautogui as pg
from time import sleep


ms = MouseControls()


def search_vehicle(res_dict, event_list, screenshots, vehicle_classes=[0, 1, 2, 3, 4]):
    """
    Search for vehicles in the screenshots using YOLOv5 model.

    Args:
        res_dict (multiprocessing.Manager.dict): Shared dictionary for storing the results.
        event_list (multiprocessing.Manager.dict): Shared dictionary for managing events.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        vehicle_classes (list): List of vehicle classes to detect.
    """
    print(f"Process {current_process().name} started")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
    model.eval()
    model.cuda()
    model.imgsz = (1280, 1280)
    model.multi_label = False
    model.max_det = 1
    model.conf_thres = 0.6
    model.iou_thres = 0.6
    model.classes = vehicle_classes  # car, motorcycle, bus, truck
    cur_x = w // 2
    success_cnt = 0
    while not event_list['success'].is_set():
        if not res_dict['detect']:
            sleep(0.3)
            continue
        if event_list['success'].is_set() or event_list['final'].is_set():
            break
        screenshot = screenshots['color']
        results = model(screenshot.copy())
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        confidence = results.xyxyn[0][:, -2].cpu().numpy()
        ind_cars = np.where(np.isin(labels, vehicle_classes))

        labels = labels[ind_cars]
        cord = cord[ind_cars]
        confidence = confidence[ind_cars]
        if len(labels) > 0 and confidence > 0.75:
            row = cord[0]
            x1, x2 = int(row[0] * w), int(row[2] * w)
            pers_x_cent = x1 + (abs(x2 - x1) // 2)

            success_cnt += 1

        else:
            success_cnt = 0
            res_dict['detected'] = False
            event_list['detected'].clear()
        if success_cnt > 2:
            res_dict['detected'] = True
            event_list['detected'].set()
            ms.move_relative(pers_x_cent - cur_x, 0)

        sleep(0.2)
    print(f"Process {current_process().name} finished")


def always_f(res_dict, event_list):
    """
    Continuously press the 'f' key.

    Args:
        res_dict (multiprocessing.Manager.dict): Shared dictionary for storing the results.
        event_list (multiprocessing.Manager.dict): Shared dictionary for managing events.
    """
    print(f"Process {current_process().name} started")
    while not event_list['success'].is_set():
        pg.press('f')
        sleep(0.05)
    print(f"Process {current_process().name} finished")


def press_f(res_dict, event_list, screenshots, map_name):
    """
    define the 'F' button during searching vehicles or loot.

    Args:
        res_dict (multiprocessing.Manager.dict): Shared dictionary for storing the results.
        event_list (multiprocessing.Manager.dict): Shared dictionary for managing events.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        map_name (str): Name of the map being played.
    """
    print(f"Process {current_process().name} started")
    found_f = False
    if gliders_or_cars[map_name] == 'car':
        template = cv2.imread(r'screenshots\cut\press_f_auto_cut.png', 0)
        threshhold = 0.65
    else:
        template = cv2.imread(r'screenshots\cut\press_f_glider_cut.png', 0)
        threshhold = 0.45

    fake_template_1 = cv2.imread(r'screenshots\cut\open_door_cut.png', 0)
    # fake_template_1 = rescale_template(fake_template_1)

    while not found_f:
        if event_list['final'].is_set() or event_list['button'].is_set():
            break
        screenshot = screenshots['color']
        im_center = screenshot[rescale_w(810):rescale_w(865), rescale_h(1420):rescale_h(1534), 0]
        # template = rescale_template(template)
        is_there, center = is_part(im_center, template, threshhold)
        if is_there:
            print('f found, checking')
            is_there, center = is_part(im_center, fake_template_1, 0.85)
            if not is_there:
                print('f confirmed')
                res_dict['success'] = True
                print('success from press_f')
                event_list['success'].set()
                found_f = True
        sleep(0.1)
    print(f"Process {current_process().name} finished")


def duck_and_f(res_dict):
    """
    When the vehicle is detected, bot goes after duck to keep more image on the screen.

    Args:
        res_dict (dict): Shared dictionary for storing the results.
    """
    pg.press('ctrl')
    res_dict['duck'] = True
    sleep(4)
    pg.press('ctrl')
    res_dict['duck'] = False


def keep_tag(res_dict, event_list, screenshots):
    """
    Monitor the presence of a tag in a compass area to detect that bot run through the target.

    Args:
        res_dict (multiprocessing.Manager.dict): Shared dictionary for storing the results.
        event_list (multiprocessing.Manager.dict): Shared dictionary for managing events.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
    """
    print(f"Process {current_process().name} started")
    TAGS_TO_DEQ = 3

    x1_close, x2_close, y1_close, y2_close = rescale_w(28), rescale_w(51), rescale_h(1110), rescale_h(1410)
    sleep(5)
    tags_cnt = 0
    while True:
        if event_list['success'].is_set() or event_list['final'].is_set() or event_list['button'].is_set():
            break
        if not event_list['keep_tag_center'].is_set():
            sleep(0.3)
            continue

        template = screenshots['color'][x1_close:x2_close, y1_close:y2_close, :]
        wide = 15
        segments_n = (y2_close - y1_close) // wide
        start = 0
        pos = -1
        for part in range(segments_n):
            ch = template[:, start:start + wide, :]
            res = define_color_new(ch)
            if np.array_equal(res, np.array([33, 237, 251])):
                pos = part
                break
            start += wide
        if pos == -1:
            tags_cnt += 1
        if tags_cnt > TAGS_TO_DEQ:
            print('tag out!!!wflw')
            if not event_list['keep_tag_center'].is_set():
                sleep(0.3)
                continue
            res_dict['tag_in'] = False
            event_list['tag_in'].set()
            tags_cnt = 0
        sleep(0.5)
    print(f"Process {current_process().name} finished")


def run(res_dict, event_list):
    """
    Keep running while not signal is set

    Args:
        res_dict (multiprocessing.Manager.dict): Shared dictionary for storing the results.
        event_list (multiprocessing.Manager.dict): Shared dictionary for managing events.
    """
    print(f"Process {current_process().name} started")
    while not event_list['success'].is_set() and not event_list['final'].is_set():
        if res_dict['run']:
            pg.keyDown('l')
            pg.keyDown('w')
    print(f"Process {current_process().name} finished")


def timing(res_dict, event_list):
    """
    Manage timing and set a timeout condition.

    Args:
        res_dict (dict): Shared dictionary for storing the results.
        event_list (dict): Shared dictionary for managing events.
    """
    print(f"Process {current_process().name} started")
    while True:
        if event_list['final'].is_set() or event_list['button'].is_set():
            break
        if not event_list['timer'].is_set():
            sleep(60)
            print('timouuuuuuuuut')
            res_dict['time_out'] = True
            event_list['timer'].set()

    print(f"Process {current_process().name} finished")


def coordinator(res_dict, event_list, screenshots):
    """
    Coordinate the execution of different processes based on events.

    Args:
        res_dict (dict): Shared dictionary for storing the results.
        event_list (dict): Shared dictionary for managing events.
        screenshots (dict): Dictionary containing the screenshots.
    """
    print(f"Process {current_process().name} started")
    while not event_list['success'].is_set():
        if event_list['button'].is_set():
            event_list['final'].set()
            event_list['success'].set()
            break
        res_dict['tag_in'] = True
        event_list['tag_in'].clear()
        res_dict['run'] = True
        res_dict['time_out'] = False
        event_list['timer'].clear()
        event_list['keep_tag_center'].set()
        while not event_list['timer'].is_set() and not event_list['success'].is_set():
            # print('coord2')
            if event_list['button'].is_set():
                event_list['final'].set()
                break

            res_dict['run'] = True
            res_dict['detect'] = True
            event_list['keep_tag_center'].set()

            if event_list['detected'].is_set():
                event_list['keep_tag_center'].clear()
                duck_and_f(res_dict)
                res_dict['detected'] = False
                event_list['detected'].clear()
                sleep(1.5)
                if event_list['success'].is_set() or event_list['final'].is_set() or event_list['button'].is_set():
                    break

                res_dict['run'] = False
                res_dict['detect'] = False

                sleep(1)
                pg.press('f')

        res_dict['run'] = False
        res_dict['detect'] = False

        sleep(1)
        pg.press('f')
        sleep(0.3)
    print(f"Process {current_process().name} finished")


def main_search(button_event, screenshots, map_name):
    """
    Main function for searching and coordinating processes based on events.

    Args:
        button_event (Event): Event indicating the button press.
        screenshots (dict): Dictionary containing the screenshots.
        map_name (str): Name of the map being played.
    """
    if not button_event.is_set():
        final_end_event = Event()
        success_event = Event()
        timout_event = Event()
        detected_event = Event()
        tag_in_event = Event()
        keep_tag_center = Event()
        event_list = {'success': success_event,
                      'timer': timout_event,
                      'detected': detected_event,
                      'tag_in': tag_in_event,
                      'final': final_end_event,
                      'button': button_event,
                      'keep_tag_center': keep_tag_center}

        manager = Manager()
        res_dict = manager.dict()
        res_dict['run'] = False
        res_dict['detect'] = False
        res_dict['detected'] = False
        res_dict['duck'] = False
        res_dict['find_tag'] = False
        res_dict['success'] = False
        res_dict['tag_in'] = False
        res_dict['time_out'] = False
        find_tag_new(screenshots)
        sleep(0.6)
        find_tag_new(screenshots)

    if not button_event.is_set():
        processes = []
        detection_proc = Process(target=search_vehicle, args=(res_dict, event_list, screenshots, ), name='search_vehicle')
        f_proc = Process(target=always_f, args=(res_dict, event_list, ), name='always_f')
        pressed_f = Process(target=press_f, args=(res_dict, event_list, screenshots, map_name, ), name='pressed f&')
        keep_tag_cent_proc = Process(target=keep_tag, args=(res_dict, event_list, screenshots,), name='keep tag center')
        timing_proc = Process(target=timing, args=(res_dict, event_list,), name='timing')
        run_proc = Process(target=run, args=(res_dict, event_list,), name='run')
        coord_proc = Process(target=coordinator, args=(res_dict, event_list, screenshots, ), name='coord')

        processes.append(detection_proc)
        processes.append(f_proc)
        processes.append(pressed_f)
        processes.append(timing_proc)
        processes.append(run_proc)
        processes.append(coord_proc)
        processes.append(keep_tag_cent_proc)

    if not button_event.is_set():
        for p in processes:
            p.start()
        event_list['final'].wait()

        for p in processes:
            p.join()

        sleep(2)
        pg.press('f')
        ms.move_relative(rescale_w(-1000), 0)
        pg.keyDown('shift')
        pg.keyDown('w')
        sleep(7)
        pg.keyUp('shift')
        pg.keyUp('w')
        pg.press('x')
