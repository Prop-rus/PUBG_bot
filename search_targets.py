from multiprocessing import Process, Event, Manager, current_process
from time import sleep
import pyautogui as pg
import cv2
import torch
import numpy as np

from is_part_image import is_part

from my_utils import find_tag_new, define_color_new, stop, look_around, steps_on_timing, rescale_w, rescale_h

from collections import deque
from configs.config import gliders_or_cars, w, h
from mouse_control import MouseControls

ms = MouseControls()


def search_vehicle(res_dict, event_list, screenshots, vihicle_classes =[0,1,2,3,4]):
    print(f"Process {current_process().name} started")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',
                           force_reload=False)
    model.eval()
    model.cuda()
    model.imgsz = (1280, 1280)
    model.multi_label = False
    model.max_det = 1
    model.conf_thres = 0.6
    model.iou_thres = 0.6
    model.classes = vihicle_classes  # car, motorcycle, bus, truck
    cur_x, cur_y = w // 2, h // 2
    success = False
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
        ind_cars = np.where(np.isin(labels, vihicle_classes))

        labels = labels[ind_cars]
        cord = cord[ind_cars]
        confidence = confidence[ind_cars]
        if len(labels) > 0 and confidence > 0.75:
            row = cord[0]
            x1, y1, x2, y2 = int(row[0] * w), int(row[1] * h), int(row[2] * w), int(row[3] * h)
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
    print(f"Process {current_process().name} started")
    while not event_list['success'].is_set():
        pg.press('f')
        sleep(0.05)
    print(f"Process {current_process().name} finished")


def press_f(res_dict, event_list, screenshots, map_name):
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
        if event_list['final'].is_set() or event_list['button'].is_set() :
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

    pg.press('ctrl')
    res_dict['duck'] = True
    sleep(4)
    pg.press('ctrl')
    res_dict['duck'] = False

def keep_tag(res_dict, event_list, screenshots):
    print(f"Process {current_process().name} started")
    TAGS_TO_DEQ = 3
    tag_in = True

    x1_close, x2_close, y1_close, y2_close = rescale_w(28), rescale_w(51), rescale_h(1110), rescale_h(1410)
    tags_list = deque(TAGS_TO_DEQ*[0], TAGS_TO_DEQ)
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
            ch = template[:, start:start+wide, :]
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
    print(f"Process {current_process().name} started")
    while not  event_list['success'].is_set() and not  event_list['final'].is_set():
        if res_dict['run']:
            pg.keyDown('l')
            pg.keyDown('w')
    print(f"Process {current_process().name} finished")


def timing(res_dict, event_list):
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

            elif event_list['tag_in'].is_set():
                event_list['keep_tag_center'].clear()
                res_dict['run'] = False
                stop()
                res_dict['detect'] = False
                look_around(res_dict, event_list)
                find_tag_new(screenshots)
                sleep(0.6)
                find_tag_new(screenshots)
                event_list['tag_in'].clear()
            sleep(0.3)
        if not event_list['success'].is_set():
            res_dict['run'] = False
            stop()
            steps_on_timing()
            find_tag_new(screenshots)
            sleep(0.6)
            find_tag_new(screenshots)
    if not event_list['final'].is_set():
        res_dict['run'] = False
        stop()
        event_list['keep_tag_center'].clear()
        event_list['timer'].set()
        event_list['final'].set()

    event_list['final'].set()
    event_list['success'].set()
    event_list['timer'].set()
    event_list['keep_tag_center'].clear()
    print(f"Process {current_process().name} finished")


def main_search(button_event, screenshots, map_name):

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
