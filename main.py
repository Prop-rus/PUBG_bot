from multiprocessing import Process, Event, Manager, current_process
from time import sleep
import pyautogui as pg
import cv2
import torch
import numpy as np
import math
from statistics import mean
from configs.config import buttons_to_click, maps_to_glide
from is_part_image import is_part
from look import MouseControls

from my_utils import search_f_key, define_map, take_screnshot, press_for_long

from collections import deque
from configs.config import tag, maps_destinations, scrolled_coordinates, gliders_or_cars
import random
import atexit

from multiprocessing import Queue
from logging.handlers import QueueHandler
import logging

ms = MouseControls()
pg.FAILSAFE = False
TAKE_LAST_N_MEASUERS = 15
w, h = 2560, 1440

def watch_ahead():
    ms.move_relative(0, 3000)
    ms.move_relative(0, -1500)

def define_color_new(image):

    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def detect_ground(event_list, screenshots):
    print(f"Process {current_process().name} started")

    while True:
        # print('ground')
        cv_imageObj = screenshots['gray']
        template = cv2.imread(r'screenshots\cut\underwater_cut.png', 0)
        is_there, center = is_part(cv_imageObj, template, 0.95)
        if is_there:
            print('water')
            pg.keyDown('space')
            sleep(5)
            pg.keyUp('space')
            event_list['ground'].set()
            break

        template = cv2.imread(r'screenshots\cut\stand_cut.png', 0)
        is_there, center = is_part(cv_imageObj, template, 0.95)
        if is_there:
            print('ground')
            event_list['ground'].set()
            # print('on the ground')
            break
    print(f"Process {current_process().name} finished")

def forward_and_detect_tag(res_dict, event_list,  mean_dist, map_name):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
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

def find_tag(screenshots, event_list=None):

    # atexit.register(ms.stop_mouse)
    done = False
    if tag == 'team':
        template = cv2.imread(rf'screenshots\compas/true_tag_team.png')
    else:
        template = cv2.imread(rf'screenshots\compas/true_tag.png')
    x1_close, x2_close, y1_close, y2_close = 24,55,1264,1297
    x1, x2, y1, y2 = 24,55,1266,1400
    close = False
    mv_rate = 150

    cv_imageObj = screenshots['color'][x1:x2, y1:y2, :]
    is_there, _ = is_part(cv_imageObj, template, 0.55)
    if is_there:
        close = True
        x1, x2, y1, y2 = x1_close, x2_close, y1_close, y2_close
        mv_rate = 20
    steps = 0
    look_down = False
    while not done:
        # print('find tag')
        if event_list is not None:
            if event_list['detected'].is_set():
                print('find tag broke detekted')
                break
            if event_list['success'].is_set() or event_list['final'].is_set():
                print('find tag broke success')
                break

        cv_imageObj = screenshots['gray'][x1:x2, y1:y2]
        is_there, _ = is_part(cv_imageObj, template, 0.55)
        if is_there:
            if close:
                print('tag')
                done = True
            else:
                close = True
                mv_rate = 20
                x1, x2, y1, y2 = x1_close, x2_close, y1_close, y2_close
        else:
            ms.move_relative(mv_rate, 0)
            steps += 1
            sleep(0.08)
            # if steps // 2000 == 0 :
            #     ms.move_relative(0, 10)
            # if steps > 10000:
            #     break


def run(res_dict, event_list):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    while not  event_list['success'].is_set() and not  event_list['final'].is_set():
        # print('run')
        if res_dict['run']:
            pg.keyDown('l')
            pg.keyDown('w')
    print(f"Process {current_process().name} finished")

def stop():

    pg.keyUp('l')
    pg.keyUp('w')

def keep_tag(res_dict, event_list, screenshots):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    TAGS_TO_DEQ = 3
    # print('start forward and detect tag')
    tag_in = True

    x1_close, x2_close, y1_close, y2_close = 28, 51, 1110, 1410
    tags_list = deque(TAGS_TO_DEQ*[0], TAGS_TO_DEQ)
    sleep(5)
    tags_cnt = 0
    while True:
        # print('keep tag')
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
            # print('tag not detected', tags_list)
            res_dict['tag_in'] = False
            event_list['tag_in'].set()
            tags_cnt = 0
        sleep(0.5)
    print(f"Process {current_process().name} finished")

def look_around(res_dict, event_list, screenshots):
    # atexit.register(ms.stop_mouse)
    moves = [20,50,100]
    for i in range(20):
        # print('look around 1')
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
    # for i in range(20):
    #     ms.move_relative(20, 0)
    #     sleep(0.1)
    # find_tag(screenshots, event_list)
    if event_list['detected'].is_set()  or event_list['final'].is_set():
        return
    res_dict['tag_in'] = True
    event_list['tag_in'].clear()


def press_f(res_dict, event_list, screenshots, map_name):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    found_f = False
    if gliders_or_cars[map_name] == 'car':
        template = cv2.imread(r'screenshots\cut\press_f_auto_cut.png', 0)
        threshhold = 0.65
    else:
        template = cv2.imread(r'screenshots\cut\press_f_glider_cut.png', 0)
        threshhold = 0.5

    fake_template_1 = cv2.imread(r'screenshots\cut\open_door_cut.png', 0)

    while not found_f:
        # print('press f')
        if event_list['final'].is_set() or event_list['button'].is_set() :
            break
        screenshot = screenshots['color']
        im_center = screenshot[810:865, 1420:1534, 0]
        is_there, center = is_part(im_center, template, threshhold)
        if is_there:
            print('f found, checking')
            is_there, center = is_part(im_center, fake_template_1, 0.9)
            if not is_there:
                print('f confirmed')
                res_dict['success'] = True
                print('success from press_f')
                event_list['success'].set()
                found_f = True
        sleep(0.1)
    print(f"Process {current_process().name} finished")

def search_vehicle(res_dict, event_list, screenshots,  vihicle_classes =[0,1,2,3,4]):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/best.pt',
                           force_reload=False)
    model.eval()
    model.cuda()
    model.imgsz = (1280, 1280)
    model.multi_label = False
    model.max_det = 1
    model.conf_thres = 0.6
    model.iou_thres = 0.6
    # vihicle_classes = [2, 3]
    # vihicle_classes = [0, 1, 2, 3]
    model.classes = vihicle_classes  # car, motorcycle, bus, truck
    cur_x, cur_y = w // 2, h // 2
    success = False
    success_cnt = 0
    while not event_list['success'].is_set():
        # print('model DL')
        if not res_dict['detect']:
            sleep(0.3)
            continue
        if event_list['success'].is_set() or event_list['final'].is_set():
            break
        screenshot = screenshots['color']
        results = model(screenshot.copy())
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        # print(labels)
        confidence = results.xyxyn[0][:, -2].cpu().numpy()
        # print(labels)
        ind_cars = np.where(np.isin(labels, vihicle_classes))

        labels = labels[ind_cars]
        cord = cord[ind_cars]
        confidence = confidence[ind_cars]
        if len(labels) > 0 and confidence > 0.75:
            # print('detected class', labels, 'confidence', confidence)

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


def timing(res_dict, event_list):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    while True:
        # print('time')
        if event_list['final'].is_set() or event_list['button'].is_set():
            break
        if not event_list['timer'].is_set():
            sleep(60)
            print('timouuuuuuuuut')
            res_dict['time_out'] = True
            event_list['timer'].set()

    print(f"Process {current_process().name} finished")


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
    # steps_right += 2
    # steps_back += 1
    print('steps end')


def always_f(res_dict, event_list):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    while not event_list['success'].is_set():
        pg.press('f')
        # print('f')
        sleep(0.05)
    print(f"Process {current_process().name} finished")


def coordinator(res_dict, event_list, screenshots):
    print(f"Process {current_process().name} started")
    # atexit.register(ms.stop_mouse)
    while not event_list['success'].is_set():
        # print('coord')
        if event_list['button'].is_set():
            event_list['final'].set()
            event_list['success'].set()
            break
        # find_tag(screenshots, event_list)
        res_dict['tag_in'] = True
        event_list['tag_in'].clear()
        res_dict['run'] = True
        res_dict['time_out'] = False
        event_list['timer'].clear()
        event_list['keep_tag_center'].set()
        # print('##########################################################################')
        while not event_list['timer'].is_set() and not event_list['success'].is_set():
            # print('coord2')
            if event_list['button'].is_set():
               event_list['final'].set()
               break
            # print(res_dict)
            # print(event_list['timer'].is_set())
            # print('tag not found' ,event_list['tag_in'].is_set())
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
                look_around(res_dict, event_list, screenshots)
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


def find_tag_new(screenshots):
    print('find tag new start')
    # atexit.register(ms.stop_mouse)
    x1_close, x2_close, y1_close, y2_close = 28, 51, 1270, 1293
    x1, x2, y1, y2 = 28, 51, 860, 1700
    # for i in range(4):

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



#ffffffffffffffffff


def duck_and_f(res_dict):

    pg.press('ctrl')
    res_dict['duck'] = True
    sleep(4)
    pg.press('ctrl')
    res_dict['duck'] = False

def main_search(button_event, screenshots, map_name):

    # atexit.register(ms.stop_mouse)

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
        detection_proc = Process(target=search_vehicle, args=(res_dict, event_list, screenshots,), name='search_vehicle')
        f_proc = Process(target=always_f, args=(res_dict, event_list, ), name='always_f')
        pressed_f = Process(target=press_f, args=(res_dict, event_list, screenshots, map_name, ), name='pressed f&')
        # keep_tag_proc = Process(target=find_tag_new, args=(screenshots, res_dict, event_list, False,), name='keep tag')
        keep_tag_cent_proc = Process(target=keep_tag, args=(res_dict, event_list, screenshots,), name='keep tag center')
        timing_proc = Process(target=timing, args=(res_dict, event_list,), name='timing')
        run_proc = Process(target=run, args=(res_dict, event_list,), name='run')
        coord_proc = Process(target=coordinator, args=(res_dict, event_list, screenshots, ), name='coord')

        processes.append(detection_proc)
        processes.append(f_proc)
        processes.append(pressed_f)
        # processes.append(keep_tag_proc)
        processes.append(timing_proc)
        processes.append(run_proc)
        processes.append(coord_proc)
        processes.append(keep_tag_cent_proc)

    if not button_event.is_set():
        for p in processes:
            p.start()
        event_list['final'].wait()
        # for p in processes:
        #     p.terminate()
        for p in processes:
            p.join()
        # pg.keyDown('ctrl')
        # pg.press('1')
        # pg.keyUp('ctrl')
        # pg.keyDown('w')
        # sleep(3)
        # pg.keyUp('w')
        # pg.keyDown('space')
        # sleep(3)
        # pg.keyUp('space')
        sleep(2)
        pg.press('f')
        ms.move_relative(-1000, 0)
        pg.keyDown('shift')
        pg.keyDown('w')
        sleep(7)
        pg.keyUp('shift')
        pg.keyUp('w')
        pg.press('x')

def fly_over(screenshots, mean_dist, map_name, button_event):
    # atexit.register(ms.stop_mouse)

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
    detect_tag_proc = Process(target=forward_and_detect_tag, args=(res_dict, event_list,  mean_dist, map_name, ), name='forw and detect')
    detect_ground_proc = Process(target=detect_ground, args=(event_list, screenshots, ), name='ground')
    # find_tag_proc = Process(target=find_tag_new, args=(screenshots, None, event_list, True))
    processes_fly.append(detect_tag_proc)
    processes_fly.append(detect_ground_proc)
    # processes_fly.append(find_tag_proc)
    for p in processes_fly:
        p.start()

    grounded_event.wait()  # <- blocks until condition met

    # for p in processes_fly:
    #     p.terminate()
    for p in processes_fly:
        p.join()
    # ffff
    pg.keyUp('w')
    print('fly over finish')

def mark_glider(map_name):
    # atexit.register(ms.stop_mouse)
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

def wait_and_jump(x, y, map_name, model_plane, screenshots):
    # atexit.register(ms.stop_mouse)
    print('wait and jump start')
    mean_dest = 0
    if map_name == 'taego':

        sleep(15)
        print('need to jump')
        press_for_long('m')
        press_for_long('f')

        # too_far = True
        # destinations = deque(TAKE_LAST_N_MEASUERS * [0], TAKE_LAST_N_MEASUERS)
        # appended_cnt = 0
        # mean_dest = 0
        # short = maps_destinations[map_name]['short']
        # while too_far:
        #     print('wait and jump')
        #     screenshot, _ = take_screnshot()
        #     results = model_plane(screenshot.copy())
        #     labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        #     # print(labels)
        #     confidence = results.xyxyn[0][:, -2].cpu().numpy()
        #     # print(labels)
        #     if len(labels) > 0 and confidence > 0.75:
        #         # print('detected class', labels, 'confidence', confidence)
        #
        #         row = cord[0]
        #         x1, y1, x2, y2 = int(row[0] * w), int(row[1] * h), int(row[2] * w), int(row[3] * h)
        #         pers_x_cent = x1 + (abs(x2 - x1) // 2)
        #         pers_y_cent = x1 + (abs(y2 - y1) // 2)
        #
        #         destination = math.sqrt((x - pers_x_cent) ** 2 + (y - pers_y_cent) ** 2)
        #         # contour_destinations.append(destination)
        #         destinations.append(destination)
        #         appended_cnt += 1
        #         print('desintaions', destinations, mean(destinations), x, y, pers_x_cent, pers_y_cent)
        #         mean_dest = mean(destinations)
        #         if (((destination - mean_dest) > 10) or (
        #                 mean_dest < short)) and appended_cnt > TAKE_LAST_N_MEASUERS:
        #             too_far = False
        #             print('need to jump')
        #             press_for_long('m')
        #             press_for_long('f')
        #             break
        #
        #     sleep(0.5)
    else:
        too_far = True
        destinations = deque(TAKE_LAST_N_MEASUERS*[0], TAKE_LAST_N_MEASUERS)
        appended_cnt = 0
        mean_dest = 0
        short = maps_destinations[map_name]['short']
        while too_far:
            print('wait and jump')
            # screen1 = screenshots['gray']
            # cv_imageObj1 = screen1[560:0, 1425:1440]
            # sleep(0.2)
            # screen2 = screenshots['gray']
            # cv_imageObj2 = screen2[560:0,   1425:1440]
            _, cv_imageObj1 = take_screnshot(region=(565,0, 1425, 1440))
            sleep(0.2)
            _, cv_imageObj2 = take_screnshot(region=(565,0, 1425, 1440))

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
                cX = int(M["m10"] / M["m00"]) + 564
                cY = int(M["m01"] / M["m00"])
                # center = [cX, cY]

                xses.append(cX)
                yses.append(cY)


            if len(xses) > 0 and len(yses) > 0:
                diff_x = abs(max(xses) - min(xses))
                diff_y = abs(max(yses) - min(yses))
                if diff_x < 50 and diff_y < 50:
                    destination = math.sqrt((x - mean(xses)) ** 2 + (y - mean(yses)) ** 2)
                        # contour_destinations.append(destination)
                    destinations.append(destination)
                    appended_cnt += 1
                    print('desintaions', destinations, mean(destinations), x, y, xses, yses)
                    mean_dest = mean(destinations)
                    if (((destination - mean_dest) > 10) or (mean_dest < short)) and appended_cnt > TAKE_LAST_N_MEASUERS:
                        too_far = False
                        print('need to jump')
                        press_for_long('m')
                        press_for_long('f')
                        break
    find_tag_new(screenshots)
    pg.keyDown('w')
    print('wait and jump finish')
    return mean_dest

def glider_actions(map_name, button_event, screenshots, model_plane):
    # atexit.register(ms.stop_mouse)

    print('start glider actions')
    x, y = mark_glider(map_name)

    mean_dist = wait_and_jump(x, y, map_name, model_plane, screenshots)

    fly_over(screenshots, mean_dist, map_name, button_event)

    # find_tag(screenshots)

    main_search(button_event, screenshots, map_name)



def car_actions(button_event, screenshots):
    print('car actions')
    pass

def suicide(button_event, screenshots):
    pass
    # print('sucide')
    # template = cv2.imread(r'screenshots\cut\jump_cut.png', 0)
    # while True:
    #
    #     cv_imageObj = screenshots['gray']
    #     is_there, center = is_part(cv_imageObj, template, 0.85)
    #     if is_there:
    #         print('jump', center)
    #         pg.moveTo(0, 0)
    #         sleep(6)
    #         pg.keyDown('f')
    #         sleep(2)
    #         pg.keyUp('f')
    #         pg.keyDown('w')
    #         sleep(50)
    #         pg.keyUp('w')
    #         pg.press('x')
    #         break

def detect_buttons(button_event, screenshots, queue):
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # get the current process
    process = current_process()
    # report initial message
    logger.info(f'Child {process.name} starting.')
    print(f"buttons started")
    # atexit.register(ms.stop_mouse)
    while not button_event.is_set():
        # print('buttons')
        sleep(1)
        cv_imageObj = screenshots['gray']

        for but in buttons_to_click:
            template = cv2.imread(fr'screenshots\cut\{but}.png', 0)
            is_there, center = is_part(cv_imageObj, template, 0.99)
            if is_there:
                if but == 'go_lobby_cut':
                    print('confirm click')
                    pg.leftClick(1125, 885, duration=1)

                if but == 'start_game':
                    pg.leftClick(2237, 54, duration=1)


                    sleep(0.5)
                    pg.leftClick(2427, 520, duration=0.3)
                    pg.leftClick(2427, 520, duration=0.3)

                    sleep(0.5)
                    pg.leftClick(2427, 720, duration=0.3)
                    pg.leftClick(2427, 720, duration=0.3)
                    sleep(0.5)
                    pg.leftClick(2427, 920, duration=0.3)
                    pg.leftClick(2427, 920, duration=0.3)
                    sleep(0.5)
                    pg.leftClick(2427, 1120, duration=0.3)
                    pg.leftClick(2427, 1120, duration=0.3)
                    sleep(0.5)
                    pg.leftClick(1275, 1000, duration=0.3)
                    pg.leftClick(1275, 1000, duration=0.3)
                    sleep(0.5)
                    pg.leftClick(240, 1400, duration=0.3)
                    pg.leftClick(240, 1400, duration=0.3)
                print(but, center)
                print('button click', but)
                pg.leftClick(center[0], center[1], duration=1)
                pg.leftClick(center[0], center[1], duration=1)
                pg.moveTo(0, 0)
                button_event.set()

                print('button event true?', button_event.is_set())
                return
    logger.info(f'Child {process.name} done.')
    print(f"buttons finished")

def ingame_acting(button_event, screenshots, model_plane, queue):
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # get the current process
    process = current_process()
    # report initial message
    logger.info(f'Child {process.name} starting.')
    print(f"ingame started")
    # atexit.register(ms.stop_mouse)
    # sleep(3)

    f_found = False
    while not f_found and not button_event.is_set():

        f_found = search_f_key(screenshots)
        # print('f found')
    if not button_event.is_set():
        # print('f f 2')
        map_name = define_map(screenshots)
        print(map_name)
        if map_name is not None and not button_event.is_set():
            # print(list(maps_to_glide.keys()))
            if map_name in maps_to_glide.keys() and not button_event.is_set():
                glider_actions(map_name, button_event, screenshots, model_plane)
            else:
                car_actions(button_event, screenshots)
        else:
            suicide(button_event, screenshots)
    logger.info(f'Child {process.name} done.')
    print('ingame finish')

def take_screenshot_always(button_event, screenshots, queue):
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # get the current process
    process = current_process()
    # report initial message
    logger.info(f'Child {process.name} starting.')

    print(f"screens started")
    # atexit.register(ms.stop_mouse)
    while not button_event.is_set():
        # print('screen')
        color, gray = take_screnshot()
        screenshots['color'] = color
        screenshots['gray'] = gray
        sleep(0.2)
    print(f"screens finished")
    logger.info(f'Child {process.name} done.')

def logger_process(queue):
    # create a logger
    logger = logging.getLogger('app')
    # configure a stream handler
    fh = logging.FileHandler('logs/log.txt', mode='a', encoding=None, delay=False)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # run forever
    while True:
        # consume a log message, block until one arrives
        message = queue.get()
        # check for shutdown
        if message is None:
            break
        # log the message
        logger.handle(message)


def main(model_plane):
    # create the shared queue
    queue = Queue()
    # create a logger
    logger = logging.getLogger('app')
    # add a handler that uses the shared queue
    logger.addHandler(QueueHandler(queue))
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    # start the logger process
    logger_p = Process(target=logger_process, args=(queue,))
    logger_p.start()
    logger.info('Main process started.')
    # atexit.register(ms.stop_mouse)
    while True:
        print('new game!')
        processes = []
        button_event = Event()
        manager = Manager()
        screenshots = manager.dict()
        color, gray = take_screnshot()
        screenshots['color'] = color
        screenshots['gray'] = gray
        screen_proc = Process(target=take_screenshot_always, args=(button_event, screenshots, queue,), name='take screens')
        detect_buttons_proc = Process(target=detect_buttons, args=(button_event, screenshots, queue, ), name='detect buttons')
        ingame_proc = Process(target=ingame_acting, args=(button_event, screenshots, model_plane, queue, ), name='ingame acting')
        processes.append(screen_proc)
        processes.append(detect_buttons_proc)

        for p in processes:
            p.start()
        processes.append(ingame_proc)
        ingame_proc.start()
        # print(' 0 IS ALIIIVE???', ingame_proc.is_alive())
        button_event.wait()  # <- blocks until condition met
        # for p in processes:
        #     p.terminate()
        # screen_proc.terminate()
        # detect_buttons_proc.terminate()
        # print('1 IS ALIIIVE???', ingame_proc.is_alive())


        for p in processes:
            p.join()
        # print('2 IS ALIIIVE???', ingame_proc.is_alive())


model_plane = None
# model_plane = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp19/weights/best.pt',
#                            force_reload=False)
# model_plane.eval()
# model_plane.cuda()
# model_plane.imgsz = (1280, 1280)
# model_plane.multi_label = False
# model_plane.max_det = 1
# model_plane.conf_thres = 0.6
# model_plane.iou_thres = 0.6


if __name__ == '__main__':
    # atexit.register(ms.stop_mouse)
    main(model_plane)