from multiprocessing import Process, Event, Manager, current_process
from time import sleep
import pyautogui as pg
import cv2

from configs.config import buttons_to_click, maps_to_glide
from is_part_image import is_part
from mouse_control import MouseControls
from glider_actions import glider_actions
from my_utils import search_f_key, define_map, take_screnshot, rescale_w, rescale_h


from multiprocessing import Queue
from logging.handlers import QueueHandler
import logging

ms = MouseControls()
pg.FAILSAFE = False


def car_actions(button_event, screenshots):
    print('car actions')
    pass

def suicide(button_event, screenshots):
    pass


def detect_buttons(button_event, screenshots, queue):
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')
    print(f"buttons started")
    while not button_event.is_set():
        sleep(1)
        cv_imageObj = screenshots['gray']

        for but in buttons_to_click:
            template = cv2.imread(fr'screenshots\cut\{but}.png', 0)
            is_there, center = is_part(cv_imageObj, template, 0.99)
            if is_there:
                if but == 'go_lobby_cut':
                    print('confirm click')
                    pg.leftClick(rescale_w(1125), rescale_h(885), duration=1)

                if but == 'start_game':
                    pg.leftClick(rescale_w(2237), rescale_h(54), duration=1)

                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(2427), rescale_h(520), duration=0.3)

                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(2427), rescale_h(720), duration=0.3)
                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(2427), rescale_h(920), duration=0.3)
                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(2427), rescale_h(1120), duration=0.3)
                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(1275), rescale_h(1000), duration=0.3)
                    sleep(0.5)
                    for _ in range(2):
                        pg.leftClick(rescale_w(240), rescale_h(1400), duration=0.3)
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
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')
    print(f"ingame started")

    f_found = False
    while not f_found and not button_event.is_set():

        f_found = search_f_key(screenshots)
    if not button_event.is_set():
        map_name = define_map(screenshots)
        print(map_name)
        if map_name is not None and not button_event.is_set():
            if map_name in maps_to_glide.keys() and not button_event.is_set():
                glider_actions(map_name, button_event, screenshots, ms)
            else:
                car_actions(button_event, screenshots)
        else:
            suicide(button_event, screenshots)
    logger.info(f'Child {process.name} done.')
    print('ingame finish')

def take_screenshot_always(button_event, screenshots, queue):
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')

    print(f"screens started")
    while not button_event.is_set():
        color, gray = take_screnshot()
        screenshots['color'] = color
        screenshots['gray'] = gray
        sleep(0.2)
    print(f"screens finished")
    logger.info(f'Child {process.name} done.')

def logger_process(queue):
    logger = logging.getLogger('app')
    fh = logging.FileHandler('logs/log.txt', mode='a', encoding=None, delay=False)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def main(model_plane):
    queue = Queue()
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    logger_p = Process(target=logger_process, args=(queue,))
    logger_p.start()
    logger.info('Main process started.')
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
        button_event.wait()  # <- blocks until condition met

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()