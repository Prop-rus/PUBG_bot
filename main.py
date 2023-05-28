from multiprocessing import Process, Event, Manager, current_process, Queue
from time import sleep
import pyautogui as pg
import cv2
import logging
from logging.handlers import QueueHandler

from configs.config import buttons_to_click, maps_to_glide
from is_part_image import is_part
from mouse_control import MouseControls
from glider_actions import glider_actions
from my_utils import search_f_key, define_map, take_screenshot, rescale_w, rescale_h


ms = MouseControls()
pg.FAILSAFE = False


def detect_buttons(button_event, screenshots, queue):
    """
    Detect and interact with buttons in the game.

    Args:
        button_event (multiprocessing.Event): Event to signal the detection of a button.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        queue (multiprocessing.Queue): Queue for logging messages.

    Returns:
        None
    """
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')
    print("Buttons started")
    while not button_event.is_set():
        sleep(1)
        cv_imageObj = screenshots['gray']

        for but in buttons_to_click:
            template = cv2.imread(fr'screenshots\cut\{but}.png', 0)
            # template = rescale_template(template)
            is_there, center = is_part(cv_imageObj, template, 0.95)  # Check if the button template is present
            if is_there:
                print('Is there button:', but)
                if but == 'go_lobby_cut':
                    print('Confirm click')
                    pg.leftClick(rescale_w(1125), rescale_h(885), duration=1)  # Click on the button

                if but == 'start_game':
                    # Perform a sequence of button clicks
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

                print('Button:', but, 'Center:', center)
                print('Button click:', but)
                pg.leftClick(center[0], center[1], duration=1)  # Click on the button
                pg.leftClick(center[0], center[1], duration=1)
                pg.moveTo(0, 0)
                button_event.set()  # Set the button event to signal completion

                print('Is button event true?', button_event.is_set())
                return
    logger.info(f'Child {process.name} done.')
    print("Buttons finished")


def car_actions(button_event, screenshots):
    # actions on the maps, where car spawn places are not fixed. Not needed to implement yet
    pass

def suicide(button_event, screenshots):
    # TODO actions for maps without vehicles
    #  and fixed loot such as Paramo
    pass


def ingame_acting(button_event, screenshots, queue):
    """
    Perform in-game actions based on the current state.

    Args:
        button_event (multiprocessing.Event): Event to signal the detection of a button.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        queue (multiprocessing.Queue): Queue for logging messages.

    Returns:
        None
    """
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')
    print("In-game started")

    f_found = False
    while not f_found and not button_event.is_set():
        f_found = search_f_key(screenshots)  # Check if the "F" key is found in the screenshots
    if not button_event.is_set():
        map_name = define_map(screenshots)  # Determine the map name
        print(map_name)
        if map_name is not None and not button_event.is_set():
            if map_name in maps_to_glide.keys() and not button_event.is_set():
                glider_actions(map_name, button_event, screenshots)  # Perform glider actions
            else:
                car_actions(button_event, screenshots)  # Perform car actions
        else:
            suicide(button_event, screenshots)  # Perform suicide action
    logger.info(f'Child {process.name} done.')
    print('In-game finished')


def take_screenshot_always(button_event, screenshots, queue):
    """
    Continuously capture screenshots while the button event is not set.

    Args:
        button_event (multiprocessing.Event): Event to signal the detection of a button.
        screenshots (multiprocessing.Manager.dict): Shared dictionary for storing screenshots.
        queue (multiprocessing.Queue): Queue for logging messages.

    Returns:
        None
    """
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.info(f'Child {process.name} starting.')

    print("Screenshots started")
    while not button_event.is_set():
        color, gray = take_screenshot() # Capture color and grayscale screenshots
        screenshots['color'] = color
        screenshots['gray'] = gray
        sleep(0.2)
    print("Screenshots finished")
    logger.info(f'Child {process.name} done.')


def logger_process(queue):
    """
    Process that handles log messages from the queue and logs them to a file.

    Args:
        queue (multiprocessing.Queue): Queue for logging messages.

    Returns:
        None
    """
    logger = logging.getLogger('app')
    fh = logging.FileHandler('logs/log.txt', mode='a', encoding=None, delay=False)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def main():
    """
    Main entry point of the script.

    Returns:
        None
    """
    queue = Queue()
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    logger_p = Process(target=logger_process, args=(queue,))
    logger_p.start()
    logger.info('Main process started.')
    while True:
        print('New game!')
        processes = []
        button_event = Event()
        manager = Manager()
        screenshots = manager.dict()
        color, gray = take_screenshot()
        screenshots['color'] = color
        screenshots['gray'] = gray
        screen_proc = Process(target=take_screenshot_always,
                              args=(button_event, screenshots, queue,),
                              name='take screens')
        detect_buttons_proc = Process(target=detect_buttons,
                                      args=(button_event, screenshots, queue, ),
                                      name='detect buttons')
        ingame_proc = Process(target=ingame_acting,
                              args=(button_event, screenshots, queue, ),
                              name='ingame acting')
        processes.append(screen_proc)
        processes.append(detect_buttons_proc)

        for p in processes:
            p.start()
        processes.append(ingame_proc)
        ingame_proc.start()
        button_event.wait()  # <- Blocks until condition is met

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
