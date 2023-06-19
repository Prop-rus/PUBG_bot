
w, h = 2560, 1440
# w, h = 1920, 1080


def rescale_w(coord):
    '''
    initially all coordinates were figured out on 2K resolution.
    this will rescale to another defind resolution
    '''
    return (w * coord) // 2560


def rescale_h(coord):
    '''
    initially all coordinates were figured out on 2K resolution.
    this will rescale to another defind resolution
    '''
    return (h * coord) // 1440


map_list = ['vikendi', 'erangel', 'miramar', 'taego', 'sanok', 'deston', 'karakin', 'paramo']

maps_to_glide = {'deston': (rescale_w(1200), rescale_h(730)),
                 'vikendi': (rescale_w(1282), rescale_h(640)),
                 'erangel': (rescale_w(1347), rescale_h(662)),
                 'taego': (rescale_w(1333), rescale_h(783)),
                 'miramar': (rescale_w(1265), rescale_h(629)),
                 'sanok': (rescale_w(1395), rescale_h(800)),
                 'karakin': (rescale_w(1338), rescale_h(789)),
                 'paramo': (rescale_w(1167), rescale_h(856))
                 }

# destinations of min closure of the plane to target pont in px when bot has to jump
maps_destinations = {'deston': {'short': rescale_w(300),
                                'long': rescale_w(350)},
                     'vikendi': {'short': rescale_w(300),
                                 'long': rescale_w(350)},
                     'erangel': {'short': rescale_w(300),
                                 'long': rescale_w(350)},
                     'taego': {'short': rescale_w(300),
                               'long': rescale_w(350)},
                     'miramar': {'short': rescale_w(300),
                                 'long': rescale_w(350)},
                     'sanok': {'short': rescale_w(620),
                               'long': rescale_w(740)},
                     'karakin': {'short': rescale_w(700),
                                 'long': rescale_w(1000)}
                     }


# buttons int the menu, that are scaned as game finished
buttons_to_click = ['start_game', 'go_lobby_cut', 'confirm_cut',
                     'next_cut', 'finish_cut', 'reconnect_cut', 'ok_cut',
                    'continue_cut', 'reload_lobby_cut']

# tags in the orientation board are different when plaing in team. So you have to switch to another screenshot
tag = 'single'  # team or single

TAKE_LAST_N_MEASUERS = 15

