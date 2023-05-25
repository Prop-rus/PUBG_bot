from my_utils import rescale_w, rescale_h


w, h = 2560, 1440

map_list = ['vikendi', 'erangel', 'miramar', 'taego', 'sanok', 'deston', 'karakin']

maps_to_glide = {'deston': (rescale_w(1036), rescale_h(542)),
                 'vikendi': (rescale_w(1312), rescale_h(777)),
                 'erangel': (rescale_w(1472), rescale_h(812)),
                 'taego': (rescale_w(1274), rescale_h(844)),
                 'miramar': (rescale_w(929), rescale_h(720)),
                 'sanok': (rescale_w(1516), rescale_h(975)),
                 'karakin': (rescale_w(1165), rescale_h(760))
                 }

# destinations of min closure of the plane to target pont in px when bot has to jump
maps_destinations = {'deston':{'short': rescale_w(300),
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

# some maps are needed to point exact target place, such as boxes with loot on it
scrolled_coordinates = {'sanok': (rescale_w(1549), rescale_h(980))}

# targets on different maps are different too
gliders_or_cars = {'taego': 'glider',
                   'vikendi': 'glider',
                   'erangel': 'car',
                   'miramar': 'car',
                   'deston': 'car',
                   'sanok': 'car',
                   'karakin': 'car'
                   }

# just do nothing maps
suicide_maps = ['paramo']

# buttons int the menu, that are scaned as game finished
buttons_to_click = ['start_game', 'go_lobby_cut', 'confirm_cut',
                     'next_cut', 'finish_cut', 'reconnect_cut']

# tags in the orientation board are different when plaing in team. So you have to switch to another screenshot
tag = 'single'  # team or single

TAKE_LAST_N_MEASUERS = 15

# steps to move when timing finished
steps_right, steps_back = 5, 2