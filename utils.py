import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import numpy as np
from bs4 import BeautifulSoup
from colour import Color
import copy
import math
import re
import time

from consts import QWERTY, THUMBS, COORDS

CACHE = {}


def cleanhtml(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    spans = soup.find_all('span')
    lowercase = ''.join([i.text.replace('Пользователь 2: ', '').replace('Пользователь 1: ', '') for i in spans]).lower()
    return re.sub('[^а-я]+', '', lowercase)


def generate_strokes(sample, QWERTY):
    zones = {}
    for idr, row in enumerate(QWERTY):
        for idk, key in enumerate(row):
            zones[key] = THUMBS[idr][idk]
    strokes = {}
    stroke = ''
    for idx, char in enumerate(sample):
        current_zone = zones[char]
        stroke += char
        if idx + 1 < len(sample) and zones[sample[idx + 1]] != current_zone:
            r_stroke = stroke[::-1]
            if stroke in strokes:
                strokes[stroke]["count"] += 1
            elif r_stroke in strokes:
                strokes[r_stroke]["count"] += 1
            else:
                strokes[stroke] = {"zone": current_zone, "count": 1}
            stroke = ''
        if idx + 1 == len(sample):
            r_stroke = stroke[::-1]
            if stroke in strokes:
                strokes[stroke]["count"] += 1
            elif r_stroke in strokes:
                strokes[r_stroke]["count"] += 1
            else:
                strokes[stroke] = {"zone": current_zone, "count": 1}
    return strokes


def calculateDistance(x1,y1,x2,y2):
     global CACHE
     if f"{x1}{y1}{x2}{y2}" in CACHE:
         return CACHE[f"{x1}{y1}{x2}{y2}"]
     if f"{x2}{y2}{x1}{y1}" in CACHE:
         return CACHE[f"{x2}{y2}{x1}{y1}"]
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     CACHE[f"{x1}{y1}{x2}{y2}"] = dist
     return dist


def finger_heatmap(finger_distances):
    return [[
        finger_distances['ЛМ'],
        finger_distances['ЛБ'],
        finger_distances['ЛС'],
        finger_distances['ЛУ'],
        finger_distances['ПУ'],
        finger_distances['ПС'],
        finger_distances['ПБ'],
        finger_distances['ПМ']
    ]]


def shift_row(c, row_num, value):
    new_coords = copy.deepcopy(c)
    for idx, cell in enumerate(new_coords[row_num]):
        new_coords[row_num][idx][0] = new_coords[row_num][idx][0] + value
    return new_coords


def shift_col(c, col_num, value):
    new_coords = copy.deepcopy(c)
    for idx, row in enumerate(new_coords):
        new_coords[idx][col_num][1] = new_coords[idx][col_num][1] + value
    return new_coords


def get_mapper(c, k):
    text_mapper = {
        item: {
            'x': c[idx][idy][0],
            'y': c[idx][idy][1],
            'thumb': THUMBS[idx][idy]
        } for idx, sublist in enumerate(k) for idy, item in enumerate(sublist)
    }
    # print(json.dumps(text_mapper, indent=2, ensure_ascii=False))
    return text_mapper


def draw_keyboard(coords, QWERTY):
    x = [i[0] for i in [item for sublist in coords for item in sublist]]
    y = [i[1] for i in [item for sublist in coords for item in sublist]]
    n = [item for sublist in QWERTY for item in sublist]

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker=",", s=620, color=(0.5, 0.5, 0.5))
    ax.set_title('Координаты клавиш', fontsize=10)
    ax.set_aspect('equal', 'box')

    # Or if you want different settings for the grids:
    major_ticks = np.arange(-20, 210, 20)
    minor_ticks = np.arange(-20, 210, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.axis([-12, 210, -12, 48])

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]), color=(1, 1, 1))


def get_keyboard(coords, QWERTY):
    x = [i[0] for i in [item for sublist in coords for item in sublist]]
    y = [i[1] for i in [item for sublist in coords for item in sublist]]
    n = [item for sublist in QWERTY for item in sublist]

    fig, ax = plt.subplots()
    ax.scatter(x, y, marker=",", s=620, color=(0.5, 0.5, 0.5))
    ax.set_title('Координаты клавиш', fontsize=10)
    ax.set_aspect('equal', 'box')

    # Or if you want different settings for the grids:
    major_ticks = np.arange(-20, 210, 20)
    minor_ticks = np.arange(-20, 210, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.axis([-12, 210, -12, 48])

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]), color=(1, 1, 1))

    return ax


def count_presses(text):
    press_count = {}
    for idx, char in enumerate(text):
        if char not in press_count:
            press_count[char] = 1
        else:
            press_count[char] += 1
    return press_count


def press_heatmap(presses_counts, QWERTY):
    return [[presses_counts[item] if item in presses_counts else 0 for item in row] for row in QWERTY]


def zone_distances(zone, press_count):
    keys = []
    default_position = {
        'ЛМ': COORDS[1][0], 
        'ЛБ': COORDS[1][1],
        'ЛС': COORDS[1][2],
        'ЛУ': COORDS[1][3],
        'ПУ': COORDS[1][6],
        'ПС': COORDS[1][7],
        'ПБ': COORDS[1][8],
        'ПМ': COORDS[1][9],
    }
    for idr, row in enumerate(QWERTY):
        for idk, key in enumerate(row):
            if THUMBS[idr][idk] == zone and len(QWERTY[idr][idk]) > 0:
                x1, y1 = default_position[zone][0], default_position[zone][1]
                x2, y2 = COORDS[idr][idk][0], COORDS[idr][idk][1]
                distance = calculateDistance(x1, y1, x2, y2)
                keys.append({
                    "symbol": QWERTY[idr][idk], 
                    "distance": distance, 
                    "press_count": press_count[QWERTY[idr][idk]]
                })
    return sorted(keys, key=lambda i: i["press_count"], reverse=True)


def distance_deltas(distance, distance_1):
    sum = 0
    for k, v in distance.items():
        delta = v - distance_1[k]
        sum += delta
        print(f"{k}: {distance_1[k] / 1000:.2f} м - меньше на {delta / 1000:.2f} м ({(1 - (distance_1[k] / v)) * 100:.2f}%)")
    print(f"\nОбщая дистанция уменшилась на {sum / 1000:.2f} м")


def count_stroke_distance(default_position, default_keys, mapper, stroke):
    text = stroke["stroke"]
    zone = stroke["zone"]
    count = stroke["count"]
    pairs = []
    total_distance = 0
    for idx, char in enumerate(text):
        if idx + 1 == len(text):
            char_1 = char
            x1 = default_position[mapper[char]['thumb']][0]
            y1 = default_position[mapper[char]['thumb']][1]

            char_2 = default_keys[zone]
            x2 = mapper[char]['x']
            y2 = mapper[char]['y']

            distance = calculateDistance(x1, y1, x2, y2)
            total_distance += distance

            pair = f"{char_1}{char_2}"
            pairs.append({
                "pair": pair,
                "distance": distance
            })
        if idx == 0:
            char_1 = default_keys[zone]
            x1 = default_position[mapper[char]['thumb']][0]
            y1 = default_position[mapper[char]['thumb']][1]

            char_2 = char
            x2 = mapper[char]['x']
            y2 = mapper[char]['y']

            distance = calculateDistance(x1, y1, x2, y2)
            total_distance += distance

            pair = f"{char_1}{char_2}"
            pairs.append({
                "pair": pair,
                "distance": distance
            })
        else:
            char_1 = text[idx - 1]
            x1 = mapper[char_1]['x']
            y1 = mapper[char_1]['y']

            char_2 = char
            x2 = mapper[char_2]['x']
            y2 = mapper[char_2]['y']

            distance = calculateDistance(x1, y1, x2, y2)
            total_distance += distance

            pair = f"{char_1}{char_2}"
            pairs.append({
                "pair": pair,
                "distance": distance
            })
    return {
        "pairs": pairs,
        "count": count,
        "total_distance": total_distance, 
        "zone": zone
    }


def draw_stroke_lines(pairs, COORDS, QWERTY, row_count, max_value, max_line_width):
    ax = get_keyboard(COORDS, QWERTY)
    mapper = get_mapper(COORDS, QWERTY)
    red = Color("green")
    colors = list(red.range_to(Color("red"),100))
    for pair, distance in pairs.items():
        stroke_a, stroke_b = pair[0], pair[1]

        x1 = mapper[stroke_a]['x']
        y1 = mapper[stroke_a]['y']

        x2 = mapper[stroke_b]['x']
        y2 = mapper[stroke_b]['y']

        linewidth = (max_line_width / max_value) * distance
        color_hue = (100 / max_value) * distance
        color_hue = int(round(color_hue))

        r, g, b = colors[color_hue - 1].rgb

        ax.plot([x1,x2],[y1,y2], linewidth=linewidth, color=(r, g, b, 1))


def process_strokes(strokes, coords, qwerty):
    distances = {
        'ЛМ': 0, 
        'ЛБ': 0,
        'ЛС': 0,
        'ЛУ': 0,
        'ПУ': 0,
        'ПС': 0,
        'ПБ': 0,
        'ПМ': 0,
    }
    default_keys = {
        'ЛМ': qwerty[1][0],
        'ЛБ': qwerty[1][1],
        'ЛС': qwerty[1][2],
        'ЛУ': qwerty[1][3],
        'ПУ': qwerty[1][6],
        'ПС': qwerty[1][7],
        'ПБ': qwerty[1][8],
        'ПМ': qwerty[1][9],
    }
    default_position = {
        'ЛМ': coords[1][0], 
        'ЛБ': coords[1][1],
        'ЛС': coords[1][2],
        'ЛУ': coords[1][3],
        'ПУ': coords[1][6],
        'ПС': coords[1][7],
        'ПБ': coords[1][8],
        'ПМ': coords[1][9],
    }
    start_time = time.time()
    mapper = get_mapper(coords, qwerty)
    pairs = {}
    num_workers = cpu_count()
    p = Pool(num_workers)
    manager = Manager()
    func = partial(count_stroke_distance, default_position, default_keys, mapper)
    results = p.map_async(func, strokes).get()
    p.close()
    p.join()
    for stroke_distance in results:
        # stroke_distance = count_stroke_distance(COORDS, QWERTY, THUMBS, default_position, default_keys, stroke)
        distances[stroke_distance["zone"]] += stroke_distance["total_distance"] * stroke_distance["count"]
        for pair in stroke_distance["pairs"]:
            if pair["pair"] in pairs: 
                pairs[pair["pair"]] += pair["distance"] * stroke_distance["count"]
            elif f'{pair["pair"][1]}{pair["pair"][0]}' in pairs:
                pairs[f'{pair["pair"][1]}{pair["pair"][0]}'] += pair["distance"] * stroke_distance["count"]
            else:
                pairs[pair["pair"]] = pair["distance"] * stroke_distance["count"]
    print("--- %s seconds ---" % (time.time() - start_time))
    return {
        "pairs": pairs, 
        "distances": distances
    }