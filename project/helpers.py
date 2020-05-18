import collections
import hashlib
import math
import pickle
import re
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from colour import Color
from constants import QWERTY, THUMBS, COORDS


def shortname(layout):
    layout_string = ''.join([''.join([str(i) for i in col]) for row in layout for col in row])
    return str(int(hashlib.sha256(layout_string.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def get_cache_key(items):
    return '_'.join([shortname(item) for item in items])


def with_cache(variable_name):
    def dec(function):
        def wrapper(*args, **kwargs):
            cache_key = kwargs.get('cache_key', "0")
            if cache_key == 'no-cache':
                return function(*args, **kwargs)
            cached_variable, cache_success = read_cache(f'{variable_name}_{cache_key}')
            if cache_success:
                return cached_variable
            else:
                result = function(*args, **kwargs)
                write_cache(f'{variable_name}_{cache_key}', result)
            return result

        return wrapper

    return dec


def save_results(results):
    now_str = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
    Path('results/').mkdir(parents=True, exist_ok=True)
    with open(f'results/results_{now_str}.pkl', 'wb') as f:
        pickle.dump(results, f)


def show_results(results_file):
    file = Path(f'results/{results_file}.pkl')
    if not file.is_file():
        print('Такого файла нет.')
        return
    with open(f'results/{results_file}.pkl', 'rb') as f:
        results = pickle.load(f)
    for sorted_variants in results:
        best_variant = sorted_variants[0]
        get_distance_map(
            thumbs_distance=best_variant["thumbs_distance"],
            max_distance=4369291.99691117,
            coords=COORDS,
            layout=best_variant["layout"],
            title=f"ЛУЧШИЙ ВАРИАНТ",
            subtitle=f"MAX: {best_variant['max']:.2f}\n"
                     f"SUM: {best_variant['sum']:.2f}\n"
                     f"AVG: {best_variant['avg']:.2f}\n"
                     f"DELTA: {best_variant['delta']:.2f}"
        )


def write_cache(variable_name, data):
    Path('cache/').mkdir(parents=True, exist_ok=True)
    with open(f'cache/{variable_name}.pkl', 'wb') as f:
        pickle.dump(data, f)


def read_cache(variable_name):
    file = Path(f'cache/{variable_name}.pkl')
    if not file.is_file():
        return None, False
    with open(f'cache/{variable_name}.pkl', 'rb') as f:
        return pickle.load(f), True


def cleanhtml(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    spans = soup.find_all('span')
    lowercase = ''.join(
        [i.text.replace('Пользователь 2: ', '').replace('Пользователь 1: ', '') for i in spans]
    ).lower()
    return re.sub('[^а-я]+', '', lowercase)


def two_point_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def visualize_results(results):
    max_x = []
    max_y = []
    sum_x = []
    sum_y = []
    avg_x = []
    avg_y = []
    delta_x = []
    delta_y = []
    score_x = []
    score_y = []
    fig, axs = plt.subplots(2, 2)
    if len(results) > 1000:
        skip_interval = 100
    else:
        skip_interval = 10
    for idx, sorted_variants in enumerate(results):
        if idx % skip_interval == 0 or idx == len(results) - 1:
            best_variant = sorted_variants[0]
            max_x.append(idx)
            sum_x.append(idx)
            avg_x.append(idx)
            delta_x.append(idx)
            score_x.append(idx)
            max_y.append(best_variant['max'])
            sum_y.append(best_variant['sum'])
            avg_y.append(best_variant['avg'])
            delta_y.append(best_variant['delta'])
            score_y.append(best_variant['avg'] + best_variant['delta'])
    axs[0, 0].plot(max_x, max_y)
    axs[0, 0].set_title("MAX")
    # axs[1, 0].plot(sum_x, sum_y)
    # axs[1, 0].set_title("SUM")
    axs[1, 0].plot(score_x, score_y)
    axs[1, 0].set_title("SCORE")
    axs[0, 1].plot(avg_x, avg_y)
    axs[0, 1].set_title("AVG")
    axs[1, 1].plot(delta_x, delta_y)
    axs[1, 1].set_title("DELTA")
    # axs[2, 1].plot(score_x, score_y)
    # axs[2, 1].set_title("SCORE")
    fig.tight_layout()


def get_keyboard(coords, layout, draw=True, title="Координаты клавиш", subtitle=""):
    x = [i[0] for i in [item for sublist in coords for item in sublist]]
    y = [i[1] for i in [item for sublist in coords for item in sublist]]
    n = [item for sublist in layout for item in sublist]
    fig, ax = plt.subplots()
    if len(subtitle) > 0:
        fig.subplots_adjust(top=0.98)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        ax.set_title(subtitle, fontsize=10)
    else:
        ax.set_title(title, fontsize=10)
    ax.scatter(x, y, marker=",", s=620, color=(0.5, 0.5, 0.5))
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
    if not draw:
        return ax


@with_cache('layout_delta_distance')
def layout_delta_distance(layout_a, layout_b, thumbs, coords, **kwargs):
    delta_distance = 0
    layout_a_mapping = get_keys_mapping(
        layout=layout_a,
        thumbs=thumbs,
        coords=coords,
        cache_key='no-cache'  # get_cache_key([layout_a, thumbs, coords])
    )
    layout_b_mapping = get_keys_mapping(
        layout=layout_b,
        thumbs=thumbs,
        coords=coords,
        cache_key='no-cache'  # get_cache_key([layout_b, thumbs, coords])
    )
    for key, values in layout_a_mapping.items():
        x1 = values["x"]
        y1 = values["y"]
        x2 = layout_b_mapping[key]["x"]
        y2 = layout_b_mapping[key]["y"]
        delta_distance += pow(two_point_distance(x1, y1, x2, y2), 2)
    return delta_distance


def get_distance_map(thumbs_distance, coords, layout, max_distance=None, title='Пройденная дистанция', subtitle=""):
    ax = get_keyboard(
        coords=coords,
        layout=layout,
        draw=False,
        title=title,
        subtitle=subtitle
    )
    max_line_width = 18
    sorted_pairs = sorted(
        [
            [
                pair_values["total_distance"],
                pair_values["coords"]["x1"],
                pair_values["coords"]["y1"],
                pair_values["coords"]["x2"],
                pair_values["coords"]["y2"]
            ]
            for thumb, thumb_values in thumbs_distance.items()
            for pair, pair_values in thumb_values["pairs"].items()
        ], key=lambda x: x[0], reverse=True
    )
    if max_distance is None:
        max_distance = sorted_pairs[0][0]
    colors = list(Color("green").range_to(Color("red"), 500))
    for pair in sorted_pairs:
        pair_distance = pair[0]
        x1 = pair[1]
        y1 = pair[2]
        x2 = pair[3]
        y2 = pair[4]
        coeff = pair_distance / max_distance
        if coeff > 1:
            coeff = 1
        linewidth = coeff * max_line_width
        color_hue = int(round(coeff * 500))
        r, g, b = colors[color_hue - 1].rgb
        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, color=(r, g, b, 1))


@with_cache('keys_mapping')
def get_keys_mapping(layout: list, thumbs: list, coords: list, **kwargs):
    mapper = {}
    for row_idx, row in enumerate(layout):
        for col_idx, col in enumerate(row):
            if len(col) > 1:
                for key in col:
                    mapper[key] = {
                        "x": coords[row_idx][col_idx][0],
                        "y": coords[row_idx][col_idx][1],
                        "thumb": thumbs[row_idx][col_idx]
                    }
            else:
                mapper[col] = {
                    "x": coords[row_idx][col_idx][0],
                    "y": coords[row_idx][col_idx][1],
                    "thumb": thumbs[row_idx][col_idx]
                }
    return mapper


@with_cache('dataset_sample')
def preprocess_dataset(**kwargs):
    dialogues = pd.read_csv("../datasets/dialogues.tsv", sep='\t')[['dialogue']]
    dialogues['dialogue'] = dialogues['dialogue'].apply(cleanhtml)
    data = dialogues['dialogue'].str.cat(sep='')
    return data


@with_cache('press_count')
def get_key_press_count(dataset_sample: str, **kwargs) -> collections.Counter:
    c = collections.Counter()
    for i in dataset_sample:
        c[i] += 1
    return c


@with_cache('pairs_count')
def get_pairs_count(dataset_sample: str, **kwargs) -> collections.Counter:
    c = collections.Counter()
    for i in range(len(dataset_sample) - 1):
        pair = dataset_sample[i:i + 2]
        mirror_pair = pair[::-1]
        if pair in c:
            c[pair] += 1
        elif mirror_pair in c:
            c[mirror_pair] += 1
        else:
            c[pair] += 1
    return c


@with_cache('thumbs_distance')
def get_thumbs_distance(pairs: collections.Counter, layout: list, thumbs: list, coords: list, **kwargs):
    key_mapping = get_keys_mapping(
        layout=layout,
        thumbs=thumbs,
        coords=coords,
        cache_key='no-cache'  # kwargs['cache_key']
    )
    result = {}
    for pair in list(pairs):
        if key_mapping[pair[0]]["thumb"] == key_mapping[pair[1]]["thumb"]:
            thumb = key_mapping[pair[0]]["thumb"]
            if thumb not in result:
                result[thumb] = {
                    "pairs": {},
                    "total_distance": 0
                }
            thumb = key_mapping[pair[0]]["thumb"]
            x1 = key_mapping[pair[0]]["x"]
            y1 = key_mapping[pair[0]]["y"]
            x2 = key_mapping[pair[1]]["x"]
            y2 = key_mapping[pair[1]]["y"]
            count = pairs[pair]
            distance = two_point_distance(x1, y1, x2, y2)
            total_distance = distance * count
            result[thumb]["pairs"][pair] = {
                "coords": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "thumb": thumb,
                "distance": distance,
                "count": count,
                "total_distance": total_distance
            }
            result[thumb]["total_distance"] += total_distance
    return result


def get_column_layouts(starting_layout, col_num=0):
    layouts = []
    new_layout = deepcopy(starting_layout)
    for c in range(3):
        items = deque([col for row in new_layout for idx, col in enumerate(row) if idx == col_num])
        items.rotate(1)
        for row_num, item in enumerate(items):
            new_layout[row_num][col_num] = item
        layouts.append(new_layout)
        new_layout = deepcopy(new_layout)
        # print(new_layout)
    return layouts


def get_left_row_layouts(starting_layout, row_num=0):
    layouts = []
    new_layout = deepcopy(starting_layout)
    for c in range(len([col for col in new_layout[row_num] if col != ''][0:5])):
        items = deque([col for col in new_layout[row_num] if col != ''][0:5])
        items.rotate(1)
        new_layout[row_num] = list(items) + [col for col in new_layout[row_num] if col != ''][5:12]
        while len(new_layout[row_num]) < 12:
            new_layout[row_num].append('')
        layouts.append(new_layout)
        new_layout = deepcopy(new_layout)
        # print(new_layout)
    return layouts


def get_right_row_layouts(starting_layout, row_num=0):
    layouts = []
    new_layout = deepcopy(starting_layout)
    for c in range(len([col for col in new_layout[row_num] if col != ''][5:13])):
        items = deque([col for col in new_layout[row_num] if col != ''][5:13])
        items.rotate(1)
        new_layout[row_num] = [col for col in new_layout[row_num] if col != ''][0:5] + list(items)
        while len(new_layout[row_num]) < 12:
            new_layout[row_num].append('')
        layouts.append(new_layout)
        new_layout = deepcopy(new_layout)
        # print(new_layout)
    return layouts


def test_layouts(layouts, draw_graphs=False):
    dataset = preprocess_dataset()
    pairs = get_pairs_count(dataset)
    iteration, qwerty_max_distance = 0, 4369291.99691117
    results = []
    for layout in layouts:
        thumbs_distance = get_thumbs_distance(
            pairs=pairs,
            layout=layout,
            thumbs=THUMBS,
            coords=COORDS,
            cache_key='no-cache'  # get_cache_key([layout, THUMBS, COORDS])
        )
        distances = [
            {
                "pair_distance": pair_values["total_distance"],
                "pair": pair
            }
            for thumb, thumb_values in thumbs_distance.items()
            for pair, pair_values in thumb_values["pairs"].items()
        ]
        max_distance = max([i["pair_distance"] for i in distances])
        sum_distance = sum([i["pair_distance"] for i in distances])
        avg_distance = sum_distance / len(distances)
        delta_distance = layout_delta_distance(
            layout_a=QWERTY,
            layout_b=layout,
            thumbs=THUMBS,
            coords=COORDS,
            cache_key='no-cache'  # get_cache_key([QWERTY, layout, THUMBS, COORDS])
        )
        results.append(
            {
                "layout": layout,
                "max": max_distance,
                "sum": sum_distance,
                "avg": avg_distance,
                "delta": delta_distance,
                "thumbs_distance": thumbs_distance
            }
        )
        # print(f"MAX: {max_distance} / SUM: {sum_distance} / AVG: {avg_distance}")
        if draw_graphs:
            get_distance_map(
                thumbs_distance=thumbs_distance,
                max_distance=qwerty_max_distance,
                coords=COORDS,
                layout=layout,
                subtitle=f"MAX: {max_distance:.2f}\n"
                         f"SUM: {sum_distance:.2f}\n"
                         f"AVG: {avg_distance:.2f}\n"
                         f"DELTA: {delta_distance:.2f}"
            )
    sorted_variats = sorted([
        i for i in results
    ], key=lambda x: (x['delta'] if x['delta'] > 0 else 999999999) + x['avg'])
    if draw_graphs:
        get_distance_map(
            thumbs_distance=sorted_variats[0]["thumbs_distance"],
            max_distance=qwerty_max_distance,
            coords=COORDS,
            layout=sorted_variats[0]["layout"],
            title=f"ЛУЧШИЙ ВАРИАНТ",
            subtitle=f"MAX: {sorted_variats[0]['max']:.2f}\n"
                     f"SUM: {sorted_variats[0]['sum']:.2f}\n"
                     f"AVG: {sorted_variats[0]['avg']:.2f}\n"
                     f"DELTA: {sorted_variats[0]['delta']:.2f}"
        )
    return sorted_variats

# get_column_layouts(QWERTY, 0)
# get_left_row_layouts(QWERTY, 0)
# get_right_row_layouts(QWERTY, 0)

# best_0_0 = test_layouts(get_left_row_layouts(QWERTY, 0))["layout"]
# best_1_0 = test_layouts(get_left_row_layouts(best_0_0, 1))["layout"]
# best_2_0 = test_layouts(get_left_row_layouts(best_1_0, 2))["layout"]

# # ЙЦУКЕН
#
# dataset = preprocess_dataset()
# pairs = get_pairs_count(dataset)
# qwerty_thumbs_distance = get_thumbs_distance(
#     pairs=pairs,
#     layout=QWERTY,
#     thumbs=THUMBS,
#     coords=COORDS,
#     cache_key=get_cache_key([QWERTY, THUMBS, COORDS])
# )
# qwerty_distances = [
#     pair_values["total_distance"]
#     for thumb, thumb_values in qwerty_thumbs_distance.items()
#     for pair, pair_values in thumb_values["pairs"].items()
# ]
# qwerty_max_distance = max(qwerty_distances)
# print(f"Максимальная пройденная дистанция: {qwerty_max_distance} мм")
# qwerty_sum_distance = sum(qwerty_distances)
# print(f"Общая пройденная дистанция: {qwerty_sum_distance} мм")
# get_distance_map(
#     thumbs_distance=qwerty_thumbs_distance,
#     max_distance=qwerty_max_distance,
#     coords=COORDS,
#     layout=DIKTOR
# )
#
# # DIKTOR
#
# diktor_thumbs_distance = get_thumbs_distance(
#     pairs=pairs,
#     layout=DIKTOR,
#     thumbs=THUMBS,
#     coords=COORDS,
#     cache_key=get_cache_key([DIKTOR, THUMBS, COORDS])
# )
# diktor_distances = [
#     pair_values["total_distance"]
#     for thumb, thumb_values in diktor_thumbs_distance.items()
#     for pair, pair_values in thumb_values["pairs"].items()
# ]
# diktor_max_distance = max(diktor_distances)
# print(f"Максимальная пройденная дистанция: {diktor_max_distance} мм")
# diktor_sum_distance = sum(diktor_distances)
# print(f"Общая пройденная дистанция: {diktor_sum_distance} мм")
# get_distance_map(
#     thumbs_distance=diktor_thumbs_distance,
#     max_distance=qwerty_max_distance,
#     coords=COORDS,
#     layout=DIKTOR
# )
#
# delta = layout_delta_distance(
#     layout_a=DIKTOR,
#     layout_b=QWERTY,
#     thumbs=THUMBS,
#     coords=COORDS,
#     cache_key=get_cache_key([DIKTOR, QWERTY, THUMBS, COORDS])
# )
# print(f"Layout delta: {delta}")

# iterator(QWERTY)
