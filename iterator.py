# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from bs4 import BeautifulSoup
from functools import partial
import copy
import math
import json
import pickle
import re

from multiprocessing import Pool, Manager, cpu_count
from utils import cleanhtml, calculateDistance, finger_heatmap, shift_col, shift_row, get_mapper, draw_keyboard, count_presses, press_heatmap, zone_distances, distance_deltas, generate_strokes, count_stroke_distance, process_strokes, draw_stroke_lines

from consts import QWERTY, THUMBS, COORDS, default_position


# %%
# Enable inline plots
# get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.width", 70)

# Set plots formats to save high resolution PNG
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")


# %%
# dialogues = pd.read_csv("datasets/dialogues.tsv", sep='\t')[['dialogue']]
# dialogues['dialogue'] = dialogues['dialogue'].apply(cleanhtml)
# sample = dialogues['dialogue'].str.cat(sep='')
with open('sample.pkl', 'rb') as f:
    sample = pickle.load(f)
# strokes = [{
#     "stroke": k, 
#     "count": v["coun# t"], 
#     "zone": v["zone"]} for k, v in generate_strokes(sample, THUMBS, QWERTY).items()]
with open('strokes.pkl', 'rb') as f:
    strokes = pickle.load(f)


# %%
def get_variant_distance(sample, QWERTY):
    strokes = [
        {
            "stroke": k, 
            "count": v["count"], 
            "zone": v["zone"]
        } for k, v in generate_strokes(sample, QWERTY).items()
    ]
    processed_strokes = process_strokes(strokes, COORDS, QWERTY)
    distances_new, pairs = processed_strokes["distances"], processed_strokes["pairs"]
    pairs_df = pd.DataFrame([
        {
            "pair": k, 
            "distance": v
        } for k, v in pairs.items()
    ]).sort_values(by='distance', ascending=False)
    mean = pairs_df["distance"].mean()
    median = pairs_df["distance"].median()
    max_value = pairs_df["distance"].max()
    sum_value = pairs_df["distance"].sum()
    row_count = pairs_df.shape[0]
    print(f'Mean: {mean}, Median: {median}, Max: {max_value}, Count: {row_count}')
    return mean, median, max_value, sum_value, row_count
    # draw_stroke_lines(pairs, COORDS, QWERTY, row_count, max_value, 15)


# %%
QWERTY = [
    ['й','ц','у','к','е','н','г','ш','щ','з','х','ъ'],
    ['ф','ы','в','а','п','р','о','л','д','ж','э',''],
    ['я','ч','с','м','и','т','ь','б','ю','','',''],
]
QWERTY_VARIANTS = []
for row_idx_1, ROW in enumerate(QWERTY):
        for key_idx_1, key in enumerate(ROW):
            first_key = QWERTY[row_idx_1][key_idx_1]
            for row_idx_2, ROW in enumerate(QWERTY):
                for key_idx_2, key in enumerate(ROW):
                    # print(f"{first_key} {second_key}")
                    second_key = QWERTY[row_idx_2][key_idx_2]
                    QWERTY[row_idx_1][key_idx_1] = second_key
                    QWERTY[row_idx_2][key_idx_2] = first_key
                    # print(f"{QWERTY[0]}\n{QWERTY[1]}\n{QWERTY[2]}")
                    QWERTY_VARIANTS.append(QWERTY)
                    QWERTY[row_idx_1][key_idx_1] = first_key
                    QWERTY[row_idx_2][key_idx_2] = second_key


# %%
num_workers = cpu_count()
p = Pool(num_workers)
manager = Manager()
func = partial(get_variant_distance, sample)
results = p.map_async(func, QWERTY_VARIANTS).get()
p.close()
p.join()
for distance_info in results:
    mean, median, max_value, sum_value, row_count = distance_info
    print(f"{mean} / {median} / {max_value} / {sum_value} / {row_count}")


# %%


