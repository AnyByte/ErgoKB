# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Импорт библиотек

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
from bs4 import BeautifulSoup
import copy
import math
import json
import pickle
import re

from utils import cleanhtml, calculateDistance, finger_heatmap, shift_col, shift_row, get_mapper, draw_keyboard, count_distance, count_presses, press_heatmap, zone_distances, distance_deltas, generate_strokes, count_stroke_distance, process_strokes, draw_stroke_lines

from consts import QWERTY, THUMBS, COORDS, default_position


# %%
# Enable inline plots
# get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.width", 70)

# Set plots formats to save high resolution PNG
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")

# %% [markdown]
# ## Распределение пальцев по зонам
# - ЛМ - левый мезинец
# - ЛБ - левый безымянный
# - ЛС - левый средний
# - ЛУ - левый указательный
# - ПУ - правый указательный
# - ПС - правый средний
# - ПБ - правый безымянный
# - ПМ - правый мезинец
# %% [markdown]
# ## Схема исходной клавиатуры

# %%
draw_keyboard(COORDS, QWERTY)

# %% [markdown]
# ## Обработка и подготовка датасета

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
processed_strokes = process_strokes(strokes, COORDS, QWERTY)
# with open('processed_strokes.pkl', 'rb') as f:
#     processed_strokes = pickle.load(f)

# %% [markdown]
# ## Тепловая карта пройденного расстояния

# %%
distances, pairs = processed_strokes["distances"], processed_strokes["pairs"]


# %%
pairs_df = pd.DataFrame([{"pair": k, "distance": v} for k, v in pairs.items()]).sort_values(by='distance', ascending=False)
mean = pairs_df["distance"].mean()
median = pairs_df["distance"].median()
max_value = pairs_df["distance"].max()
row_count = pairs_df.shape[0]
print(f'Mean: {mean}, Median: {median}, Max: {max_value}, Count: {row_count}')


# %%
draw_stroke_lines(pairs, COORDS, QWERTY, row_count, max_value, 15)


# %%
pairs_df[pairs_df["distance"] >= mean]


# %%
finger_distance_heatmap = finger_heatmap(distances)
heat_map = sb.heatmap(finger_distance_heatmap, linewidth=.5, square=True, annot=[['ЛМ','ЛБ','ЛС','ЛУ','ПУ','ПС','ПБ','ПМ']], fmt = '')
heat_map.set_title('Тепловая карта пройденного расстояния', fontsize=10)
heat_map.axis('off')
plt.show()

# %% [markdown]
# ## Тепловая карта частоты нажатий

# %%
press_count = count_presses(sample)
keypresses_heatmap = press_heatmap(press_count, QWERTY)


# %%
heat_map = sb.heatmap(keypresses_heatmap, linewidth=.5, square=True, annot=QWERTY, fmt = '')
heat_map.set_title('Тепловая карта частоты нажатий', fontsize=10)
heat_map.axis('off')
plt.show()

# %% [markdown]
# ## Частота нажаний и расстояние от исходного положения пальцев по каждой зоне
# %% [markdown]
# ### Левый мизинец

# %%
pd.DataFrame(zone_distances('ЛМ', press_count))

# %% [markdown]
# ### Левый безымянный
# 

# %%
pd.DataFrame(zone_distances('ЛБ', press_count))

# %% [markdown]
# ### Левый средний
# 

# %%
pd.DataFrame(zone_distances('ЛС', press_count))

# %% [markdown]
# ### Левый указательный

# %%
pd.DataFrame(zone_distances('ЛУ', press_count))

# %% [markdown]
# ### Правый указательный
# 

# %%
pd.DataFrame(zone_distances('ПУ', press_count))

# %% [markdown]
# ### Правый средний

# %%
pd.DataFrame(zone_distances('ПС', press_count))

# %% [markdown]
# ### Правый безымянный

# %%
pd.DataFrame(zone_distances('ПБ', press_count))

# %% [markdown]
# ### Правый мизинец

# %%
pd.DataFrame(zone_distances('ПМ', press_count))

# %% [markdown]
# ## Переставим кнопки в каждой зоне так, чтобы наиболее частые клавиши находились как можно ближе к исходному положению

# %%
QWERTY_1 = [
    ['й','ц','у','к','м','г','н','ш','щ','ж','ъ',''],
    ['я','ч','с','е','а','т','о','л','д','з','э',''],
    ['ф','ы','в','и','п','ь','р','б','ю','х','',''],
]


# %%
strokes_1 = [{
    "stroke": k, 
    "count": v["count"], 
    "zone": v["zone"]} for k, v in generate_strokes(sample, QWERTY_1).items()]


# %%
draw_keyboard(COORDS, QWERTY_1)


# %%
keypresses_heatmap = press_heatmap(press_count, QWERTY_1)
heat_map = sb.heatmap(keypresses_heatmap, linewidth=.5, square=True, annot=QWERTY_1, fmt = '')
heat_map.set_title('Тепловая карта частоты нажатий', fontsize=10)
heat_map.axis('off')
plt.show()


# %%
processed_strokes_1 = process_strokes(strokes, COORDS, QWERTY_1)
# with open('processed_strokes_1.pkl', 'rb') as f:
#     processed_strokes_1 = pickle.load(f)


# %%
distances_1, pairs_1 = processed_strokes_1["distances"], processed_strokes_1["pairs"]


# %%
distance_deltas(distances, distances_1)


# %%
pairs_df_1 = pd.DataFrame([{"pair": k, "distance": v} for k, v in pairs_1.items()]).sort_values(by='distance', ascending=False)
mean = pairs_df_1["distance"].mean()
median = pairs_df_1["distance"].median()
max_value = pairs_df_1["distance"].max()
row_count = pairs_df_1.shape[0]
print(f'Mean: {mean}, Median: {median}, Max: {max_value}, Count: {row_count}')


# %%
draw_stroke_lines(pairs_1, COORDS, QWERTY_1, row_count, max_value, 15)


# %%
pairs_df_1[pairs_df["distance"] >= mean]


# %%
finger_distance_heatmap_1 = finger_heatmap(distances_1)
heat_map = sb.heatmap(finger_distance_heatmap_1, linewidth=.5, square=True, annot=[['ЛМ','ЛБ','ЛС','ЛУ','ПУ','ПС','ПБ','ПМ']], fmt = '')
heat_map.set_title('Тепловая карта пройденного расстояния', fontsize=10)
heat_map.axis('off')
plt.show()

# %% [markdown]
# ## Переставим Т и Ч местами

# %%
QWERTY_2 = [
    ['й','ц','с','к','м','г','д','ш','щ','ж','ъ',''],
    ['я','т','у','е','а','ч','о','л','н','з','э',''],
    ['ф','ы','в','и','п','ь','р','б','ю','х','',''],
]


# %%
strokes_2 = [{
    "stroke": k, 
    "count": v["count"], 
    "zone": v["zone"]} for k, v in generate_strokes(sample, QWERTY_2).items()]


# %%
draw_keyboard(COORDS, QWERTY_2)


# %%
keypresses_heatmap = press_heatmap(press_count, QWERTY_2)
heat_map = sb.heatmap(keypresses_heatmap, linewidth=.5, square=True, annot=QWERTY_2, fmt = '')
heat_map.set_title('Тепловая карта частоты нажатий', fontsize=10)
heat_map.axis('off')
plt.show()


# %%
processed_strokes_2 = process_strokes(strokes_2, COORDS, QWERTY_2)
# with open('processed_strokes_1.pkl', 'rb') as f:
#     processed_strokes_1 = pickle.load(f)


# %%
distances_2, pairs_2 = processed_strokes_2["distances"], processed_strokes_2["pairs"]


# %%
distance_deltas(distances, distances_2)


# %%
pairs_df_2 = pd.DataFrame([{"pair": k, "distance": v} for k, v in pairs_2.items()]).sort_values(by='distance', ascending=False)
mean = pairs_df_2["distance"].mean()
median = pairs_df_2["distance"].median()
max_value = pairs_df_2["distance"].max()
row_count = pairs_df_2.shape[0]
print(f'Mean: {mean}, Median: {median}, Max: {max_value}, Count: {row_count}')


# %%
draw_stroke_lines(pairs_2, COORDS, QWERTY_2, row_count, max_value, 15)


# %%


