import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pandas as pd
import collections
import numpy as np
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
from colour import Color
import copy
import math
import re
import time

from consts import QWERTY, THUMBS, COORDS


def cleanhtml(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    spans = soup.find_all('span')
    lowercase = ''.join([i.text.replace('Пользователь 2: ', '').replace('Пользователь 1: ', '') for i in spans]).lower()
    return re.sub('[^а-я]+', '', lowercase)


def get_dataset_string(filename: str) -> str:
    if Path('sample.pkl').is_file():
        with open('sample.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        dialogues = pd.read_csv(filename, sep='\t')[['dialogue']]
        dialogues['dialogue'] = dialogues['dialogue'].apply(cleanhtml)
        return dialogues['dialogue'].str.cat(sep='')


def get_pairs(sample: str) -> collections.Counter:
    c = collections.Counter()
    for i in range(len(sample)-1):
        pair = sample[i:i+2]
        c[pair] += 1
    return c


def get_viable_thumb_pairs(pairs: collections.Counter, keyboard: list, ):
    pass


if __name__ == "__main__":
    sample_string = get_dataset_string("datasets/dialogues.tsv")
    pairs = get_pairs(sample_string)
    print("TEST")
