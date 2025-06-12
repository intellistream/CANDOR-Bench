import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.font_manager import FontProperties
import matplotlib
from matplotlib.lines import Line2D


OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

# you may want to change the color map for different figures
LABEL_WEIGHT = 'bold'
LINE_WIDTH = 3.0
MARKER_SIZE = 12.5
MARKER_FREQUENCY = 1000

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

markers = ['o', 's', 'D', 'v', '^', 'P', '*', 'X', 'h', '>', '<', 'p', 'H', 'd', '1', '2', '3', '4']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'brown', 'purple', 'gray', 'lime', 'teal', 'indigo',
          'violet', 'gold', 'coral']
algorithms = [
    "nswlib_HNSW"
]

categories = {
    # 'Tree': ['candy_sptag'],
    # 'Hash': ['candy_lshapg', 'faiss_lsh'],
    # 'Clustering': ['faiss_pq', 'faiss_onlinepq', 'faiss_IVFPQ', 'puck', 'faiss_fast_scan'],
    'Graph-based': ['nswlib_HNSW']
}
