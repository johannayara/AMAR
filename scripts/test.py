import argparse
import random # Added
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import *
from src.data.load_data import load_data_x, load_data_y, encode_data_y
from src.utils import *
from configs.preset import preset

""" fig = plt.figure(figsize=(6,6))
ax = plt.subplot(aspect=1)
plt.set_cmap("cool")
ax.plot(range(10))
plt.show() """

#WANDB_MODE=offline python3 scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 1 --env empty_room > ./output/res_AMAR_WO_RVQ_test.txt 2>&1
data_x_train = []
data_x_test = []
data_y_train = []
data_y_test = []
all_envs_results = run_cross_domain(data_x_train, data_y_train, test_sets_by_env = {}, var_repeat = 1, var_task = "location")