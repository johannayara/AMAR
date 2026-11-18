
import numpy as np
import torch 
import scipy.io as sio
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.unified_rooms import (load_wimans_room_targets, load_wimans_room)

df = pd.read_csv('dataset/synth_wimans/fewshot-anywhere-wimans/annotation.csv')
print(df.values)
print(df.columns)
print(np.max(df["user_1_x"]))
print(np.max(df["user_1_y"]))
print(np.min(df["user_1_x"]))
print(np.min(df["user_1_y"]))
print(df["user_6_action"].dropna())
a = float('nan')
print(np.isnan(a))
load_wimans_room("anywhere")

z,y = (1,2)
print(z)

# locations = ["a", "b", "c", "d", "e"]
# loc_cols = [f"user_{k}_location" for k in range(1, 6)]
# for j in range(1, 6):
#     df_count_j = df[df["number_of_users"] == j]
#     print(f"Count {j}: {len(df_count_j)}")
#     for i in locations:
#         mask = df_count_j[loc_cols].eq(i).any(axis=1)
#         df_location = df_count_j[mask]
#         print(f"Location {i}: {len(df_location)}")       