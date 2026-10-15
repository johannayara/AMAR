
#WANDB_MODE=offline python3 scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 1 --env empty_room > ./output/res_AMAR_WO_RVQ_test.txt 2>&1
import numpy as np
import torch 
import scipy.io as sio

mat_contents = sio.loadmat("/cluster/raid/home/jajo/AMAR/dataset/wifi_csi/mat/act_1_1.mat")
print(sorted(mat_contents.keys()))
print(mat_contents["trace"])