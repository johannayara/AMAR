
#WANDB_MODE=offline python3 scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 1 --env empty_room > ./output/res_AMAR_WO_RVQ_test.txt 2>&1
import numpy as np

y_pred = [[0,0,1],[0,2,1],[0,3,2]]
y_true = [[0,0,1],[0,3,1],[1,3,2]]
print(np.sum(y_pred))
print(np.sum(y_pred, axis=0))
print(np.sum(y_pred, axis=1))