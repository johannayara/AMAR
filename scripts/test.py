
#WANDB_MODE=offline python3 scripts/run_main.py --model AMAR_WO_RVQ --task location --repeat 1 --env empty_room > ./output/res_AMAR_WO_RVQ_test.txt 2>&1
import numpy as np
import torch 

y_pred = [[0,0,1],[0,2,1],[0,3,2]]
y_true = [[0,0,1],[0,3,1],[1,3,2]]
print(np.sum(y_pred))
print(np.sum(y_pred, axis=0))
print(np.sum(y_pred, axis=1))

m = torch.nn.AvgPool1d(3, stride=2)
t = torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]])
print(t)
print(m(t))
t = torch.tensor([[[4, 2, 30, 4, 5, 4, 7]]])
print(t)
print(m(t))
print(t.shape)
t = torch.tensor([[4, 2, 30, 4, 5, 4, 9]])
print(t)
print(m(t))
print(t.shape)
var_position = torch.arange(0.0, 6).unsqueeze(1).repeat(1, 3)
print(var_position)
var_sigma = torch.tensor([50.0] * 3).unsqueeze(0)
print(var_sigma)
var_embedding = torch.zeros([3, 4], dtype=torch.float)
var_embedding = torch.nn.init.xavier_uniform_(var_embedding)
print(var_embedding)