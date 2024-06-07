import numpy as np
import torch
from unet import UNetSmall as UNet
PATH = r"/home/hungdv/tcgroup/extractTable/PytorchUNet/checkpoints/checkpoint_s11.pth"
THRS = 0.1
def load_model(model, path:str):
    state_dict = torch.load(path, map_location="cpu")
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    
    return model

def getNameMinMaxAvg(model_var):
    result = []
    for name, var in model_var.named_parameters():
        _min = torch.min(var).item()
        _max = torch.max(var).item()
        avg = torch.mean(var).item()
        result.append([name, _min, _max, avg]) 
    return np.array(result)

model = load_model(UNet(3, 2), PATH)

name = getNameMinMaxAvg(model)[:, 0]
_min = np.array(getNameMinMaxAvg(model)[:, 1], dtype=np.float32)
_max = np.array(getNameMinMaxAvg(model)[:, 2], dtype=np.float32)
avg = np.array(getNameMinMaxAvg(model)[:, 3], dtype=np.float32)
[print(i[0], i[1]) for i in zip(name, avg)]
print("\n[Weight low]:")
for cnt, i in enumerate(name):
    if abs(avg[cnt]-1) < THRS:
        print(i, abs(avg[cnt]-1))
