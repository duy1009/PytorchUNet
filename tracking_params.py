import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from unet import UNetNano
from utils import load_model
import numpy as np

model_var = UNetNano(3, 2)


def getNameMinMaxAvg(model_var: UNetNano):
    result = []
    for name, var in model_var.named_parameters():
        var = torch.abs(var)
        _min = torch.min(var).item()
        _max = torch.max(var).item()
        avg = torch.mean(var).item()
        result.append([name, _min, _max, avg]) 
    return np.array(result)

fig, ax1 = plt.subplots()
        

def animate(i):
    var = load_model(model_var, "./var.pth")
    name = getNameMinMaxAvg(var)[:, 0]
    _min = np.array(getNameMinMaxAvg(model_var)[:, 1], dtype=np.float32)
    _max = np.array(getNameMinMaxAvg(model_var)[:, 2], dtype=np.float32)
    avg = np.array(getNameMinMaxAvg(model_var)[:, 3], dtype=np.float32)

    ax1.clear()
    ax1.bar(name, _max, color ='red', 
        width = 0.4)
    ax1.bar(name, avg, color ='green', 
        width = 0.4)
    ax1.bar(name, _min, color ='blue', 
        width = 0.4)

ani = animation.FuncAnimation(fig, animate, interval=2000)

plt.show()