import torch
import sys,os, cv2
sys.path.append(os.path.dirname(__file__))
from unet import UNet, UNetNano, UNetSmall

MODEL_PATH = r"/home/hungdv/tcgroup/extractTable/checkpoints/checkpoint_n43_5.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    net = UNetNano(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    MODE = 0
except:
    try:
        
        net = UNetSmall(n_channels=3, n_classes=2, bilinear=False)
        net.to(device=device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        MODE = 1
    except:
        net = UNet(n_channels=3, n_classes=2, bilinear=False)
        net.to(device=device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        MODE = 2


torch_input = torch.randn(1, 3, 640, 640)
print(os.path.basename(MODEL_PATH).split(".")[0] + ".onnx")

onnx_program = torch.onnx.export(net, torch_input, os.path.basename(MODEL_PATH).split(".")[0] + ".onnx")
