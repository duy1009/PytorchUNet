import torch
import sys,os, cv2
sys.path.append(os.path.dirname(__file__))
import torch.nn.functional as F
from utils.data_loading_model3 import BasicDataset as BasicDataset_3
from utils.data_loading import BasicDataset

from unet import UNet, UNetNano, UNetSmall

MODEL_PATH = os.path.dirname(__file__) + "/checkpoints/checkpoint_s11.pth"

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

print("Total params:", sum(p.numel() for p in net.parameters()))

def predict_img(full_img,
                net=net,
                device=device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    w, h = full_img.size
    if MODE == 1:
        img = torch.from_numpy(BasicDataset_3.preprocess(None, full_img, (640, 640), is_mask=False))
    else:
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    mask = mask[0].long().squeeze().numpy()
    if MODE == 1:
        mask = cv2.resize(mask, (w, h))
    return mask

