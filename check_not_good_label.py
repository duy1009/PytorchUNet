from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torch
from PIL import Image
import cv2, glob, shutil
import numpy as np
from os.path import splitext, basename, join
from os import makedirs
image_dir = "/home/hungdv/tcgroup/dataset/tabelSeg/Scitsr_aug_full_3/test/images"
label_dir = "/home/hungdv/tcgroup/dataset/tabelSeg/Scitsr_aug_full_3/test/labels"

paths = glob.glob(image_dir+"/**")

file = paths[0]
mask_values = [[0, 0, 0], [255, 255, 255]]


import torch.nn.functional as F
from utils.data_loading_model3 import BasicDataset as BasicDataset_3
from utils.data_loading import BasicDataset

from unet import UNet, UNetNano, UNetSmall

MODEL_PATH = "/home/hungdv/Downloads/checkpoint_s57_5.pth"

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
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    mask = mask[0].long().squeeze()
    return mask

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def loadlabel(labelpath):
    mask = load_image(labelpath)
    mask = BasicDataset_3.preprocess(mask_values, mask, (640, 640), is_mask=True)

    return torch.as_tensor(mask.copy()).long().contiguous()

try:
    makedirs("./test/false")
    makedirs("./test/true")
except:
    pass

s_score = 0
bad = 0
good = 0
for i, file in enumerate(paths):
    image = Image.open(file).convert("RGB")
    mask_pred = predict_img(image)
    mask_true = loadlabel(join(label_dir, basename(file)[:-4]+".png"))
    score = dice_coeff(mask_pred, mask_true)
    s_score+=score
    if score < 0.8:
        shutil.copyfile(file, "./test/false/"+basename(file))
        bad+=1
    elif score > 0.98:
        shutil.copyfile(file, "./test/true/"+basename(file))
        good+=1
    img = cv2.imread(file)
    # cv2.imshow("a", mask_pred.numpy().astype(np.uint8)*255)
    # cv2.imshow("b", img)
    # cv2.waitKey(1)
    print("\n[Result]")
    print(f"- Dice score avg: {s_score/(i+1)}")
    print(f"- Bad labels: {bad/len(paths)} ({bad}/{i+1})")
    print(f"- Good labels: {good/len(paths)} ({good}/{i+1})")
    
print("\n[Result]")
print(f"- Dice score avg: {s_score/len(paths)}")
print(f"- Bad labels: {bad/len(paths)} ({bad}/{len(paths)})")
print(f"- Good labels: {good/len(paths)} ({good}/{len(paths)})")