import torch
import torch.nn.functional as F
from PytorchUNet.utils.data_loading import BasicDataset
from PytorchUNet.unet import UNet

MODEL_PATH = "/home/hungdv/tcgroup/extractTable/PytorchUNet/checkpoints/checkpoint_48e.pth"
net = UNet(n_channels=3, n_classes=2, bilinear=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
state_dict = torch.load(MODEL_PATH, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

def predict_img(full_img,
                net=net,
                device=device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
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

    return mask[0].long().squeeze().numpy()
