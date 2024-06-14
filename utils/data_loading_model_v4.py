import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset



def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, imgsz:tuple = (640, 640), mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_dir_str = mask_dir
        self.imgsz = imgsz
        # self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        

        self.mask_values = [[0, 0, 0], [255, 255, 255]]
        self.mask_suffix = "_mask_"
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, imgsz, is_mask):
        w, h = imgsz
        newW, newH = int(w), int(h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '*.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        mask_file.sort()
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
                
        masks = []
        for mask_f in mask_file:
            mask = load_image(mask_f)
            mask = self.preprocess(self.mask_values, mask, self.imgsz, is_mask=True)
            mask = torch.as_tensor(mask.copy()).long().contiguous().unsqueeze(0)
            masks.append(mask)
        
        masks = torch.cat(masks, dim=0)
        img = load_image(img_file[0])
        img = self.preprocess(self.mask_values, img, self.imgsz, is_mask=False)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': masks
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, imgsz = (640, 640)):
        super().__init__(images_dir, mask_dir, imgsz, mask_suffix='_mask')
