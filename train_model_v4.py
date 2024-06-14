import argparse
import logging
import os, wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNetSmall, UNetNano, UNet
from utils.data_loading_model_v4 import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        dataset_dir:str,
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        imgsz:tuple = (640, 640),
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    dir_img = Path(os.path.join(dataset_dir, "images")) 
    # dir_mask = Path('/home/hungdv/tcgroup/dataset/tabelSeg/Scitsr_v6/test/labels')
    dir_mask = Path(os.path.join(dataset_dir, "labels"))
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, imgsz)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, imgsz)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {imgsz}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    wandb.init(project="Unet_small",
               name="ex1",
               config={
                   "learning_rate":learning_rate,
                   "epochs" : epochs,
                   "batch_size" : batch_size,
                   "imgsz":imgsz
               }
               )
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks.float())
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            true_masks,
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
            print("Epoch loss:", epoch_loss*batch_size/len(train_set))
            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))
            print('Validation Dice score: {}'.format(val_score))


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            # torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint.pth'))
            logging.info(f'Checkpoint {epoch} saved!')
        wandb.log({"Loss train": epoch_loss*batch_size/len(train_set),
                   "Validation Dice score": val_score
                   })
    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--dataset', '-d', type=str, help='dataset folder contain labels and images folder', required=True)
    parser.add_argument('--imgsz', '-s', type=int, default=320, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model_type', '-t', type=int, default=2, help='0: basis, 1:small, 2: nano')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model_type == 1:
        model = UNetSmall(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_type == 2:
        model = UNetNano(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            dataset_dir=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            imgsz=(args.imgsz, args.imgsz),
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            dataset_dir=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            imgsz=(args.imgsz, args.imgsz),
            val_percent=args.val / 100,
            amp=args.amp
        )
