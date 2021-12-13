# -*- coding: utf-8 -*-
"""
💯 Created on Wed Dec  8 02:38:31 2021
🛡️ Give me your power
🧬 @author: Turtle 💕
🌐 Facebook: https://www.facebook.com/bk.turtle.1
"""
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import NeoUnet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs  
)


# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10
# NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 995 originally
IMAGE_WIDTH = 224 # 1280 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train/train/"
TRAIN_MASK_DIR = "dataset/train_gt/train_gt/"
VAL_IMG_DIR = "dataset/val/val/"
VAL_MASK_DIR = "dataset/val_gt/val_gt/"


def tversky(preds, targets, alpha=0.3, smooth=1.0):
    y_true_pos = torch.flatten(preds)
    y_pred_pos = torch.flatten(targets)
    true_pos = torch.sum(y_true_pos*y_pred_pos)
    false_neg = torch.sum(y_true_pos*(1-y_pred_pos))
    false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
    tversky = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    return tversky

def multi_class_loss(preds, targets, gamma=4/3):
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(preds, targets)
    tversky_loss = tversky(preds, targets)
    focal_tversky_loss = torch.pow((1-tversky_loss), gamma)
    return 0.5*(bce_loss+focal_tversky_loss)

def segmentation_loss(preds, targets):
    _preds = (preds > 0.5).float()
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(_preds, targets)
    tversky_loss = tversky(_preds, targets)
    return 0.5*(bce_loss+tversky_loss)

def loss_fn(preds, targets):
    w_c = 0.75
    w_s = 0.25
    Lc = multi_class_loss(preds, targets)
    Ls = segmentation_loss(preds, targets)
    return w_c*Lc + w_s*Ls

def iou_dice_score(masks, preds):
    _preds = (preds > 0.5).float()
    _masks = masks.view(-1)
    _preds = _preds.view(-1)
    intersection = (_masks * _preds).sum()
    union = (_masks + _preds).sum()
    iou = float(intersection + 1e-4)/(float(union) + 1e-4)
    dice = 2.0*float(intersection + 1e-4)/(float(union) + 1e-4)
    return iou, dice
    
def train_fn(loader, model, optimizer, scaler, epoch):
    loop = tqdm(loader)
    
    train_loss = 0
    iou_score = 0
    dice_score = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            iou, dice = iou_dice_score(targets.cpu().data, predictions.cpu().data)
            train_loss += loss.item()
            iou_score += iou
            dice_score += dice
        
        print()
        print('########################### Epoch:', epoch, ', --  batch:',  batch_idx, '/', len(loader))
        print(f'Average train loss: {train_loss/(batch_idx+1):.4f}')
        print(f'MeanIoU: {iou_score/(batch_idx+1):.4f}')
        print(f'MeanDice: {dice_score/(batch_idx+1):.4f}')

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    print()
    print('#################### EVALUATE ##########################')
    print(f'Average train loss after {epoch+1}: {train_loss/len(loader):.4f}')
    print(f'MeanIoU after {epoch+1}: {iou_score/len(loader):.4f}')
    print(f'MeanDice after {epoch+1}: {dice_score/len(loader):.4f}')
    

def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # ToTensor doesn't divide by 255 like PyTorch,
            # it's done inside Normalize function
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],    
    
    )
    
    val_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # ToTensor doesn't divide by 255 like PyTorch,
            # it's done inside Normalize function
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],    
    
    )
    
    model = NeoUnet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
#        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, scaler, epoch)
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        


if __name__ == '__main__':
    main()





