# -*- coding: utf-8 -*-
"""
💯 Created on Wed Dec  8 01:12:56 2021
🛡️ Give me your power
🧬 @author: Turtle 💕
🌐 Facebook: https://www.facebook.com/bk.turtle.1
"""
import torch
import torchvision
from dataset import NeoDataset
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, file_name='my_checkpoint.pth.tar'):
    print('###### Saving checkpoint...')
    torch.save(state, file_name)
    
def load_checkpoint(checkpoint, model):
    print("###### Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, batch_size, train_transform, val_transform, pin_memory=True):
    train_ds = NeoDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_memory,
        shuffle=True,
        )
    
    val_ds = NeoDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_memory,
        shuffle=False,
        )
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    iou = 0
    dice = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            # preds = torch.argmax(preds, dim=1)
            y = y.view(-1)
            preds = preds.view(-1)
            
            # Acc
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # IoU, Dice
            # intersection = 0
            # union = 0
            # for sem_class in range(3):
            #     pred_inds = (preds == sem_class)
            #     target_inds = (y == sem_class)
            #     intersection += pred_inds[target_inds].sum()
            #     union += pred_inds.sum() + target_inds.sum()
            # iou += float(intersection + 1e-4)/(float(union) + 1e-4)
            # dice += 2.0*float(intersection + 1e-4)/(float(union) + 1e-4)
            intersection = (preds * y).sum()
            union = (preds + y).sum()
            iou += float(intersection + 1e-4)/(float(union) + 1e-4)
            dice += 2.0*float(intersection + 1e-4)/(float(union) + 1e-4)
        
        
    print(f"###### Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels:.2f}")
    print(f"###### IoU score: {iou/len(loader)}")
    print(f"###### Dice score: {dice/len(loader)}")

    model.train()
    
def save_predictions_as_imgs(
  loader, model, folder="saved_images/", device="cuda"  
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1),
                                     f"{folder}{idx}.png")
        
    model.train()