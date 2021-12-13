# -*- coding: utf-8 -*-
"""
💯 Created on Wed Dec  8 00:05:58 2021
🛡️ Give me your power
🧬 @author: Turtle 💕
🌐 Facebook: https://www.facebook.com/bk.turtle.1
"""
import numpy as np
from PIL import Image
import cv2 as cv
import os
from torch.utils.data import Dataset


IMAGE_DIR = 'dataset/train/train/'
MASK_DIR = 'dataset/train_gt/train_gt/'

def preprocess(mask): 
    image = cv.imread(mask)
    image = cv.resize(image, (224, 160))
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    lower_mask = cv.inRange(image, lower1, upper1)
    upper_mask = cv.inRange(image, lower2, upper2)

    red_mask = lower_mask + upper_mask;
    red_mask[red_mask != 0] = 2
    
    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv.inRange(image, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 1
    
    full_mask = cv.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)
    return full_mask

class NeoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert('RGB'))
        image = cv.resize(image, (224, 160))
        mask = preprocess(mask_path)
        masks = [(mask == v) for v in range(1,3)]
        mask = np.stack(masks).astype('float')
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask

if __name__ == '__main__':
    dataset = NeoDataset(IMAGE_DIR, MASK_DIR)
    image, mask = dataset.__getitem__(1)
