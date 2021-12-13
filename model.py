# -*- coding: utf-8 -*-
"""
💯 Created on Wed Dec  8 01:02:02 2021
🛡️ Give me your power
🧬 @author: Turtle 💕
🌐 Facebook: https://www.facebook.com/bk.turtle.1
"""
import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.up(x)
        return x
    
class AGN(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AGN, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.resampler = up_conv(ch_in=1, ch_out=1)
    
    def forward(self, g, x):
        '''
        g: input
        x: skip_connection
        '''
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        psi = self.resampler(psi)
        return psi * x

class NeoUnet(nn.Module):
    def __init__(self):
        super(NeoUnet, self).__init__()
        self.model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
        children = list(self.model.children())
        self.conv1 = nn.Sequential(*children[0][0:2])
        self.conv2 = nn.Sequential(*children[0][2:5]) 
        self.conv3 = nn.Sequential(*children[0][5:10]) 
        self.conv4 = nn.Sequential(*children[0][10:13]) 
        self.conv5 = nn.Sequential(*children[0][13:16])
        
        self.AGN1 = AGN(F_g=1024, F_l=640, F_int=320)
        self.up5 = up_conv(ch_in=1024, ch_out=640)
        self._conv5 = conv_block(ch_in=1280, ch_out=640)
        
        self.AGN2 = AGN(F_g=640, F_l=320, F_int=128)
        self.up4 = up_conv(ch_in=640, ch_out=320)
        self._conv4 = conv_block(ch_in=640, ch_out=320)
        
        self.AGN3 = AGN(F_g=320, F_l=128, F_int=64)
        self.up3 = up_conv(ch_in=320, ch_out=128)
        self._conv3 = conv_block(ch_in=256,ch_out=128)
        
        self.AGN4 = AGN(F_g=128, F_l=64, F_int=32)
        self.up2 = up_conv(ch_in=128, ch_out=64)
        self._conv2 = conv_block(ch_in=128, ch_out=64)
        
        self.up1 = up_conv(ch_in=64, ch_out=32)
        self._conv1 = conv_block(ch_in=35, ch_out=32)
        
        self._conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
    
    def forward(self, im_data):
        x1 = self.conv1(im_data)  # 112x112x64
        x2 = self.conv2(x1)       # 56x56x128
        x3 = self.conv3(x2)       # 28x28x320
        x4 = self.conv4(x3)       # 14x14x640
        
        b = self.conv5(x4)       # 7x7x1024
        
        x4 = self.AGN1(b, x4)
        d4 = self.up5(b) # 14x14x640
        d4 = torch.cat((x4, d4), dim=1) # 14x14x1280
        d4 = self._conv5(d4) # 14x14x640
        
        x3 = self.AGN2(d4, x3)
        d3 = self.up4(d4) # 28x28x320
        d3 = torch.cat((x3, d3), dim=1) # 28x28x640
        d3 = self._conv4(d3) # 28x28x320
        
        x2 = self.AGN3(d3, x2)
        d2 = self.up3(d3) # 56x56x128
        d2 = torch.cat((x2, d2), dim=1) # 56x56x256
        d2 = self._conv3(d2) # 56x56x128
        
        x1 = self.AGN4(d2, x1)
        d1 = self.up2(d2) # 112x112x64
        d1 = torch.cat((x1, d1), dim=1) # 112x112x128
        d1 = self._conv2(d1) # 112x112x64
        
        d0 = self.up1(d1) # 224x224x32
        d0 = torch.cat((im_data, d0), dim=1) # 224x224x35
        d0 = self._conv1(d0) # 224x224x32
        
        d0 = self._conv(d0) # 224x224x3
        output = self.output(d0)
        
        return output
    
if __name__ == '__main__':
    model = NeoUnet()
    summary(model.cuda(), (3, 160, 224))
    x = torch.randn((3, 3, 160, 224))
    x = x.cuda()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    preds = torch.argmax(preds, dim=1)

