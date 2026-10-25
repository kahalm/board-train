"""Netz + Augmentierung, wörtlich aus ebook-board-training/hr/cnn.py (Stand 25.10.)."""
import torch, torch.nn as nn, torch.nn.functional as F
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(1,24),blk(24,48),blk(48,96))  # 32->4
        s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(96*16,128),nn.ReLU(),nn.Linear(128,13))
    def forward(s,x): return s.h(s.f(x))
def augment(x):
    # x: B,1,64,64 float in [0,1]
    B_=x.shape[0]
    th=torch.zeros(B_,2,3); sc=1+(torch.rand(B_)-0.5)*0.2
    th[:,0,0]=sc; th[:,1,1]=sc; th[:,:,2]=(torch.rand(B_,2)-0.5)*0.16
    grid=F.affine_grid(th,x.shape,align_corners=False); x=F.grid_sample(x,grid,padding_mode='border',align_corners=False)
    c=1+(torch.rand(B_,1,1,1)-0.5)*0.6; br=(torch.rand(B_,1,1,1)-0.5)*0.3
    x=((x-0.5)*c+0.5+br)
    m=torch.rand(B_)<0.25
    if m.any(): x[m]=(x[m]>torch.rand(int(m.sum()),1,1,1)*0.3+0.35).float()
    m=torch.rand(B_)<0.3
    if m.any(): x[m]=F.avg_pool2d(F.pad(x[m],(1,1,1,1),mode='replicate'),3,1)
    x=x+torch.randn_like(x)*0.03*torch.rand(B_,1,1,1)
    return x.clamp(0,1)
