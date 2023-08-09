import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import cv2

def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1) # 笛卡尔积 创建时空网格 最后5数值的维度代表时间维度(256,256,5,3)
    mgrid = mgrid.reshape(-1, len(sidelen)) # (327680 , 3)
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)  # (16384,8) -> (16384,9)
    h = h.view(-1, 3, 3)  # (16384,9) -> (16384,3,3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)  # (16384,2) -> (16384,3,1)
    o = torch.bmm(h, x).squeeze(-1)  # (16384,3,3)与(16384,3,1) 后两位矩阵相乘 -> (16384,3,1) -> (16384,3)
    o = o[:, :-1] / o[:, [-1]]  # 坐标归一化，将前两列的值除以最后一列的值，以获得归一化的坐标值。o[:, :-1]=(16384,2) o[:, [-1]]=(16384,1) 后者会有广播机制
    return o

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian


class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            #
            W, H = img.size
            if W >= 720 and H >= 1083:
                # 720 * 1083 宽高 分辨率太大 改成  570 * 838
                img = img.resize((570, 838))

            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels