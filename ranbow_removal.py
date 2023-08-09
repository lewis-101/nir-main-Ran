import imageio
import os
import numpy as np
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import model
from model import Siren, Homography
from util import get_mgrid, apply_homography, jacobian, VideoFitting



def train(path, total_steps, lambda_interf=0.001, lambda_excl=0.002, verbose=True, steps_til_summary=100):
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    v = VideoFitting(path, transform)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

    g = Homography(hidden_features=256, hidden_layers=2)
    g.cuda()
    f1 = Siren(in_features=2, out_features=3, hidden_features=256,
               hidden_layers=4, outermost_linear=True)
    f1.cuda()
    f2 = Siren(in_features=3, out_features=3, hidden_features=128,
               hidden_layers=4, outermost_linear=True)
    f2.cuda()

    # 三个模型的参数连接在一起，并创建一个Adam优化器
    params = chain(g.parameters(), f1.parameters(), f2.parameters())
    optim = torch.optim.Adam(lr=1e-4, params=params)

    model_input, ground_truth = next(iter(videoloader)) # coords, pixels
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 4
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xy = model_input[start:end, :-1].requires_grad_() # xy 就是除了帧数维度之外的坐标信息，用于表示像素点的空间位置。
        t = model_input[start:end, [-1]].requires_grad_() # t 就是仅包含帧数维度的坐标信息，用于表示时间信息。
        h = g(t)
        xy_ = apply_homography(xy, h)
        o_scene = f1(xy_)
        o_moire = f2(torch.cat((xy, t), -1))
        o = o_scene + o_moire
        loss_recon = ((o - ground_truth[start:end]) ** 2).mean()
        loss_interf = o_moire.abs().mean()

        g_scene = jacobian(o_scene, xy_)
        g_moire = jacobian(o_moire, xy)
        n_scene = (g_moire.norm(dim=0, keepdim=True) / g_scene.norm(dim=0, keepdim=True)).sqrt()
        n_moire = (g_scene.norm(dim=0, keepdim=True) / g_moire.norm(dim=0, keepdim=True)).sqrt()
        loss_excl = (torch.tanh(n_scene * g_scene) * torch.tanh(n_moire * g_moire)).pow(2).mean()

        loss = loss_recon + lambda_interf * loss_interf + lambda_excl * loss_excl

        if verbose and not step % steps_til_summary:
            print("Step [%04d/%04d]: recon=%0.4f, interf=%0.4f, excl=%0.4f" % (step, total_steps, loss_recon, loss_interf, loss_excl))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return g, f1, f2, v.video

def pred(orig):
    with torch.no_grad():
        N, _, H, W = orig.size()
        xyt = get_mgrid([H, W, N]).cuda()
        h = g(xyt[:, [-1]])
        o_scene = f1(apply_homography(xyt[:, :-1], h))
        o_moire = f2(xyt)
        o_scene = o_scene.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_moire = o_moire.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_scene = (np.clip(o_scene * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        o_moire = (np.clip(o_moire * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
        o_scene = [o_scene[i] for i in range(len(o_scene))]
        o_moire = [o_moire[i] for i in range(len(o_moire))]
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        orig = ((orig * 0.5 + 0.5) * 255).astype(np.uint8)
        orig = [orig[i] for i in range(len(orig))]

        return o_scene, o_moire, orig




if __name__ == "__main__":
    # 训练
    g, f1, f2, orig = train('./data/result_720_2', 3000)
    # 测试
    o_scene, o_moire, orig = pred(orig)
    # fn_orig = os.path.join('./data/moire_orig_2.mp4') #result_720_2
    # fn_scene = os.path.join('./data/moire_scene_2.mp4')
    # fn_moire = os.path.join('./data/moire_interf_2.mp4')
    fn_orig = os.path.join('./data/result_720_orig.mp4')  # result_720_2
    fn_scene = os.path.join('./data/result_720_scene.mp4')
    fn_moire = os.path.join('./data/result_720_interf.mp4')
    imageio.mimwrite(fn_orig, orig, fps=1)
    imageio.mimwrite(fn_scene, o_scene, fps=1)
    imageio.mimwrite(fn_moire, o_moire, fps=1)