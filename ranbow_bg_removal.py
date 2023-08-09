import numpy as np
from itertools import chain
import os
import torch
from torch.utils.data import DataLoader

from model import Siren
from util import get_mgrid, jacobian, VideoFitting
import imageio

def train_reflection(path, total_steps, lambda_interf=0.02, lambda_flow=0.02, lambda_excl=0.0005, verbose=True, steps_til_summary=100):
    g = Siren(in_features=3, out_features=2, hidden_features=256,
              hidden_layers=4, outermost_linear=True)
    g.cuda()
    f1 = Siren(in_features=2, out_features=3, hidden_features=256,
               hidden_layers=4, outermost_linear=True)
    f1.cuda()
    f2 = Siren(in_features=3, out_features=3, hidden_features=256,
               hidden_layers=4, outermost_linear=True)
    f2.cuda()

    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f1.parameters(), f2.parameters()))

    v = VideoFitting(path)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 8
    # batch_size = 32
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xyt = model_input[start:end].requires_grad_()
        xy, t = xyt[:, :-1], xyt[:, [-1]]
        h = g(xyt)
        xy_ = xy + h
        o_scene = torch.sigmoid(f1(xy_))
        o_obst = torch.sigmoid(f2(torch.cat((xy, t), -1)))
        o = o_scene + o_obst
        loss_recon = ((o - ground_truth[start:end]) ** 2).mean()
        loss_interf = o_obst.abs().mean()
        loss_flow = jacobian(h, xyt).abs().mean()

        g_scene = jacobian(o_scene, xy_)
        g_obst = jacobian(o_obst, xy)
        n_scene = (g_obst.norm(dim=0, keepdim=True) / g_scene.norm(dim=0, keepdim=True)).sqrt()
        n_obst = (g_scene.norm(dim=0, keepdim=True) / g_obst.norm(dim=0, keepdim=True)).sqrt()
        loss_excl = (torch.tanh(n_scene * g_scene) * torch.tanh(n_obst * g_obst)).pow(2).mean()

        loss = loss_recon + lambda_interf * loss_interf + lambda_flow * loss_flow + lambda_excl * loss_excl

        if not step % steps_til_summary:
            print("Step [%04d/%04d]: recon=%0.8f, interf=%0.4f, flow=%0.4f, excl=%0.4f" % (step, total_steps, loss_recon, loss_interf, loss_flow, loss_excl))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return g, f1, f2, v.video

def pred(orig):
    with torch.no_grad():
        N, _, H, W = orig.size()
        xyt = get_mgrid([H, W, N]).cuda()
        h = g(xyt)
        o_scene = torch.sigmoid(f1(xyt[:, :-1] + h))
        o_obst = torch.sigmoid(f2(xyt))
        o_scene = o_scene.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_obst = o_obst.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        o_obst = (o_obst * 255).astype(np.uint8)
        o_scene = [o_scene[i] for i in range(len(o_scene))]
        o_obst = [o_obst[i] for i in range(len(o_obst))]
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        orig = (orig * 255).astype(np.uint8)
        orig = [orig[i] for i in range(len(orig))]

        return o_scene, o_obst, orig
if __name__ == "__main__":
    # 训练
    g, f1, f2, orig = train_reflection('./data/reflection', 3000)
    # 测试输出
    o_scene, o_obst, orig =  pred(orig)
    fn_orig = os.path.join('./data/reflecrtion_orig.mp4')
    fn_scene = os.path.join('./data/reflection_scene.mp4')
    fn_obst = os.path.join('./data/reflection_interf.mp4')
    imageio.mimwrite(fn_orig, orig, fps=1)
    imageio.mimwrite(fn_scene, o_scene, fps=1)
    imageio.mimwrite(fn_obst, o_obst, fps=1)