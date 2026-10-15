import torch
import torch.nn as nn
import torch.nn.functional as F


class VFE(nn.Module):
    def __init__(self, ch, p_kernel=(3, 5), reduction=16):
        super().__init__()
        red = max(1, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, ch, bias=False),
            nn.Sigmoid()
        )
        if p_kernel is None:
            p_kernel = (3, 5)
        self.dw = nn.ModuleList([
            nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False) for k in p_kernel
        ])
        self.pw = nn.Conv2d(ch * len(self.dw), ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, vi):
        w = self.fc(vi).unsqueeze(-1).unsqueeze(-1)
        feats = [dw(vi) for dw in self.dw]
        fused = self.pw(torch.cat(feats, dim=1))
        fused = self.act(self.bn(fused))
        return vi * w + fused


class IRFE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.edge = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        sobel = torch.tensor([[[-1., -2., -1.],
                               [ 0.,  0.,  0.],
                               [ 1.,  2.,  1.]]])
        sobel = sobel.repeat(ch, 1, 1, 1)
        with torch.no_grad():
            self.edge.weight.copy_(sobel)
        self.edge.weight.requires_grad = False
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, ir):
        edge_map = self.edge(ir)
        return ir + self.scale * edge_map
