import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from .AME import VFE, IRFE


class ECCA(nn.Module):
    def __init__(self, ch, reduction=8, drop=0.05, align_scale_init=0.5, center=True):
        super().__init__()
        r = max(4, ch // reduction)
        self.r = r
        self.q = nn.Conv2d(ch, r, 1, bias=False)
        self.k = nn.Conv2d(ch, r, 1, bias=False)
        self.v = nn.Conv2d(ch, r, 1, bias=False)
        self.proj = nn.Conv2d(r, ch, 1, bias=False)
        self.norm = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(drop)
        self.center = center
        self.align_scale = nn.Parameter(torch.tensor(float(align_scale_init)))

    def forward(self, a, b):
        B, C, H, W = a.shape
        N = H * W

        Qa = self.q(a).view(B, self.r, N)
        Kb = self.k(b).view(B, self.r, N)
        Vb = self.v(b).view(B, self.r, N)

        if self.center:
            Qa = Qa - Qa.mean(dim=2, keepdim=True)
            Kb = Kb - Kb.mean(dim=2, keepdim=True)

        M = torch.bmm(Qa, Kb.transpose(1, 2)) / (N ** 0.5 + 1e-6)
        M = M - M.max(dim=-1, keepdim=True).values
        M = F.softmax(M, dim=-1)

        out_flat = torch.bmm(M, Vb)
        out = out_flat.view(B, self.r, H, W)
        out = self.proj(out)
        out = self.norm(out)
        out = self.drop(out)

        return a + torch.clamp(self.align_scale, 0.0, 1.5) * out


class AMAF(nn.Module):
    def __init__(self,
                 ch=256,
                 p_kernel=(3, 5),
                 fuse='add',
                 reduction=4,
                 use_ecca=True,
                 drop=0.02,
                 align_scale_init=0.3,
                 alpha_init=0.35,
                 learnable_fuse=True
                 ):
        super().__init__()
        self.vfe = VFE(ch, p_kernel=p_kernel, reduction=max(8, 2 * reduction))
        self.irfe = IRFE(ch)

        self.use_ecca = use_ecca
        if self.use_ecca:
            self.ecca_vi = ECCA(ch, reduction=reduction, drop=drop,
                                         align_scale_init=align_scale_init)
            self.ecca_ir = ECCA(ch, reduction=reduction, drop=drop,
                                         align_scale_init=align_scale_init)
        else:
            self.ecca_vi = None
            self.ecca_ir = None

        self.alpha_vi = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_ir = nn.Parameter(torch.tensor(alpha_init))

        self.fuse = fuse
        self.learnable_fuse = learnable_fuse
        if self.fuse == 'cat':
            self.proj = nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False)
        elif self.fuse == 'add' and self.learnable_fuse:
            self.fuse_logits = nn.Parameter(torch.tensor([0.0, 0.0]))

    def _fuse_add(self, out_vi, out_ir):
        if self.learnable_fuse:
            w = F.softmax(self.fuse_logits, dim=0)
            return w[0] * out_vi + w[1] * out_ir
        else:
            return 0.5 * (out_vi + out_ir)

    def forward(self, x):
        vi, ir = x[0], x[1]

        vi_enh = self.vfe(vi)
        ir_enh = self.irfe(ir)

        if self.use_ecca:
            vi_att = self.ecca_vi(vi_enh, ir_enh)
            ir_att = self.ecca_ir(ir_enh, vi_enh)
        else:
            vi_att, ir_att = vi_enh, ir_enh

        out_vi = vi + torch.clamp(self.alpha_vi, 0.0, 1.0) * vi_att
        out_ir = ir + torch.clamp(self.alpha_ir, 0.0, 1.0) * ir_att

        out_vi = 0.95 * out_vi + 0.05 * vi
        out_ir = 0.95 * out_ir + 0.05 * ir

        if self.fuse == 'add':
            out = self._fuse_add(out_vi, out_ir)
        elif self.fuse == 'cat':
            out = torch.cat([out_vi, out_ir], dim=1)
            out = self.proj(out)
        else:
            raise ValueError(f"Unsupported fuse mode: {self.fuse}")

        return out
