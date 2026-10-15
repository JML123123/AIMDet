import torch
import torch.nn as nn
import torch.nn.functional as F

class FGD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.key_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.noise_base = nn.Parameter(torch.tensor(0.1))
        self.noise_scale = nn.Parameter(torch.tensor(2.5))
        self.denoise_down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x, seed=4399):
        key_mask = self.key_feature_extractor(x)
        key_mask = torch.sigmoid(key_mask)
        if self.training:
            noise_strength = self.noise_base * (self.noise_scale * key_mask + 0.1 * (1 - key_mask))
            if seed is not None:
                state = torch.get_rng_state()
                torch.manual_seed(seed)
                noise = torch.randn_like(x) * noise_strength
                torch.set_rng_state(state)
            else:
                noise = torch.randn_like(x) * noise_strength
            x_noisy = x + noise
        else:
            x_noisy = x

        x_down = self.denoise_down(x_noisy)
        return x_down
