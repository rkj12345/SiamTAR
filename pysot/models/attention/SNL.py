import torch.nn as nn
import torch

class Resblock(nn.Module):
    def __init__(self,in_channels):
        super(Resblock, self).__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):   #x:[1, 256, 31, 31]
        b, c, w, h = x.shape
        residual = x
        # [N, C, H * W]  [1, 256, 961]
        input_x = x.view(b, c, h * w)
        # [N, 1, C, H * W]  [1, 1, 256, 961]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]  [1, 1, 31, 31]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]  [1, 1, 961]
        context_mask = context_mask.view(b, 1, h * w)
        # [N, 1, H * W]  [1, 1, 961]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]  [1, 1, 961, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]#####[1, 1, 256, 961]  ####[1, 1, 961, 1]  输出：  [1, 1, 256, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]  [1, 256, 31, 31]
        context = context.view(b, c, 1, 1) + x
        x = residual + context
        return x
