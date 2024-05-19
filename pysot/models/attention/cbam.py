# ------------------------#
# CBAM模块的Pytorch实现
# ------------------------#
import torch
import torch.nn as nn
import torchvision

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()
    # self.avg_pool(x).shape: [1, 256, 1, 1]--->[1,256*1*1]--->[1, 256, 1, 1]
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3) # [1, 256, 1, 1]

        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)  #[1, 256, 1, 1]
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  #x:[1, 256, 31, 31]
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)  #[1, 1, 31, 31]
        maxout, _ = torch.max(x, dim=1, keepdim=True)  #[1, 1, 31, 31]
        out = torch.cat([avgout, maxout], dim=1)  #[1, 2, 31, 31]
        out = self.sigmoid(self.conv2d(out))  #self.conv2d(out).shape: [1, 1, 31, 31]
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):  #x:[1, 256, 31, 31]
        out = self.channel_attention(x) * x  #[1, 256, 31, 31]
        out = self.spatial_attention(out) * out
        return out  #[1, 256, 31, 31]

