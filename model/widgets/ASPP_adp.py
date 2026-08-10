import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(ASPP, self).__init__()
        
        # 保存通道数参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 计算layer5的输入通道数（5个分支的输出拼接）
        self.layer5_in_channels = 5 * out_channels

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        # 关键修改：layer5的输入通道数应该是5 * out_channels
        self.layer5 = nn.Sequential(
            nn.Conv2d(self.layer5_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        
        # 拼接所有分支
        out = torch.cat([
            global_feature, 
            self.layer1(x), 
            self.layer2(x), 
            self.layer3(x),
            self.layer4(x)
        ], dim=1)
        
        out = self.layer5(out)
        return out


# 创建一个工厂函数，用于根据不同的reduce_dim创建ASPP实例
def create_adaptive_aspp(reduce_dim):
    """创建自适应ASPP模块，根据给定的通道数"""
    return ASPP(in_channels=reduce_dim, out_channels=reduce_dim)
