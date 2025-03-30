import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

def _initialize_weights(module):
    """Initialize the weights of convolutional, batch normalization, and linear layers"""

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        return self.convs(x)

class Yolov2(nn.Module):
    def __init__(self, S: int=13, C: int=20, anchors: list=[[1,1],
                                                            [2,2],
                                                            [3,3],
                                                            [4,4],
                                                            [5,5]], init_weight: bool = True) -> None:
        super().__init__()
        self.S = S
        self.C = C

        self.anchors=torch.as_tensor(anchors)
        self.num_anchors=len(anchors)

        _resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(_resnet18.children())[:-2])

        self.head = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 5*(5+C), 1),
        )

        if init_weight:
            self.head.apply(_initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out = self.head(x)
        # 将 out reshape 为 (B, S, S, A, 5+C)
        out = out.view(len(x), self.S, self.S, self.num_anchors, 5 + self.C)
        pred = out[..., :5]
        grid_x = torch.arange(self.S, device=out.device).view(1, self.S, 1, 1)  # 对应 row（Y）
        grid_y = torch.arange(self.S, device=out.device).view(1, 1, self.S, 1)  # 对应 col（X）
        # ----------------------------------------------------------------------#
        x = torch.sigmoid(pred[..., 0]) + grid_x
        y = torch.sigmoid(pred[..., 1]) + grid_y
        anchor_w = self.anchors[:, 0].view(1, 1, 1, self.num_anchors)  # shape: (1,1,1,A)
        anchor_h = self.anchors[:, 1].view(1, 1, 1, self.num_anchors)  # shape: (1,1,1,A)
        w = torch.exp(pred[..., 2]) * anchor_w
        h = torch.exp(pred[..., 3]) * anchor_h
        c = torch.sigmoid(pred[..., 4])
        # ----------------------------------------------------------------------#
        # 将变换后的5个数合并回去，得到 shape (B, S, S, A, 5)
        transformed = torch.stack([x, y, w, h, c], dim=-1)
        out[..., :5] = transformed
        return out.view(-1, self.S, self.S, self.num_anchors * (5 + self.C))
    
def darknet19(init_weight: bool = True) -> Yolov2:
    return Yolov2(init_weight=init_weight)

if __name__=='__main__':
    dummy_input = torch.randn(1, 3, 416, 416)

    darknet19 = darknet19()
    darknet19_features = darknet19.backbone

    print('Output Shape of DarkNet19: {}'.format(darknet19(dummy_input).shape))