import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
    def __init__(self, S: int=13, C: int=20, anchors: list=[], init_weight: bool = True) -> None:
        super().__init__()
        self.S = S
        self.C = C

        self.anchors=anchors

        _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(_resnet50.children())[:-2])

        self.head = nn.Sequential(
            Conv_BN_LeakyReLU(2048, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 2048, 3, 1),
            Conv_BN_LeakyReLU(2048, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 2048, 3, 1),
            Conv_BN_LeakyReLU(2048, 1024, 3, 1),

            Conv_BN_LeakyReLU(1024, 5*(5+C), 1)
        )

        if init_weight:
            self.head.apply(_initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        out = self.head(x)
        out = out.view(-1,self.S,self.S,5*(5+self.C))
        for batch in range(len(out)):
            for row in range(self.S):
                for col in range(self.S):
                    for i in range(len(self.anchors)):
                        xywhc=out[batch,row,col,i*(5+20):i*(5+20)+5].view(-1)
                        x=torch.sigmoid(xywhc[0])+row
                        y=torch.sigmoid(xywhc[1])+col
                        w=torch.exp(xywhc[2])*self.anchors[i][0]
                        h=torch.exp(xywhc[3])*self.anchors[i][1]
                        c=torch.sigmoid(xywhc[4])
                        out[batch,row,col,i*(5+20):i*(5+20)+5]=torch.stack([x,y,w,h,c])
        return out
    
def darknet19(num_classes: int = 1000, init_weight: bool = True) -> Yolov2:
    return Yolov2(num_classes=num_classes, init_weight=init_weight)

if __name__=='__main__':
    dummy_input = torch.randn(1, 3, 416, 416)

    darknet19 = darknet19()
    darknet19_features = darknet19.backbone

    print('Output Shape of DarkNet19: {}'.format(darknet19(dummy_input).shape))