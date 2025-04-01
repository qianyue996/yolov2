import torch
import torch.nn as nn

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        return self.convs(x)
    
class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_layer1 = Conv_BN_LeakyReLU(3,32)
        self.conv_layer2 = Conv_BN_LeakyReLU(32,64)
        self.conv_layer3 = nn.Sequential(Conv_BN_LeakyReLU(64,128),Conv_BN_LeakyReLU(128,64),Conv_BN_LeakyReLU(64,128))
        self.conv_layer4 = nn.Sequential(Conv_BN_LeakyReLU(128,256),Conv_BN_LeakyReLU(256,128),Conv_BN_LeakyReLU(128,256))
        self.conv_layer5 = nn.Sequential(Conv_BN_LeakyReLU(256,512),Conv_BN_LeakyReLU(512,256),Conv_BN_LeakyReLU(256,512),Conv_BN_LeakyReLU(512,256),Conv_BN_LeakyReLU(256,512))
        self.conv_layer6 = nn.Sequential(Conv_BN_LeakyReLU(512,1024),Conv_BN_LeakyReLU(1024,512),Conv_BN_LeakyReLU(512,1024),Conv_BN_LeakyReLU(1024,512),Conv_BN_LeakyReLU(512,1024))

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.maxpool(x)
        x = self.conv_layer2(x)
        x = self.maxpool(x)
        x = self.conv_layer3(x)
        x = self.maxpool(x)
        x = self.conv_layer4(x)
        x = self.maxpool(x)
        feature = self.conv_layer5(x)
        x = self.maxpool(feature)
        x = self.conv_layer6(x)
        return x,feature

class Detection(nn.Module):
    def __init__(self, is_Conv = True):
        super(Detection, self).__init__()
        self.is_conv = is_Conv
        self.conv_layer = Conv_BN_LeakyReLU(1024,1024)
        self.conv_decrease_channel = Conv_BN_LeakyReLU(512,64)
        self.passthrough = nn.Unfold([2, 2], stride=2, padding=0)
        self.pred = nn.Conv2d(1024, 125, (1, 1)) 
    def forward(self, x,feature):
        x = self.conv_layer(x)
        x = self.conv_layer(x)
        if self.is_conv:  
            feature = self.conv_decrease_channel(feature)
        _ , channel , h , w = feature.shape
        feature = self.passthrough(feature)
        feature = feature.reshape((-1, channel*4, h//2, w//2))
        x = torch.cat((x,feature),dim=1)
        x = Conv_BN_LeakyReLU(x.shape[1],1024).to(x.device)(x)
        x = self.pred(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self, scale = 32, is_Conv=True):
        super(YOLOv2, self).__init__()
        self.darknet = Darknet19()
        self.detection = Detection(is_Conv)
    def forward(self,x):
        x,feature = self.darknet(x)
        x = self.detection(x,feature)
        return x.view(-1,13,13,5,25)

if __name__=='__main__':
    input=torch.randn((1,3,416,416))
    # 查看Darknet网络
    net = YOLOv2()
    output = net(input)
    print(output.shape)