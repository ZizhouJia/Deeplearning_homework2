import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class resnet50(nn.Module):
    def __init__(self,pretrain,type="classify"):
        super(resnet50,self).__init__()
        self.base=models.resnet50(pretrained=pretrain)
        self.type=type
        self.fc=nn.Linear(2048,65)

    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out_features=torch.mean(x.view(-1,x.size(1),x.size(2)*x.size(3)),2)
        out=self.fc(out_features)
        if(self.type=="classify"):
            return out
        if(self.type=="t_SNE"):
            return out_features

class my_net(nn.Module):
    def __init__(self,channels=64,type="classify",dropout=False,bn=False,pooling="avg"):
        super(my_net,self).__init__()
        self.type=type
        self.base=nn.Sequential(nn.Conv2d(3,channels,3,1,1),nn.BatchNorm2d(channels),nn.ReLU(),
        nn.Conv2d(channels,channels,3,1,1),nn.BatchNorm2d(channels),nn.ReLU(),nn.AvgPool2d(2),
        nn.Conv2d(channels,channels*2,3,1,1),nn.BatchNorm2d(channels*2),nn.ReLU(),
        nn.Conv2d(channels*2,channels*2,3,1,1),nn.BatchNorm2d(channels*2),nn.ReLU(),nn.AvgPool2d(2),
        nn.Conv2d(channels*2,channels*4,3,1,1),nn.BatchNorm2d(channels*4),nn.ReLU(),
        nn.Conv2d(channels*4,channels*4,3,1,1),nn.BatchNorm2d(channels*4),nn.ReLU())
        self.fc=nn.Linear(channels*4,65)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self,x):
        for name,module in self.base._modules.items():
            if(name=='avgpool'):
                break
            x=module(x)
        out_features=torch.mean(x.view(-1,x.size(1),x.size(2)*x.size(3)),2)
        out=self.fc(out_features)
        if(self.type=="classify"):
            return out
        if(self.type=="t_SNE"):
            return out_features

class my_net(nn.Module):
    def __init__(self,channels=64,type="classify",dropout=False,bn=False,pooling="avg"):
        super(my_net,self).__init__()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
