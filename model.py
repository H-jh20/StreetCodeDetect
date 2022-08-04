import torch
from torch import nn
from torchvision import models

class SVHDmodule(nn.Module):
    def __init__(self,ckpt=True):
        super(SVHDmodule, self).__init__()
        module = models.resnet18(pretrained=ckpt)
        # self.module = module.
        self.backbone = nn.Sequential(*list(module.children())[:-1])
        self.fc1=nn.Linear(512,11)
        self.fc2=nn.Linear(512,11)
        self.fc3=nn.Linear(512,11)
        self.fc4=nn.Linear(512,11)
        self.fc5=nn.Linear(512,11)
        self.fc6=nn.Linear(512,11)


    def forward(self,x):
        y = self.backbone(x)
        y = y.view(y.shape[0],-1)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y5 = self.fc5(y)
        y6 = self.fc6(y)
        return y1,y2,y3,y4,y5,y6

    def print_module(self):
        print("*************************************************************************************")
        print(self.backbone)
