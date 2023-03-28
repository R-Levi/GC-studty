import torch
import numpy as np
from torch import  nn
class Generator(nn.Module):
    def __init__(self,opt,img_shape):
        super(Generator,self).__init__()
        self.imgshape = img_shape
        def block(in_feat, out_feat, normalize=True):  # 对传入数据应用线性转换（输入节点数，输出节点数）
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # BN
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 激活函数
            return layers

        self.model = nn.Sequential(
            *block(opt.hidden_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.imgshape))),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.shape[0],*self.imgshape)
        return x