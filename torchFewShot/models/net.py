import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .res12 import ResNet


class ResNet12(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet12, self).__init__()
        self.base  = ResNet()

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        return f


class AngleLinear(nn.Module):
    def __init__(self, num_classes, scale_cls, nFeat=640, **kwargs):
        super(AngleLinear, self).__init__()
        self.scale_cls = scale_cls
        weight_base = torch.FloatTensor(num_classes, nFeat).normal_(
            0.0, np.sqrt(1.0/nFeat))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True) #[K, D]

    def forward(self, x):
        w = self.weight_base

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)
        return self.scale_cls * F.linear(x, w)


class Classifier(nn.Module):
    def __init__(self, num_classes, nFeat=640, **kwargs):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(nFeat, num_classes)

    def forward(self, x):
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    from res12 import ResNet

    net = ResNet12_norm(scale_cls=10)
    net = net.cuda()
    x = torch.rand(32, 3, 84, 84).cuda()
    print(net)
    y, f = net(x)
    print (y.size())
    print (f.size())
