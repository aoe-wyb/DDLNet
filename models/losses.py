import torch.nn as nn
import torch
from PIL import Image
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models, transforms


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:  # True
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss_v3(nn.Module):
    '''
    contrastive = 0.0125 * (d_ap1 / (d_an3 + 1e-7)) + 0.025 * (d_ap1 / ( d_an2 + 1e-7)) + d_ap1 / (d_an1 + 1e-7)
    '''
    def __init__(self, ablation=True):

        super(ContrastLoss_v3, self).__init__()
        # vgg19 潜在的特征空间
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a1, a2, a3, p3, n1):
        # anchor 锚, positive 正, negative 负
        # return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        # 上采样 anchor
        a2 = F.interpolate(a2, scale_factor=1, mode='bilinear')  # 128 * 128
        a1 = F.interpolate(a1, scale_factor=2, mode='bilinear')  # 64 * 64

        # # 下采样 anchor
        # n2 = F.interpolate(n1, scale_factor=0.5, mode='bilinear')  # 128 * 128
        # n3 = F.interpolate(n1, scale_factor=0.25, mode='bilinear')  # 64 * 64

        a1_vgg, a2_vgg, a3_vgg = self.vgg(a1), self.vgg(a2), self.vgg(a3)
        # p1_vgg, p2_vgg, p3_vgg = self.vgg(p1), self.vgg(p2), self.vgg(p3)
        p3_vgg = self.vgg(p3)
        # n1_vgg, n2_vgg, n3_vgg = self.vgg(n1), self.vgg(n2), self.vgg(n3)
        n1_vgg = self.vgg(n1)
        loss = 0
        d_ap, d_an1, d_an2, d_an3 = 0, 0, 0, 0
        for i in range(len(a1_vgg)):
            # L1 Loss: (f(x) - y) / n
            # L2 Loss, Smooth L1
            # d_ap1 = self.l1(a1_vgg[i], p1_vgg[i].detach())
            # d_ap2 = self.l1(a2_vgg[i], p2_vgg[i].detach())
            d_ap3 = self.l1(a3_vgg[i], p3_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a1_vgg[i], n1_vgg[i].detach())
                d_an2 = self.l1(a2_vgg[i], n1_vgg[i].detach())
                d_an3 = self.l1(a3_vgg[i], n1_vgg[i].detach())

                # contrastive = d_ap3 / (d_an1 + 0.0125 * d_an2 + 0.025 * d_an3 + 1e-7)
                contrastive = d_ap3 / (d_an1 + d_an2 + d_an3 + 1e-7)
                # contrastive = 0.0125 * (d_ap3 / (d_an3 + 1e-7)) + 0.025 * (d_ap2 / ( d_an2 + 1e-7)) + d_ap1 / (d_an1 + 1e-7)

            else:
                contrastive = d_ap3

            loss += self.weights[i] * contrastive
        return loss


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        # vgg19 潜在的特征空间
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        # anchor 锚, positive 正, negative 负
        # return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            # L1 Loss: (f(x) - y) / n
            # L2 Loss, Smooth L1
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss



