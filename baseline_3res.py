import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
train_size = (1, 3, 256, 256)


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


# --------------------------------------------------------------------------------


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)




class Gap_conv1(nn.Module):
    def __init__(self, in_channel, mode, groups=1, b=1, gamma=2) -> None:
        super().__init__()

        # self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.gap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.gap = AvgPool2d(base_size=210)

        t = int(abs((math.log(in_channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv_l = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        # self.conv_h = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.out = nn.Conv2d(in_channel * 2, in_channel, 1, padding=0, bias=True)
        # self.ca_l = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channel // 8, in_channel, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )
        # self.ca_h = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channel // 8, in_channel, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        gap = self.gap(x)
        # x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        # x_d = x_d * self.fscale_d[None, :, None, None]
        # x_lower = self.ca_l(x_l) * x
        # x_high = self.ca_l(x - x_l) * x
        x_l = gap.squeeze(-1).transpose(-1, -2)  # 1,1,32
        x_l = self.conv_l(x_l).transpose(-1, -2).unsqueeze(-1)
        x_lower = self.sigmoid(x_l) * x
        #
        x_h = (x - x_l) * (self.fscale_h[None, :, None, None] + 1.)

        x_high = self.sigmoid(x_h) * x
        return x_lower + x_high
        # return self.out(torch.cat([x_lower, x_high], dim=1))


class Patch_ap_conv1d(nn.Module):
    def __init__(self, mode, inchannel, patch_size, b=1, gamma=2):
        super(Patch_ap_conv1d, self).__init__()

        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1, 1))
        elif mode[0] == 'test':
            if mode[1] == 'ITS-test':
                self.ap = nn.AdaptiveAvgPool2d((1, 1))
            elif mode[1] == 'OTS-test':
                self.ap = nn.AdaptiveAvgPool2d((1, 1))
            elif mode[1] == 'Dense-test':
                self.ap = nn.AdaptiveAvgPool2d((1, 1))
            elif mode[1] == 'NH-test':
                self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.patch_size = patch_size
        self.channel = inchannel * patch_size ** 2

        t = int(abs((math.log(self.channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.conv_l = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fscale_h = nn.Parameter(torch.zeros(self.channel), requires_grad=True)
        # self.conv_h = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.ca = nn.Sequential(*[
        #     nn.Conv2d(self.channel, self.channel // 16, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     # GRN(self.channel // 16),
        #     nn.Conv2d(self.channel // 16, self.channel, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # ])

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        gap = self.ap(patch_x)  # c*1*1

        x_l = gap.squeeze(-1).transpose(-1, -2)  # 1,1,32
        x_l = self.conv_l(x_l).transpose(-1, -2).unsqueeze(-1)
        x_lower = self.sigmoid(x_l) * patch_x
        # print(self.fscale_h.shape)
        x_h = (patch_x - x_l) * (self.fscale_h[None, :, None, None] + 1.)
        x_high = self.sigmoid(x_h) * patch_x

        out = rearrange(x_lower + x_high, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size,
                        p2=self.patch_size)

        return out


class DynamicDWConv_(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        # self.bn = nn.BatchNorm2d(dim // reduction)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        self.conv1 = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        self.conv2 = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        self.act = nn.Softmax()
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        # weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        avg_pool = self.conv1(self.avg_pool(x))
        max_pool = self.conv2(self.max_pool(x))
        weight = (max_pool - avg_pool).view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        weight = self.act(weight)
        # weight_b = weight - weight_sigmoid

        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding,
                     groups=b * self.groups)

        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x + shortcut


class DynamicDWConv_1(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        # self.bn = nn.BatchNorm2d(dim // reduction)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        self.conv1 = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        # self.conv2 = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        self.act = nn.Sigmoid()
        self.fscale_h = nn.Parameter(torch.zeros(dim), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        # weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        avg_pool = self.conv1(self.avg_pool(x))
        # max_pool = self.conv2(self.max_pool(x))
        weight = avg_pool.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        weight_sigmoid = self.act(weight)
        # weight_b = weight - weight_sigmoid

        x_sigmoid = F.conv2d(x.reshape(1, -1, h, w), weight_sigmoid, self.bias.repeat(b), stride=self.stride,
                             padding=self.padding,
                             groups=b * self.groups)

        x_sigmoid = x_sigmoid.view(b, c, x.shape[-2], x.shape[-1])
        x_b = (x - x_sigmoid) * self.fscale_h[None, :, None, None] * x
        return x_b + x_sigmoid


class PDD(nn.Module):
    """
    kernel size: 5
    MPL: kernel size: 1
    """

    def __init__(self, in_channel, kernel_size=5):
        super(PDD, self).__init__()

        self.Wv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channel,
                      padding_mode='reflect'),
            nn.Conv2d(in_channel, in_channel, 1)
        )
        self.td = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                      groups=in_channel,
                      padding_mode='reflect'),
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // 8, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # GRN(in_channel // 8),
            nn.Conv2d(in_channel // 8, in_channel, 1, stride=1, padding=0),
            # nn.Conv2d(in_channel, in_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.Wv(x)
        t = self.td(x)

        j = torch.mul(t, v) + torch.mul((1 - t), v)

        return j


class PDD_(nn.Module):
    """
    kernel size: 5
    MPL: kernel size: 1
    """

    def __init__(self, in_channel, kernel_size=3):
        super(PDD_, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.zeros(in_channel))

        self.norm = nn.BatchNorm2d(in_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ka = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * kernel_size * kernel_size, 1),
            nn.BatchNorm2d(in_channel * kernel_size * kernel_size),
            # nn.Conv2d(in_channel, in_channel // 8, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channel // 8, in_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.Wv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channel,
                      padding_mode='reflect')
        )
        self.td = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                      groups=in_channel,
                      padding_mode='reflect'),
            nn.Conv2d(in_channel, in_channel // 8, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 8, in_channel, 1, stride=1, padding=0),
            # nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        avg_pool = self.ka(self.avg_pool(x))
        weight = avg_pool.view(b * self.in_channel, 1, self.kernel_size, self.kernel_size)

        v = self.Wv(x)
        t = self.td(x)

        x_b = torch.mul(t, v)

        x_sigmoid = F.conv2d((x-x_b).reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=1,
                             padding=1,
                             groups=b * self.in_channel)
        x_sigmoid = x_sigmoid.view(b, c, x.shape[-2], x.shape[-1])

        # f = torch.mul((x - b), a)

        return x_sigmoid + x_b
        # return self.reconstruct(f, b)

    # def reconstruct(self, x_1, x_2):
    #     x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
    #     x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
    #     return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class ResBlock(nn.Module):
    '''
    Total_params: ==> 6.136437
    '''

    def __init__(self, in_channel, out_channel, mode, window_size=3, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        # self.dyna = dynamic_filter(in_channel // 2, mode) if filter else nn.Identity()
        # self.dyna_2 = dynamic_filter(in_channel // 2, mode, kernel_size=5) if filter else nn.Identity()
        #
        # self.localap = Patch_ap(mode, in_channel // 2, patch_size=2)
        # self.global_ap = Gap_conv1(in_channel, mode) if filter else nn.Identity()
        #self.global_ap = Patch_ap_conv1d(mode, in_channel // 2, patch_size=1) if filter else nn.Identity()
        #self.localap = Patch_ap_conv1d(mode, in_channel // 2, patch_size=2) if filter else nn.Identity()
        # self.DDWConv = DynamicDWConv_1(in_channel, kernel_size=window_size, stride=1, padding=window_size // 2,
        #                               groups=in_channel) if filter else nn.Identity()
        #self.DDWConv = PDD(in_channel // 2, kernel_size=3) if filter else nn.Identity()
        #self.DDWConv5 = PDD(in_channel // 2, kernel_size=5) if filter else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        # out = self.DDWConv(out)
        # out = self.global_ap(out)
        #if self.filter:
            #non_local, local = torch.chunk(out, 2, dim=1)
            #non_local = self.global_ap(non_local)
            #local = self.localap(local)
            #out = torch.cat((non_local, local), dim=1)

            #non_local, local = torch.chunk(out, 2, dim=1)
            #non_local = self.DDWConv(non_local)
            #local = self.DDWConv5(local)
            #out = torch.cat((non_local, local), dim=1)

        # out = self.DDWConv(out)
        # out = self.global_ap(out)
        # out = self.global_ap(out)
        # non_local, local = torch.chunk(out, 2, dim=1)
        # out = self.global_ap(out)
        # out = self.DDWConv(out)
        # local = self.localap(local)
        # out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()
        # layers = [ResBlock(out_channel, out_channel, mode) for _ in range(num_res)]
        # layers = [ResBlock(out_channel, out_channel, mode, filter=True) for _ in range(num_res - 1)]
        layers = [ResBlock(out_channel, out_channel, mode, filter=False) for _ in range(num_res-2)]
        #layers.append(ResBlock(out_channel, out_channel, mode, filter=True))
        #layers.append(ResBlock(out_channel, out_channel, mode, filter=True))
        #layers.append(ResBlock(out_channel, out_channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()
        # layers = [ResBlock(channel, channel, mode) for _ in range(num_res)]
        # layers = [ResBlock(channel, channel, mode, filter=True) for _ in range(num_res - 1)]
        layers = [ResBlock(channel, channel, mode, filter=False) for _ in range(num_res-2)]
        #layers.append(ResBlock(channel, channel, mode, filter=True))
        #layers.append(ResBlock(channel, channel, mode, filter=True))
        #layers.append(ResBlock(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class SFNet(nn.Module):
    def __init__(self, mode, num_res=6):
        super(SFNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, mode),
            EBlock(base_channel * 2, num_res, mode),
            EBlock(base_channel * 4, num_res, mode),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, mode),
            DBlock(base_channel * 2, num_res, mode),
            DBlock(base_channel, num_res, mode)
        ])

        # self.Convs = nn.ModuleList([
        #     BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
        #     BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        # ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.skff1 = SKFF(base_channel, 2)
        self.skff2 = SKFF(base_channel * 2, 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256*256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128*128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64*64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128*128
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        # print(z.shape, res2.shape)
        z = self.skff2([z, res2])
        # z = torch.cat([z, res2], dim=1)
        # z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256*256
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = self.skff1([z, res1])
        # z = torch.cat([z, res1], dim=1)
        # z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net(mode):
    return SFNet(mode)
