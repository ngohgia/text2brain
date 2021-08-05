import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

class SimpleConvResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn):
        super(SimpleConvResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act_fn = act_fn

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        return out

class ConvResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn):
        super(ConvResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act_fn = act_fn
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        identity = self.conv1(x)
        residue = self.bn1(identity)
        residue = self.act_fn(residue)
        residue = self.conv2(residue)
        out = identity + residue
        out = self.bn2(out)
        out = self.act_fn(out)
        return out


class TransConvResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn):
        super(TransConvResBlock3D, self).__init__()
        self.trans_conv1 = nn.ConvTranspose3d(in_channels,out_channels, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act_fn = act_fn

    def forward(self, x):
        out = self.trans_conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        return out


class ImageDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, act_fn=nn.Sigmoid, num_filter=256):
        super(ImageDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filter = num_filter
        # act_fn = nn.LeakyReLU(0.2, inplace=True)
        # act_fn = nn.ReLU()
        act_fn = nn.Hardtanh(min_val=-6, max_val=6)

        self.trans_1 = TransConvResBlock3D(self.in_channels, self.num_filter, act_fn)
        self.trans_2 = TransConvResBlock3D(self.num_filter, self.num_filter // 2, act_fn)
        self.trans_3 = TransConvResBlock3D(self.num_filter // 2, self.num_filter // 4, act_fn)

        self.out = SimpleConvResBlock3D(self.num_filter // 4, self.out_channels, act_fn)


    def forward(self,x):
        up_1 = self.trans_1(x)
        up_2 = self.trans_2(up_1)
        up_3 = self.trans_3(up_2)

        out = self.out(up_3)

        return out[:, :, 1:, 1:, 1:]

if __name__ == "__main__":
    x = torch.rand(2, 512, 5, 6, 5)
    decoder = ImageDecoder(64, 36)

    out = decoder(x)

    print(out.shape)
