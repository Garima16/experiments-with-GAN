import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.convt1 = nn.ConvTranspose2d(in_channels=nz,
                                         out_channels=ngf * 8,
                                         kernel_size=4,
                                         stride=1,
                                         padding=0,
                                         bias=False)
        self.batchnorm1 = nn.BatchNorm2d(ngf * 8)  # layer1 out: (ngf*8, 4x4)

        self.convt2 = nn.ConvTranspose2d(in_channels=ngf * 8,
                                         out_channels=ngf * 4,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 4)  # layer2 out: (ngf*4, 8x8)

        self.convt3 = nn.ConvTranspose2d(in_channels=ngf * 4,
                                         out_channels=ngf * 2,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 2)  # layer3 out: (ngf*2, 16x16)

        self.convt4 = nn.ConvTranspose2d(in_channels=ngf * 2,
                                         out_channels=ngf,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf)  # layer4 out: (ngf, 32x32)

        self.convt5 = nn.ConvTranspose2d(in_channels=ngf,
                                         out_channels=nc,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)

    def forward(self, x):
        layer1_out = F.relu(input=self.batchnorm1(self.convt1(x)), inplace=True)
        layer2_out = F.relu(input=self.batchnorm2(self.convt2(layer1_out)), inplace=True)
        layer3_out = F.relu(input=self.batchnorm3(self.convt3(layer2_out)), inplace=True)
        layer4_out = F.relu(input=self.batchnorm4(self.convt4(layer3_out)), inplace=True)

        output = torch.tanh(input=self.convt5(layer4_out))

        return output
