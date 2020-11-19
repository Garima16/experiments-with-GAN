import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz,
                               out_channels=ngf * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(ngf * 8),  # layer1 out: (ngf*8, 4x4)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 8,
                               out_channels=ngf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 4)  # layer2 out: (ngf*4, 8x8)

        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 4,
                               out_channels=ngf * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 2)  # layer3 out: (ngf*2, 16x16)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2,
                               out_channels=ngf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf)  # layer4 out: (ngf, 32x32)
        )

        self.last_layer = nn.ConvTranspose2d(in_channels=ngf,
                                             out_channels=nc,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1,
                                             bias=False)

    def forward(self, x):
        layer1_out = F.relu(input=self.layer1(x), inplace=True)
        layer2_out = F.relu(input=self.layer2(layer1_out), inplace=True)
        layer3_out = F.relu(input=self.layer3(layer2_out), inplace=True)
        layer4_out = F.relu(input=self.layer4(layer3_out), inplace=True)

        output = torch.tanh(input=self.last_layer(layer4_out))

        return output


class Discriminator(nn.Module):

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        # input to Discriminator layer 1 = (nc, 64x64), inplace=True ; ndf=64, ngf=64
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4,
                               stride=2, padding=1, bias=False)

        # input to Discriminator layer 2 = (ndf, 32x32)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4,
                               stride=2, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(ndf * 2)

        # input to Discriminator layer 3 = (ndf * 2, 16x16)
        self.conv3 = nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4,
                               stride=2, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(ndf * 4)

        # input to Discriminator layer 4 = (ndf * 4, 8x8)
        self.conv4 = nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4,
                               stride=2, padding=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(ndf * 8)

        # input to Discriminator layer 5 = (ndf * 8, 4x4)
        self.conv5 = nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4,
                               stride=1, padding=0, bias=False)

    def forward(self, x):
        layer1_out = F.leaky_relu(input=self.conv1(x),
                                  negative_slope=0.2,
                                  inplace=True)

        layer2_out = F.leaky_relu(input=self.batch_norm1(self.conv2(layer1_out)),
                                  negative_slope=0.2,
                                  inplace=True)

        layer3_out = F.leaky_relu(input=self.batch_norm2(self.conv3(layer2_out)),
                                  negative_slope=0.2,
                                  inplace=True)

        layer4_out = F.leaky_relu(input=self.batch_norm3(self.conv4(layer3_out)),
                                  negative_slope=0.2,
                                  inplace=True)

        output = torch.sigmoid(input=self.conv5(layer4_out))

        return output
