import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_var_size):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_var_size, out_features=1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=128 * 7 * 7),
            nn.BatchNorm2d(128 * 7 * 7),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            # input size: (128, 7, 7)
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # output size: (64, 14, 14)

            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.Tanh()  # output size: (1, 28, 28)
        )

    def forward(self, x):
        fc_out = self.fc(x)
        fc_out = fc_out.view(-1, 128, 7, 7)
        output = self.conv(fc_out)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # input size: (1, 28, 28)

        self.convT = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # output size: (64, 14, 14)

            nn.ConvTranspose2d(in_channels=64,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # output size: (128, 7, 7)
        )

        self.fc = nn.Sequential(
            nn.Linear()
        )

    def forward(self, x):
        pass
