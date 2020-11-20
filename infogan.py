import torch.nn as nn
import torch.nn.functional as F


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


class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

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
            nn.Linear(in_features=128 * 7 * 7,
                      out_features=1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        out = self.convT(x)
        out = self.fc(out)

        return out


class Auxillary(nn.Module):
    def __init__(self, n_cats=10, cont_codes_size=2):  # values for MNIST dataset
        super(Auxillary, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # categorical codes - softmax will be applied to get final output
        self.cat_codes = nn.Linear(in_features=128, out_features=n_cats)

        # continuous codes - for MNIST, using 2 cont. codes - network outputs mean and variance
        self.cont_codes_mu = nn.Linear(in_features=128, out_features=cont_codes_size)
        self.cont_codes_var = nn.Linear(in_features=128, out_features=cont_codes_size)

    def forward(self, x, shared_nw_obj):
        shared_nw_out = shared_nw_obj(x)

        cat_codes = F.softmax(self.cat_codes(self.fc(shared_nw_out)))

        cont_codes_mu = self.cont_codes_mu(self.fc(shared_nw_out)).squeeze()

        # taking exponent, so that variance is positive
        cont_codes_var = self.cont_codes_var(self.fc(shared_nw_out)).squeeze().exp()

        return cat_codes, cont_codes_mu, cont_codes_var


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.last = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, shared_nw_obj):
        shared_nw_out = shared_nw_obj(x)
        disc_out = self.last(shared_nw_out)

        return disc_out


if __name__ == "__main__":
    shared_obj = SharedNetwork()
    aux_model = Auxillary(n_cats=10)
    out = aux_model(x=1, shared_nw_obj=shared_obj)
