import numpy as np
import os
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, latent_var_size=74):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_var_size, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
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
        fc_out = self.fc(x.float())
        fc_out = fc_out.view(-1, 128, 7, 7)
        output = self.conv(fc_out)

        return output


# Shared network part between Recognition Network and Discriminator
class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

        # input size: (1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # output size: (64, 14, 14)

            nn.Conv2d(in_channels=64,
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
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        # print("input size: ", x.size())
        out = self.conv1(x)
        # print("shared network conv1 size: ", out.size())
        out = self.conv2(out)
        # print("shared network conv2 size: ", out.size())
        out = out.view(-1, 128 * 7 * 7)
        # print("shared network conv2 size: ", out.size())
        out = self.fc(out)
        # print("shared network fc size: ", out.size())
        return out


class RecognitionNetwork(nn.Module):
    def __init__(self, n_cats=10, cont_codes_size=2):  # values for MNIST dataset
        super(RecognitionNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # categorical codes - softmax will be applied to get final output
        self.cat_codes = nn.Linear(in_features=128, out_features=n_cats)

        # continuous codes - for MNIST, using 2 cont. codes - network outputs mean and variance
        self.cont_codes_mu = nn.Linear(in_features=128, out_features=cont_codes_size)
        self.cont_codes_var = nn.Linear(in_features=128, out_features=cont_codes_size)

    def forward(self, x):
        cat_codes = F.softmax(self.cat_codes(self.fc(x)), dim=1)
        cont_codes_mu = self.cont_codes_mu(self.fc(x)).squeeze()

        # taking exponent, so that variance is positive
        cont_codes_var = self.cont_codes_var(self.fc(x)).squeeze().exp()

        return cat_codes, cont_codes_mu, cont_codes_var


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.last = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        # print("input size to D: ", x.size())
        disc_out = torch.sigmoid(self.last(x))
        # print("discriminator output size: ", disc_out.size())
        return disc_out


class LogGaussian(object):

    # this method will be called on an instance of this class, by passing params to the instance
    def __call__(self, x, mu, var):
        # taking log of Gaussian distribution
        log_likelihood = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                         (x - mu).pow(2).div(var.mul(2) + 1e-6)
        return log_likelihood.sum(1).mean().mul(-1)


class InfoGAN(object):
    def __init__(self, noise_dim, disc_codes_dim, cont_code1_dim, cont_code2_dim, bs, image_size, epochs, dis_lr,
                 gen_lr):

        # generator_net = Generator().to(self.device)
        # self.discriminator_net = Discriminator().to(self.device)
        # recognition_net = RecognitionNetwork().to(self.device)
        # shared_nw = SharedNetwork().to(self.device)

        self.bs = bs
        self.image_size = image_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.dis_lr = dis_lr  # D's learning rate
        self.gen_lr = gen_lr  # G's learning rate

        self.noise_dim = noise_dim
        self.disc_codes_dim = disc_codes_dim
        self.cont_code1_dim = cont_code1_dim
        self.cont_code2_dim = cont_code2_dim

        self.z_dim = self.noise_dim + self.disc_codes_dim + self.cont_code1_dim + self.cont_code2_dim

        self.criterionD = nn.BCELoss()
        self.criterionRN_disc = nn.CrossEntropyLoss()
        self.criterionRN_cont = LogGaussian()

    def generate_noise_input(self, noise, disc_code, cont_code):
        # generate a random integer from the range [0, 10) of size batch size
        # this integer will be fed as input to G as one-hot code.
        idx = np.random.randint(low=10, size=self.bs)
        c = np.zeros((self.bs, self.disc_codes_dim))
        c[range(self.bs), idx] = 1.0  # create one hot encoding

        disc_code.data.copy_(torch.Tensor(c))
        cont_code.data.uniform_(-1, 1)
        noise.data.uniform_(-1, 1)

        z = torch.cat([noise, disc_code, cont_code], 1).view(-1, self.z_dim)

        return z, idx, cont_code

    def train(self, dataloader, discriminator_net, generator_net, recognition_net, shared_net, img_save_filepath,
              wts_file):
        # real_x = torch.FloatTensor(self.bs, 1, self.image_size, self.image_size).to(self.device)

        torch.autograd.set_detect_anomaly(True)

        # optimizers
        optimG = optim.Adam([{'params': generator_net.parameters()}, {'params': recognition_net.parameters()}],
                            lr=self.gen_lr, betas=(0.5, 0.999))
        optimD = optim.Adam([{'params': discriminator_net.parameters()}, {'params': shared_net.parameters()}],
                            lr=self.dis_lr, betas=(0.5, 0.999))

        noise = torch.FloatTensor(self.bs, self.noise_dim).to(self.device)
        disc_code = torch.FloatTensor(self.bs, self.disc_codes_dim).to(self.device)
        cont_code = torch.FloatTensor(self.bs, self.cont_code1_dim + self.cont_code2_dim).to(self.device)

        real_label = 1
        fake_label = 0

        # fixed random variables, to generate same 100 images after every specific number of iterations
        c = np.linspace(start=-1, stop=1, num=10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)
        c1 = torch.from_numpy(np.hstack([c, np.zeros_like(c)]))
        c2 = torch.from_numpy(np.hstack([np.zeros_like(c), c]))

        # 10 one-hot codes for 10 classes(10 digits). Thus, attempting to generate 10 images of each of the 10 digits
        idx = np.arange(10).repeat(10)
        one_hot = torch.from_numpy(np.zeros((100, 10)))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, self.noise_dim).uniform_(-1, 1)
        img_count = 1  # iteratively increase count to name image files differently
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader):
                # updating D
                optimD.zero_grad()

                # real image input to D
                real_img = data[0].to(self.device)
                batch_size = real_img.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)

                # resize noise and codes according to batch size of current batch
                noise.data.resize_(batch_size, self.noise_dim)
                disc_code.data.resize_(batch_size, self.disc_codes_dim)
                cont_code.data.resize_(batch_size, self.cont_code1_dim + self.cont_code2_dim)

                shared_obj = shared_net(real_img.detach())
                output = discriminator_net(shared_obj).view(-1)
                d_error_real = self.criterionD(output, label)
                d_error_real.backward(retain_graph=True)
                d_x = output.mean().item()

                # fake image input to D
                z, idx, cont_code = self.generate_noise_input(noise, disc_code, cont_code)
                # idx is the digit class whose discrete code is fed to G.
                # idx will be used to calculate discrete code loss to update recognition network parameters
                fake_img = generator_net(z)

                label.fill_(fake_label)
                shared_nw_out = shared_net(fake_img)
                disc_fake_output = discriminator_net(shared_nw_out).view(-1)
                d_error_fake = self.criterionD(disc_fake_output, label)
                d_error_fake.backward(retain_graph=True)

                d_error = d_error_real + d_error_fake
                optimD.step()

                # updating G and Q
                optimG.zero_grad()
                label.fill_(real_label)

                # disc_fake_output computed above can't be used here again, since backward() has already
                # been called upon it and has changed. so, calculate it again.
                shared_nw_out = shared_net(fake_img)
                disc_fake_output = discriminator_net(shared_nw_out).view(-1)

                d_error_fake2 = self.criterionD(disc_fake_output, label)
                q_logits, q_mu, q_var = recognition_net(shared_nw_out)
                class_label = torch.LongTensor(idx).to(self.device)
                discrete_loss = self.criterionRN_disc(q_logits, class_label)
                cont_loss = self.criterionRN_cont(cont_code, q_mu, q_var)

                g_error = d_error_fake2 + discrete_loss + cont_loss
                g_error.backward(retain_graph=True)
                optimG.step()

                # print results and generate 100 fake images after model has seen 100 batches of data in each epoch
                if i % 100 == 0:
                    print("Epoch:{}/{}\tBatch:{}/{}\tD-error:{}\tG-error:{}".
                          format(epoch, self.epochs, i, len(dataloader), d_error, g_error))

                    # noise.data.copy_(fix_noise)
                    # disc_code.data.copy_(torch.Tensor(one_hot))
                    #
                    # cont_code.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([fix_noise, one_hot, c1], 1).view(-1, 74)  # size=(100, 74, 1, 1)
                    print("z size: ", z.size())
                    # z = torch.cat([noise, disc_code, cont_code], 1).view(-1, 74, 1, 1)
                    fake_img = generator_net(z).detach()
                    save_image(fake_img.data, '%s/image_c1_%d.png' % (img_save_filepath, img_count), nrow=10,
                               normalize=True)

                    # cont_code.data.copy_(torch.from_numpy(c2))
                    z = torch.cat([fix_noise, one_hot, c2], 1).view(-1, 74)  # size=(100, 74, 1, 1)
                    fake_img = generator_net(z).detach()
                    save_image(fake_img.data, '%s/image_c2_%d.png' % (img_save_filepath, img_count), nrow=10,
                               normalize=True)

                    img_count += 1

            torch.save(generator_net.state_dict(), '%s/Gwts_epoch%d' % (wts_file, epoch))
            torch.save(discriminator_net.state_dict(), '%s/Dwts_epoch%d' % (wts_file, epoch))
            torch.save(recognition_net.state_dict(), '%s/RNwts_epoch%d' % (wts_file, epoch))
            torch.save(shared_net.state_dict(), '%s/SNwts_epoch%d' % (wts_file, epoch))
