import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils


def load_data(dataroot, img_size, bs, workers, dset_name='mnist'):
    """
    Create dataloader from data of given bs and transform images
    :param dataroot:
    :param img_size:
    :param bs:
    :param workers:
    :param dset_name:
    :return:
    """
    if dset_name == 'mnist':
        data = datasets.MNIST(root=dataroot,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                              ]))

    if dset_name == 'celeb':
        data = datasets.CelebA(root=dataroot,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    '''
    transforms.Normalize():
    specify mean and stddev for 3 channels - 1st tuple is mean for 3 channels, 2nd is stddev
    to normalize: img_pixel = img_pixel - mean / stddev
    '''

    dataloader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True, num_workers=workers)

    return dataloader


def print_sample_data(dataloader, bs, device):
    data_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("{} Training Images".format(bs))
    plt.imshow(
        np.transpose(vutils.make_grid(data_batch[0].to(device)[:bs], padding=2, normalize=True).cpu(), (1, 2, 0)))


def initialise_weights(model):
    """
    Initialise weights of Convolution and Batch norm layers

    Convolution layers: weights sampled from Gaussian distribution with mu=0.0, stddev=0.02
    Batch norm layers: weights sampled from Gaussian distribution with mu=1.0, stddev=0.02 and bias with 0
    :param model:
    :return: None
    """
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(tensor=model.weight, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(tensor=model.weight, mean=1.0, std=0.02)
        nn.init.zeros_(tensor=model.bias)

    print("Initialised weights.")


def save_wts(generator, discriminator, epoch, wts_file):
    """
    Save Discriminator and Generator weights after an epoch in given filepath.
    :param generator: model
    :param discriminator: model
    :param epoch:
    :param wts_file:
    :return: None
    """
    torch.save(generator.state_dict(), '%s/Gwts_epoch%d' % (wts_file, epoch))
    torch.save(discriminator.state_dict(), '%s/Dwts_epoch%d' % (wts_file, epoch))
    print("Saved weights.")


def load_wts(generator, discriminator, gen_wts_file, dis_wts_file):
    """
    Load Discriminator and Generator models with previously saved weights.
    :param generator:
    :param discriminator:
    :param gen_wts_file: file containing G wts
    :param dis_wts_file: file containing D wts
    :return:
    """
    generator.load_state_dict(torch.load(gen_wts_file))
    discriminator.load_state_dict(torch.load(dis_wts_file))
    print("Loaded weights.")

    return generator, discriminator


def initialise_wts(generator, discriminator):
    """
    Call to initialise_weights to initialise weights of D and G.
    :param generator:
    :param discriminator:
    :return:
    """
    generator.apply(initialise_weights)
    discriminator.apply(initialise_weights)

    return generator, discriminator


def generate_img_from_pretrained_generator(generator, bs, nz, device, imgs_dir='None'):
    """
    Generate images from a trained Generator model and plot images, and optionally, save them.
    :param generator: pre-trained model
    :param bs: batch size
    :param nz: size of latent variable
    :param device: cuda or cpu
    :param imgs_dir: images directory to save images, default: None
    :return:
    """
    fake_img_list = []
    fixed_noise = torch.randn(bs, nz, 1, 1, device=device)
    fake = generator(fixed_noise).detach().cpu()
    fake_img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    if imgs_dir:
        vutils.save_image(
            fake,
            '%s/%d_fake_samples.png' % (imgs_dir, bs),
            normalize=True
        )
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.show()


def plot_losses(g_losses, d_losses):
    """
    Plot G and D training losses.
    :param g_losses: List of G's loss values
    :param d_losses: List of D's loss values
    :return:
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
