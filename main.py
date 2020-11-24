import infogan
from torch import cuda
import utils

if __name__ == "__main__":
    root_dir = 'give filepath'

    # declare constants for InfoGAN training
    bs = 128
    noise_dim = 62
    disc_codes_dim = 10
    cont_code1_dim = 1
    cont_code2_dim = 1
    img_size = 28
    epochs = 10
    device = 'cuda' if cuda.is_available() else 'cpu'

    dataloader = utils.load_data(dataroot=root_dir, img_size=img_size, bs=bs)

    utils.print_sample_data(dataloader=dataloader,
                            bs=bs,
                            device=device)

    infogan_obj = infogan.InfoGAN(noise_dim=noise_dim,
                                  disc_codes_dim=disc_codes_dim,
                                  cont_code1_dim=cont_code1_dim,
                                  cont_code2_dim=cont_code2_dim,
                                  bs=bs,
                                  image_size=img_size,
                                  epochs=epochs)

    disc_net = infogan.Discriminator()
    gen_net = infogan.Generator()
    rn_net = infogan.RecognitionNetwork()
    shared_net = infogan.SharedNetwork()

    # initialise weights of parameters in all models
    for i in [disc_net, gen_net, rn_net, shared_net]:
        i.to(device)
        i.apply(utils.initialise_weights)

    infogan_obj.train(dataloader=dataloader,
                      discriminator_net=disc_net,
                      generator_net=gen_net,
                      recognition_net=rn_net,
                      shared_net=shared_net)
