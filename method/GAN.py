import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import dataloader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm
from torch import manual_seed,cuda


def setup_seed(seed):
    manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(47)
cuda = True if cuda.is_available() else False


class Generator(nn.Module):

    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 32, normalize=False),
            *block(32, 32),
            nn.Linear(32, self.data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


# For training gan
if cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


class GAN:
    def __init__(self, num_gen, num_epoch, lr, batch_size, latent_dim):
        self.num_gen = num_gen
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim

    def train(self, train_data):
        data_dim = train_data.shape[1]
        if cuda:
            netG = Generator(latent_dim=self.latent_dim, data_dim=data_dim).cuda()
            netD = Discriminator(data_dim=data_dim).cuda()
        else:
            netG = Generator(latent_dim=self.latent_dim, data_dim=data_dim)
            netD = Discriminator(data_dim=data_dim)
        dataset = torch.tensor(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        adversarial_loss = torch.nn.BCELoss()
        for epoch in range(self.num_epoch):
            d_losses = 0
            g_losses = 0
            loop = tqdm(data_loader, total=len(data_loader))
            for datas in loop:
                # Adversarial ground truths
                valid = Variable(Tensor(datas.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(datas.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_datas = Variable(datas.type(Tensor))
                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (datas.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_datas = netG(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(netD(gen_datas), valid)

                # print(netD(gen_datas).shape,valid.shape)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(netD(real_datas), valid)
                fake_loss = adversarial_loss(netD(gen_datas.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
                loop.set_description(f'Epoch[{epoch}/{self.num_epoch}]')
                loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
                d_losses += d_loss.item()
                g_losses += g_loss.item()
        return netG, netD

    def generate_data(self, netG):
        if cuda:
            netG = netG.cuda()
        netG.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if cuda:
                fake_data = netG.forward(z.cuda())
            else:
                fake_data = netG.forward(z)
            fake_data = fake_data.cpu().numpy()
        return fake_data

    def fit(self, original_data):
        net_G, net_D = self.train(original_data)
        gen_data = self.generate_data(net_G)
        np.random.shuffle(gen_data)
        gen_data = gen_data[:self.num_gen]
        list_net = [net_G, net_D]
        return list_net, gen_data


def gan_execute(original_data, num_gen, para):
    gan = GAN(num_gen=num_gen, num_epoch=para[0], lr=para[1], batch_size=para[2], latent_dim=para[3])
    list_net, gen_data = gan.fit(original_data)
    return list_net, gen_data
