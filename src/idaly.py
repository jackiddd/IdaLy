import numpy as np
import random
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from scipy.linalg import eigh, svd, qr, solve
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import FLOAT_DTYPES
from model import Gan
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torch import manual_seed,cuda




class GNI:
    def __init__(self, num_gen, mean, variance):
        self.mean = mean
        self.variance = variance
        self.num_gen = num_gen

    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            X_f = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            X_f = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        noise = np.random.normal(loc=self.mean, scale=self.variance,
                                 size=(X_f.shape[0], X_f.shape[1]))
        GNI_gen = X_f + noise
        return GNI_gen
    
class Smote(object):
    def __init__(self, N=100, k=10, r=2):
        self.N = N
        self.k = k
        self.r = r
        self.newindex = 0

    def fit(self, samples):
        T = samples.shape[0]
        numattrs = samples.shape[1]
        self.synthetic = np.zeros((self.N, numattrs))
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(samples)
        repeats = int(self.N / T)
        yu = self.N % T
        if repeats > 0:
            for i in range(T):
                nnarray = neighbors.kneighbors(samples[i].reshape((1, -1)),
                                               return_distance=False)[0][1:]
                self._populate(repeats, i, nnarray, samples)

        samples_shuffle = samples
        np.random.shuffle(samples_shuffle)
        neighbors_1 = NearestNeighbors(n_neighbors=self.k + 1,
                                       algorithm='ball_tree',
                                       p=self.r).fit(samples_shuffle)
        for j in range(yu):
            nnarray_1 = neighbors_1.kneighbors(samples_shuffle[j].reshape((1, -1)),
                                               return_distance=False)[0][1:]
            self._populate(1, j, nnarray_1, samples_shuffle)
        return self.synthetic

    def _populate(self, N, i, nnarray, samples):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            diff = samples[nnarray[nn]] - samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newindex] = samples[i] + gap * diff
            self.newindex += 1
    
class Lle:
    def __init__(self, num_gen, n_neighbor, reg, n_component):
        self.num_gen = num_gen
        self.k = n_neighbor
        self.reg = reg
        self.n_components = n_component

    def barycenter_weights(self, x, y, indices):
        x = check_array(x, dtype=FLOAT_DTYPES)
        y = check_array(y, dtype=FLOAT_DTYPES)
        indices = check_array(indices, dtype=int)

        n_samples, n_neighbors = indices.shape
        assert x.shape[0] == n_samples

        B = np.empty((n_samples, n_neighbors), dtype=x.dtype)
        v = np.ones(n_neighbors, dtype=x.dtype)

        # this might raise a LinalgError if G is singular and has trace
        # zero
        for i, ind in enumerate(indices):
            A = y[ind]
            C = A - x[i]  # broadcasting
            G = np.dot(C, C.T)
            trace = np.trace(G)
            if trace > 0:
                R = self.reg * trace
            else:
                R = self.reg
            G.flat[:: n_neighbors + 1] += R
            w = solve(G, v, sym_pos=True)
            B[i, :] = w / np.sum(w)
        return B

    def reconstruct(self, x_vir_low, x_low, x_train):  
        x = np.vstack((x_vir_low, x_low))
        knn = NearestNeighbors(n_neighbors=self.k + 1).fit(x) 
        X = knn._fit_X
        ind = knn.kneighbors(X, return_distance=False)[:, 1:]  
        w = self.barycenter_weights(X, X, ind)  
        x_vir = np.dot(w[0], x_train[ind - 1])
        return x_vir[0]

    def random_sample(self, x_low, nums, n_components):
        x_min = np.min(x_low, 0)
        x_max = np.max(x_low, 0)
        z = np.random.rand(nums, n_components)
        x_vir_lows = z * (x_max - x_min) + x_min
        return x_vir_lows

    def fit(self, samples):
        res = np.zeros((self.num_gen, samples.shape[1]))
        x_low = manifold.LocallyLinearEmbedding(n_neighbors=self.k, n_components=self.n_components,
                                                method='standard').fit_transform(samples)  
        x_vir = self.random_sample(x_low, self.num_gen, self.n_components)
        for i in range(self.num_gen):
            a = np.array([x_vir[i]])
            res[i] = self.reconstruct(a, x_low, samples)
        return res

class MTD:
    def __init__(self, n_obs=100, random_state=8):
        self.n_obs = n_obs
        self._gen_obs = n_obs * 20
        self.synthetic = None
        np.random.RandomState(random_state)

    def diffusion(self, sample):
        new_sample = []
        m = np.mean(sample)
        thelta = np.std(sample, ddof=1)
        # sample = np.array([i for i in sample if np.abs(i - m) < 3 * thelta])
        n = len(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2
        if u_set == min_val or u_set == max_val:
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample, ddof=1)
        if var == 0:
            a = min_val / 5
            b = max_val * 5
            new_sample = np.random.uniform(a, b, size=self._gen_obs)
        else:
            a = u_set - (skew_l * np.sqrt(-2 * (var / Nl) * np.log(10 ** (-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var / Nu) * np.log(10 ** (-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            while len(new_sample) < self._gen_obs:
                x = np.random.uniform(L, U)
                if x <= u_set:
                    MF = (x - L) / (u_set - L)
                elif x > u_set:
                    MF = (U - x) / (U - u_set)
                elif x < L or x > U:
                    MF = 0
                rs = np.random.uniform(0, 1)
                if MF > rs:
                    new_sample.append(x)
                else:
                    continue
        return np.array(new_sample)

    def fit(self, original_data):
        samples = original_data
        numattrs = samples.shape[1]
        M = 0
        T = samples.shape[0]
        temp = np.zeros((self._gen_obs, numattrs))
        for col in range(numattrs):
            y = samples[:, col]
            diff_out = self.diffusion(y)
            temp[:, col] = diff_out
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic

class kNNMTD:
    def __init__(self, n_obs=100, k=3, random_state=42):
        self.n_obs = n_obs
        self._gen_obs = k * 10
        self.k = k
        self.synthetic = None
        np.random.RandomState(random_state)

    def diffusion(self, sample):
        new_sample = []
        n = len(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2
        if u_set == min_val or u_set == max_val:
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample, ddof=1)
        if var == 0:
            a = min_val / 5
            b = max_val * 5
            new_sample = np.random.uniform(a, b, size=self._gen_obs)
        else:
            a = u_set - (skew_l * np.sqrt(-2 * (var / Nl) * np.log(10 ** (-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var / Nu) * np.log(10 ** (-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            while len(new_sample) < self._gen_obs:
                x = np.random.uniform(L, U)
                if x <= u_set:
                    MF = (x - L) / (u_set - L)
                elif x > u_set:
                    MF = (U - x) / (U - u_set)
                elif x < L or x > U:
                    MF = 0
                rs = np.random.uniform(0, 1)
                if MF > rs:
                    new_sample.append(x)
                else:
                    continue
        return np.array(new_sample)

    def getNeighbors(self, val, x):
        dis_set = []
        for m in x:
            dis = np.abs(m - val)
            dis_set.append(dis)
        dist_array = np.array(dis_set).reshape(1, -1)[0]
        indx = np.argsort(dist_array)
        k_nei = x[indx][:self.k]
        return k_nei

    def fit(self, original_data):
        samples = original_data
        numattrs = samples.shape[1]
        M = 0
        while M < self.n_obs:
            T = samples.shape[0]
            temp = np.zeros((self.k * T, numattrs))
            for row in range(T):
                val = samples[row]
                for col in range(numattrs):
                    y = samples[:, col].reshape(-1, 1)
                    neighbor_df = self.getNeighbors(val[col], y)
                    diff_out = self.diffusion(neighbor_df)
                    k_col = self.getNeighbors(val[col], diff_out)
                    temp[row * self.k:(row + 1) * self.k, col] = k_col
            samples = np.concatenate([samples, temp], axis=0)
            np.random.shuffle(samples)
            M = temp.shape[0]
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic

class Gmm(object):
    def __init__(self, N=50, n_components=2):
        self.N = N
        self.n_components = n_components

    def fit(self, samples):
        gmm = GMM(n_components=self.n_components)
        gmm.fit(samples)
        self.synthetic = gmm.sample(self.N)[0]
        return self.synthetic


    
class GAN:
    def __init__(self, num_gen, num_epoch, lr, batch_size, latent_dim):
        self.num_gen = num_gen
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.cuda = True if cuda.is_available() else False
        self.setup_seed(47)
    
    def setup_seed(self,seed):
        manual_seed(seed)
        cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self, train_data):
        if self.cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        data_dim = train_data.shape[1]
        if self.cuda:
            netG = Gan.Generator(latent_dim=self.latent_dim, data_dim=data_dim).cuda()
            netD = Gan.Discriminator(data_dim=data_dim).cuda()
        else:
            netG = Gan.Generator(latent_dim=self.latent_dim, data_dim=data_dim)
            netD = Gan.Discriminator(data_dim=data_dim)
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
        if self.cuda:
            netG = netG.cuda()
        netG.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if self.cuda:
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
