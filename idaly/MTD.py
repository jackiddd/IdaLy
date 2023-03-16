from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm


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
        # clf = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.2)
        # clf.fit(original_data)
        # labels = clf.predict(original_data)
        # samples = original_data[labels>0]
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


def MTD_execute(original_data, num_gen, para):
    mtd = MTD(n_obs=num_gen)
    MTD_gen = mtd.fit(original_data)
    return MTD_gen
