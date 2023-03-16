from sklearn.mixture import GaussianMixture as GMM
import numpy as np


class Gmm(object):
    def __init__(self, N=50, n_components=2):
        self.N = N
        # GMM组分的个数
        self.n_components = n_components

    def fit(self, samples):
        gmm = GMM(n_components=self.n_components)
        gmm.fit(samples)
        self.synthetic = gmm.sample(self.N)[0]

        return self.synthetic


def gmm_execute(original_data, num_gen, para):
    gmm = Gmm(N=num_gen, n_components=para[0])
    gmm_gen = gmm.fit(original_data)
    return gmm_gen

