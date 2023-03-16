import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from scipy.linalg import eigh, svd, qr, solve
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


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

    def reconstruct(self, x_vir_low, x_low, x_train):  # 对潜空间数据进行反向映射并重构
        x = np.vstack((x_vir_low, x_low))
        knn = NearestNeighbors(n_neighbors=self.k + 1).fit(x)  # 寻找knn
        X = knn._fit_X
        ind = knn.kneighbors(X, return_distance=False)[:, 1:]  # xi对应的k近邻坐标
        w = self.barycenter_weights(X, X, ind)  # 权重
        x_vir = np.dot(w[0], x_train[ind - 1])
        return x_vir[0]

    def random_sample(self, x_low, nums, n_components):
        x_min = np.min(x_low, 0)
        x_max = np.max(x_low, 0)
        z = np.random.rand(nums, n_components)
        x_vir_lows = z * (x_max - x_min) + x_min
        return x_vir_lows

    def data_aug(self, samples):
        res = np.zeros((self.num_gen, samples.shape[1]))
        x_low = manifold.LocallyLinearEmbedding(n_neighbors=self.k, n_components=self.n_components,
                                                method='standard').fit_transform(samples)  # 潜空间数据
        x_vir = self.random_sample(x_low, self.num_gen, self.n_components)
        for i in range(self.num_gen):
            a = np.array([x_vir[i]])
            res[i] = self.reconstruct(a, x_low, samples)
        return res


def LLE_execute(original_data, num_gen, para):
    lle = Lle(num_gen=num_gen, n_neighbor=para[0], reg=para[1], n_component=para[2])
    lle_gen = lle.data_aug(original_data)
    return lle_gen
