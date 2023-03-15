from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
        # print(dist_array)
        indx = np.argsort(dist_array)
        # print(indx)
        k_nei = x[indx][:self.k]
        # print(k_nei)
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

                    # print('y', y)
                    neighbor_df = self.getNeighbors(val[col], y)
                    # print('nd', neighbor_df)
                    diff_out = self.diffusion(neighbor_df)
                    # print('do', diff_out)
                    k_col = self.getNeighbors(val[col], diff_out)
                    temp[row * self.k:(row + 1) * self.k, col] = k_col
            samples = np.concatenate([samples, temp], axis=0)
            np.random.shuffle(samples)
            M = temp.shape[0]
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic


def knnMTD_execute(original_data, num_gen, para):
    knnMTD = kNNMTD(n_obs=num_gen, k=para[0])
    knnMTD_gen = knnMTD.fit(original_data)
    return knnMTD_gen


def main():
    x_train = np.load('D:\Code\program\data\soft_sensor\ori_data.npy')
    # print(x_train)
    # x_vir = np.loadtxt('x_vir')
    # print(x_vir)
    # sample = x_vir[700]
    # for i in range(len(x_train)):
    #     aa = x_train[i] - sample
    #     print(aa)
    #     if aa.any() == 0:
    #         print('xxx')
    #         break
    smote = kNNMTD(n_obs=1500, k=20)

    x_vir = smote.fit(x_train)
    print(x_vir)

    x_all = np.concatenate([x_train, x_vir])
    np.savetxt('x_vir', x_vir)
    p = PCA(n_components=2)
    pca = p.fit_transform(x_all)
    shape = ['+', 'o']
    color_real = ['darkorange', 'red', 'magenta', 'sienna']
    real_i_x = pca[0:len(x_train), 0]
    real_i_y = pca[0:len(x_train), 1]
    gen_i_x = pca[len(x_train):, 0]
    gen_i_y = pca[len(x_train):, 1]
    plt.scatter(gen_i_x, gen_i_y, c='blue', marker=shape[1], s=20,
                label='gen_data')
    plt.scatter(real_i_x, real_i_y, c=color_real[0], marker=shape[0], s=20, label='real_data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
