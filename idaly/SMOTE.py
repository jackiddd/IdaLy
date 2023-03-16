import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote(object):
    def __init__(self, N=100, k=10, r=2):
        # 初始化self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        # self.r是距离决定因子，取2的话用的是欧式距离
        self.r = r
        # self.newindex用于记录SMOTE算法已合成的样本个数
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
                # 调用kneighbors方法搜索k近邻
                nnarray = neighbors.kneighbors(samples[i].reshape((1, -1)),
                                               return_distance=False)[0][1:]

                # 把N,i,nnarray输入样本合成函数self._populate
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
        # 按照倍数N做循环
        for j in range(N):
            # attrs用于保存合成样本的特征
            # attrs = []
            # 随机抽取1～k之间的一个整数，即选择k近邻中的一个样本用于合成数据
            nn = random.randint(0, self.k - 1)
            # 计算差值
            diff = samples[nnarray[nn]] - samples[i]
            # 随机生成一个0～1之间的数
            gap = random.uniform(0, 1)
            # 合成的新样本放入数组self.synthetic
            self.synthetic[self.newindex] = samples[i] + gap * diff
            # self.newindex加1， 表示已合成的样本又多了1个
            self.newindex += 1


def smote_execute(original_data, num_gen, para):
    smote = Smote(N=num_gen, k=para[0], r=2)
    smote_gen = smote.fit(original_data)
    return smote_gen
