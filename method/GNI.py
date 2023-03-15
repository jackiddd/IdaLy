import numpy as np


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


def GNI_execute(original_data, num_gen, para):
    gni = GNI(num_gen=num_gen, mean=para[0], variance=para[1])
    gni_gen = gni.fit(original_data)
    return gni_gen
