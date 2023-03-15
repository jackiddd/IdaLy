from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from random import shuffle


def svc(train_x, train_y, test_x):
    # model = GridSearchCV(SVC(), param_grid={"kernel": ("linear", 'rbf', 'sigmoid'), "C": np.logspace(-3, 3, 7),
    #                                         "gamma": np.logspace(-3, 3, 7)})
    model = SVC(kernel='rbf', C=100, gamma=0.1)
    model.fit(train_x, train_y)
    # print(model.best_estimator_,model.best_params_)
    pre = model.predict(test_x)
    return model, pre


def svc_simulate(original_data, new_data, test_x):
    new_x = new_data[:, :-1]
    new_y = new_data[:, -1]
    print(new_data.shape)
    ori_x = original_data[:, :-1]
    ori_y = original_data[:, -1]
    model_ori, pre_ori = svc(ori_x, ori_y, test_x)
    print('1')
    model_new, pre_new = svc(new_x, new_y, test_x)
    print('done')
    return pre_ori, pre_new


def data_pre():
    path = 'D:/Code/program/data_gen/fault_diagnosis/4/'
    x_ori = np.load('D:/Code/program/data/fault_diagnosis/fault_norm_3.npy')
    plus_gen = True
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    gen_dir = os.listdir(path='D:/Code/program/data_gen/fault_diagnosis/4/')
    fault_type_list = list(range(4))
    for fault_i in fault_type_list:
        x_i = x_ori[fault_i * 800:fault_i * 800 + 800, ]
        np.random.shuffle(x_i)
        x_i_train = x_i[:600, ]
        y_i_train = np.full(600, fault_i, dtype=int)
        if plus_gen:
            gen_i = np.load(os.path.join(path, gen_dir[fault_i]))[3000:4000, ]
            x_i_train = np.concatenate([x_i_train, gen_i])
            np.random.shuffle(x_i_train)
            y_i_train = np.full(1600, fault_i, dtype=int)
        x_i_test = x_i[600:, ]
        y_i_test = np.full(200, fault_i, dtype=int)
        x_train.append(x_i_train)
        x_test.append(x_i_test)
        y_train.append(y_i_train)
        y_test.append(y_i_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.reshape(x_train, (-1, 52))
    x_test = np.reshape(x_test, (-1, 52))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    train = np.concatenate([x_train, y_train], axis=1)
    np.random.shuffle(train)
    x_train = train[:, :52]
    y_train = train[:, 52]
    return x_train, x_test, y_train, y_test


def main():
    train_x, test_x, train_y, test_y = data_pre()
    model = svc(train_x, train_y,test_x)
    y_pre = model.predict(test_x)
    np.save('y_pre_plus_gen_1.npy', y_pre)
    np.save('y_real.npy', test_y)
    print(y_pre,test_y,model.score(test_x,test_y))



if __name__ == '__main__':
    main()
