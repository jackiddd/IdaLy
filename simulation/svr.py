from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def svr(train_x, train_y, test_x):
    # model = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf', 'sigmoid'), "C": np.logspace(-3, 3, 7),
    #                                         "gamma": np.logspace(-3, 3, 7)})
    model = SVR(kernel='rbf', C=10, gamma=10)
    print('hey')
    model.fit(train_x, train_y)
    # print(model.best_params_, model.best_estimator_)
    pre = model.predict(test_x)
    return model, pre


def svr_simulate(original_data, new_data, test_x):
    print('hey')
    new_x = new_data[:, :-1]
    new_y = new_data[:, -1]
    ori_x = original_data[:, :-1]
    ori_y = original_data[:, -1]
    model_ori, pre_ori = svr(ori_x, ori_y, test_x)
    print('1')
    model_new, pre_new = svr(new_x, new_y, test_x)
    print('done')
    return pre_ori, pre_new


def data_pre():
    x_path = "D:\Code\program\Btrain_input.txt"
    y_path = "D:\Code\program\Btrain_output.txt"
    x_data = rdnumpy(x_path)
    y_data = rdnumpy(y_path)
    con_data = np.concatenate([x_data, y_data], axis=1)
    data_df = pd.DataFrame(data=con_data)
    data_dropna = data_df.dropna(axis=0, how='any')
    data_out = np.array(data_dropna)
    np.save('D:/Code/program/data/soft_sensor/ori_data.npy', data_out)
    return data_out


def rdnumpy(txtname):
    with open(txtname, "r") as f:  # 打开文件
        line = f.readlines()
        lines = len(line)  # 行数
        for l in line:
            le = l.strip('\n').split(' ')
            columns = len(le)  # 列
        A = np.zeros((lines, columns), dtype=float)
        A_row = 0
        for lin in line:
            list = lin.strip('\n').split(' ')
            A[A_row:] = list[0:columns]
            A_row += 1
        return A


def main():
    original_data = np.load('D:\Code\program\data\soft_sensor\ori_data.npy')
    gen_data = np.load('D:\Code\program\data_gen\soft_sensor\SMOTE_DA_20221107-200203.npy')
    data_ori = original_data
    np.random.shuffle(data_ori)
    original_train = data_ori[:int(0.7 * len(data_ori)), :]
    test_data = data_ori[int(0.7 * len(data_ori)):, :]
    all_train = np.concatenate((original_train, gen_data), axis=0)
    np.random.shuffle(all_train)
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]
    pre_ori, pre_new, model_ori, model_new = svr_simulate(original_data=original_train,
                                                          new_data=all_train, test_x=test_x)
    # train_original = np.load('D:/Code/program/data/soft_sensor/train_original.npy')
    # test_ = np.load('D:/Code/program/data/soft_sensor/test_original.npy')
    # data_gen = np.load('D:/Code/program/data_gen/soft_sensor/GAN_data1.npy')
    # all_train = np.concatenate([train_original, data_gen], axis=0)
    # print(all_train.shape)
    # sim_names = ['original', 'plus_gen']
    # test_x = test_[:, :-1]
    # test_y = test_[:, -1]
    t = list(range(len(test_x)))
    # y_pred_set = []
    # for sim_name in sim_names:
    #     if sim_name == 'original':
    #         train_x = train_original[:, :-1]
    #         train_y = train_original[:, -1]
    #     elif sim_name == 'plus_gen':
    #         train_x = all_train[:, :-1]
    #         train_y = all_train[:, -1]
    #     model, y_pred = svr_train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    #     y_pred_set.append(y_pred)
    #     joblib.dump(model, 'model_svr_' + sim_name + '.pkl')
    # plt.figure(figsize=[5, 14])
    ax1 = plt.subplot(211)
    plt.plot(t, pre_new, c='r')
    plt.plot(t, test_y, c='b')
    ax2 = plt.subplot(212)
    plt.plot(t, pre_ori, c='r')
    plt.plot(t, test_y, c='b')
    plt.show()
    # all_data = data_pre()
    # # data_gen = np.load('D:/Code/program/data_gen/soft_sensor/GAN_data1.npy')
    # # all_data = np.concatenate([all_data, data_gen], axis=0)
    # data_x = all_data[:, :-1]
    # data_y = all_data[:, -1]
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
    #
    # model_svr, y_pred = svr_train(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test)
    # x = list(range(len(y_test)))
    print(r2_score(test_y, pre_ori))
    print(r2_score(test_y, pre_new))
    # plt.plot(x, y_test, c='b')
    # plt.plot(x, y_pred, c='r')
    # plt.show()


if __name__ == '__main__':
    main()
