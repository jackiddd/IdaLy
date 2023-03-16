from sklearn.svm import SVC,SVR

def svc(train_x, train_y, test_x):
    model = SVC(kernel='rbf', C=100, gamma=0.1)
    model.fit(train_x, train_y)
    pre = model.predict(test_x)
    return model, pre


def svc_simulate(original_data, new_data, test_x):
    new_x = new_data[:, :-1]
    new_y = new_data[:, -1]
    ori_x = original_data[:, :-1]
    ori_y = original_data[:, -1]
    model_ori, pre_ori = svc(ori_x, ori_y, test_x)
    model_new, pre_new = svc(new_x, new_y, test_x)
    return pre_ori, pre_new

def svr(train_x, train_y, test_x):
    model = SVR(kernel='rbf', C=10, gamma=10)
    model.fit(train_x, train_y)
    pre = model.predict(test_x)
    return model, pre


def svr_simulate(original_data, new_data, test_x):
    new_x = new_data[:, :-1]
    new_y = new_data[:, -1]
    ori_x = original_data[:, :-1]
    ori_y = original_data[:, -1]
    model_ori, pre_ori = svr(ori_x, ori_y, test_x)
    model_new, pre_new = svr(new_x, new_y, test_x)
    return pre_ori, pre_new