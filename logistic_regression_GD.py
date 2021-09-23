# -*- coding: utf-8 -*-
'''
this python file validates Logistic Regression model using 1 VS other method

dataset: seed data

attributes: [float], number: 7

label: [int], valueset: [1, 2, 3]

training method: gradient descent
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmod(z):
    """sigmod function

    Parameters
    ----------
    z : [Number or Number List]

    Returns
    -------
    [Number or Number List]
    """    ''''''
    return 1.0/(1.0 + np.exp(-1.0*z))


def dW(X, Y, W):
    """function that caculates dW

    Parameters
    ----------
    m is num_traindata, n is num_feature

    X : [m x n]
        [the matrix contains traindata]
    Y : [m x 1]
        [the matrix contains traindata's label]
    W : [n x 1]
        [the Logistic Regression parameters]

    Returns
    -------
    [n x 1]
        [Update W using the result dW]
    """    ''''''
    result = np.zeros((np.shape(W)[0], 1))  # result is a n x 1 vector
    for i in range(0, np.shape(X)[0]):
        xi = X[i].reshape(-1, 1)
        yi = np.sum(Y[i])
        wi = np.sum(np.dot(W.T, xi))
        pi = sigmod(wi)
        t = (pi - yi)*xi
        result += t
    return result


def param_init(d):
    """initialize parameters W and b

    Parameters
    ----------
    d : [unsigned int]
        [the dimension of W]

    Returns
    -------
    [a d x 1 zero vector W and float zero b]
    """    ''''''
    w = np.zeros((d, 1), dtype="f8")
    return w


def loss(X, Y, W):
    """caculate the target function's loss

    Parameters
    ----------
    m is num_traindata, n is num_feature

    X : [m x n]
        [the matrix contains traindata]
    Y : [m x 1]
        [the matrix contains traindata's label]
    W : [n x 1]
        [the Logistic Regression parameters]

    Returns
    -------
    [float]
    """    ''''''
    result = 0.
    for i in range(0, np.shape(X)[0]):
        xi = X[i]
        yi = np.sum(Y[i])
        wi = np.dot(W.T, xi)
        result += (np.dot(-yi, wi) + np.log(1 + np.exp(wi)))
    return np.sum(result)


def logistic(X, Y, W, theta, max_step):
    """iteratively calculate the best W and b

    Parameters
    ----------
    m is num_traindata, n is num_feature

    X : [m x n]
        [the matrix contains traindata]
    Y : [m x 1]
        [the matrix contains traindata's label]
    W : [n x 1]
        [the Logistic Regression parameters]
    theta : [float]
        [theta is iteration parameter, which controls the speed of iteration]
    max_step : [unsigned int]
        [max_step is the iteration step number]
    ----------
    Return
    ----------
    w : [n x 1] trained parameter vector
    """    ''''''
    w = W
    for step in range(0, max_step):
        dW_old = dW(X, Y, w)
        w = w - theta*dW_old
        if step % 100 == 0 or step == 999:
            print("step {}, w: {}, loss: {}".format(
                step, w, loss(X, Y, w)))
    return w


def predict(test_feature, w):
    """using final w and b to get predict label

    Parameters
    ----------
    test_feature : [m x n]
        [test set]
    w : [n x 1]
        [parameter vector]

    Returns
    -------
    [m x 1]
        [a vector contains 1 or 0, which indicates the label]
    """    ''''''
    test_feature = np.insert(test_feature, np.shape(
        test_feature)[1], values=1, axis=1)
    predict_prob = sigmod(np.dot(test_feature, w))
    predict_prob[predict_prob > 0.5] = 1
    predict_prob[predict_prob < 0.5] = 0
    return predict_prob.astype(int)


if __name__ == "__main__":
    seed_train = pd.read_csv("./seed_data_train.csv")  # train list
    seed_test = pd.read_csv("./seed_data_test.csv")  # test list

    train_data = np.asarray(seed_train)
    train_feature = train_data[:, :-1]  # features of train data, as Xn
    train_label = train_data[:, -1:].astype(int)  # labels of train data, as Y

    test_data = np.asarray(seed_test)
    test_feature = test_data[:, :-1]  # features of test data, as Xn
    test_label = test_data[:, -1:].astype(int)  # labels of test data, as Y

    num_feature = np.shape(train_feature)[1]
    num_traindata = np.shape(train_feature)[0]

    parameter_list = []  # record the tuple[W, b] of each model

    # count the unique label number, which decides the number of classification model
    num_classification = len(np.unique(train_label))
    thetalist = [0.00017, 0.0115, 0.0115]

    for i in range(0, num_classification):
        X = train_feature
        X = np.insert(X, np.shape(X)[1], values=1, axis=1)

        # replace the label, using 1 VS other method
        Y = train_label.copy()
        Y[Y != (i + 1)] = 0
        Y[Y == (i + 1)] = 1

        w = param_init(np.shape(X)[1])  # initialize w and b as 0
        w = logistic(X, Y, w, thetalist[i], 1000)
        parameter_list.append(w)

    for i in range(0, num_classification):
        X = test_feature

        # replace the label, using 1 VS other method
        Y = test_label.copy()
        Y[Y != (i + 1)] = 0
        Y[Y == (i + 1)] = 1

        # find the different value between prediction and X
        predict_result = predict(X, parameter_list[i]) - Y
        right_num = np.sum(predict_result == 0)
        wrong_num = np.sum(predict_result != 0)

        print("classification {}: \nright: {}, wrong: {}, accuracy: {}".format(
            i, right_num, wrong_num, float(right_num)/(right_num + wrong_num)))

    '''
        plot figure using attribute 0 and attribute 1
    '''
    seed_csv = pd.read_csv("./seed_dataset.csv")
    seed_array = np.asarray(seed_csv)

    x1_min, x1_max = seed_array[:, 0].min() - 0.5, seed_array[:, 0].max() + 0.5
    x2_min, x2_max = seed_array[:, 1].min() - 0.5, seed_array[:, 1].max() + 0.5
    h = 0.02
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))

    plt.figure(1, figsize=(6, 5))

    plt.scatter(seed_array[:69, 0], seed_array[:69, 1], marker='*',
                edgecolors="red", label="seed 1")
    plt.scatter(seed_array[69:139, 0], seed_array[69:139, 1], marker='+',
                edgecolors="green", label="seed 2")
    plt.scatter(seed_array[139:, 0], seed_array[139:, 1], marker='o',
                edgecolors="yellow", label="seed 3")
    plt.xlabel("area")
    plt.ylabel("perimeter")
    plt.legend(loc=2)

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title("Logistic Regression - seed classification", fontsize=12)
    plt.xticks(())
    plt.yticks(())
    plt.grid()

    plt.show()
