import numpy as np


def derive(x):                                                 # 定义求导
        return x*(1-x)


def sigmoid(x):                                                # 定义sigmoid函数
    return 1.0 / (1.0 + np.exp(-x))


def training(dataset_x , dataset_y):                              # a= 500 b =10
    a = len(dataset_x[0])
    b = len(dataset_y[0])
    x = np.array(dataset_x) * np.array(dataset_x)
    y = np.array(dataset_y)
    w1 = 2 * np.random.random((a, 200)) - 1                        # 生成第一层权值矩阵
    w2 = 2 * np.random.random((200, 80)) - 1                       # 生成第二层权值矩阵
    w3 = 2 * np.random.random((80, b)) - 1                        # 生成第三层权值矩阵
    syn = np.array([w1, w2, w3])                                  # 合并成多维矩阵
    print("随机初始化权值矩阵")
    for i in range(3):
        print('\n第'+str(i)+'层初始权值矩阵:\n', syn[i])
    l0 = x
    print("\n迭代误差：")
    for j in range(1000):
        l1 = sigmoid(np.dot(l0, syn[0]))
        l2 = sigmoid(np.dot(l1, syn[1]))
        l3 = sigmoid(np.dot(l2, syn[2]))
        l3_error = y - l3
        if (j % 100) == 0:
            print("第"+str(j)+"轮迭代误差：", end="")          # 循环打印迭代误差
            print(str(np.mean(abs(l3_error))))
        l3_delta = l3_error * derive(l3)
        l2_error = l3_delta.dot(syn[2].T)
        l2_delta = l2_error * derive(l2)
        l1_error = l2_delta.dot(syn[1].T)
        l1_delta = l1_error * derive(l1)
        syn[2] += l2.T.dot(l3_delta)
        syn[1] += l1.T.dot(l2_delta)
        syn[0] += l0.T.dot(l1_delta)
    # 打印训练结果
    print("\n训练后的输出矩阵是：\n"+str(l3))
    for i in range(3):
        print("\n第"+str(i)+"层训练后权值矩阵:\n", syn[i])
    return syn                                               # 函数返回三个权值矩阵


def forward(x_input, syn):                                   # 测试集输出
    l0 = x_input
    l1 = sigmoid(np.dot(l0, syn[0]))
    l2 = sigmoid(np.dot(l1, syn[1]))
    l3 = sigmoid(np.dot(l2, syn[2]))
    return l3

'''
        y_delta = (layer[3]-y) * derive(layer[3])
        b_error = y_delta.dot(nt.weights[2].T)
        b_delta = b_error * derive(layer[2])
        a_error = b_delta.dot(nt.weights[1].T)
        a_delta = a_error * derive(layer[1])
        nt.weights[2] -= layer[2].T.dot(y_delta)
        nt.weights[1] -= layer[1].T.dot(b_delta)
        nt.weights[0] -= layer[0].T.dot(a_delta)
'''
delta = layer[1:]
delta[2] = (layer[3] - y) * derive(layer[3])
b_error = delta[2].dot(nt.weights[2].T)
delta[1] = b_error * derive(layer[2])
a_error = delta[1].dot(nt.weights[1].T)
delta[0] = a_error * derive(layer[1])