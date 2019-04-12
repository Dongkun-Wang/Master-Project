import numpy as np
import random

# 单次训练样本数量
batch_size = 400
# 迭代轮数
epoch = 20000
# 学习率
learn_rate = 5/batch_size


# 定义一个神经网络的类
class Network(object):
    def __init__(self, size):
        self.num_layers = len(size)
        self.sizes = size
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(batch_size, y)
                       for y in size[1:]]


# 定义求导
def derive(y):
        return y*(1-y)


# 定义均方差损失
def loss(y, y_):
    return 0.5*(y-y_) ** 2


# 定义sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 前向传播方法
def forward(x_input, nt):
    layer = []
    layer.append(abs(np.fft.fft(x_input)[:batch_size]))
    for i in range(nt.num_layers-1):
        layer.append(sigmoid((np.dot(layer[i], nt.weights[i]))+nt.biases[i]))
    return layer


# 训练网络
def training(dataset_x, dataset_y):
    # 创建神经网络
    nt = Network([len(dataset_x[0]), 100, 30, len(dataset_y[0])])
    print("随机初始化权值矩阵完成")
    print("\n开始计算迭代误差：")
    num = nt.num_layers
    # 开始多轮迭代训练
    for i in range(epoch):
        # 将训练数据打乱
        c = list(zip(dataset_x, dataset_y))
        random.shuffle(c)
        dataset_x, dataset_y = zip(*c)
        # 开始每一轮训练
        for j in range(0, int(len(dataset_x)/batch_size)):
            layer = forward(dataset_x[batch_size * j:batch_size * (j+1)], nt)
            y = np.array(dataset_y[batch_size * j:batch_size * (j+1)])
        # 损如果失函数小于1，停止训练
        losses = np.mean(loss(layer[-1], y))
        if losses < 0.001:
            print("第%d迭代结束，迭代误差：" % i, losses)
            break
        # 每100轮训练输出损失
        if i % 100 == 0:
            print("第"+str(i)+"轮迭代误差：", losses)
        # 方向更新权值矩阵以及偏置矩阵
        delta = layer[1:]
        delta[num-2] = (layer[num-1]-y) * derive(layer[num-1])
        delta[num-3] = delta[num-2].dot(nt.weights[num-2].T) * derive(layer[num-2])
        delta[num-4] = delta[num-3].dot(nt.weights[num-3].T) * derive(layer[num-3])
        # 反向更新权值矩阵以及偏置矩阵
        for k in range(1, num-1):
            nt.weights[-k] -= learn_rate * layer[-k-1].T.dot(delta[-k])
            nt.biases[-k] -= learn_rate * delta[-k]
    # 函数返回神经网络
    return nt






