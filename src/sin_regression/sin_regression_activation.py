import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
# 这个python文件用于比较不同的激活函数


class CompareActivation:
    def __init__(self):
        # 总共有三个不同的神经网络
        # 使用三种不同的激活函数
        net_configs = [[1, 30, 0, 1], [1, 30, 1, 1], [1, 30, 2, 1]]
        self.activation_str = ["Sigmoid", "Tanh", "ReLU"]
        learning_rate = 0.1
        networks = []
        for net_config in net_configs:
            networks.append(Network(net_config, learning_rate))
        self.networks = networks
        self.l_loss = []
        pass

    def training(self):
        # training
        x_train = np.linspace(-np.pi, np.pi, 1000).reshape(1000, -1)
        y_train = np.sin(x_train)

        l_loss = [[], [], []]
        for e in range(3000):
            for i in range(len(self.networks)):
                loss = self.networks[i].train(x_train, y_train)
                l_loss[i].append(loss)
        self.l_loss = l_loss

    def show_result(self):
        l_loss = self.l_loss
        x = np.arange(1, len(l_loss[0]) + 1)
        plt.title('loss coordinated with different activation type')
        for i in range(len(l_loss)):
            label_str = ' activation = ' + self.activation_str[i]
            plt.plot(x[100:], l_loss[i][100:], label=(label_str))

        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def testing(self):
        # testing
        x_test = np.linspace(-np.pi, np.pi, 100).reshape(100, -1)
        y_test = np.sin(x_test)

        networks = self.networks
        activation_str = self.activation_str
        for i in range(len(networks)):
            y_predict = networks[i].predict(x_test)
            loss = np.square(y_predict - y_test).sum() / x_test.shape[0]
            print("test loss is ", loss, "with activation = ", activation_str[i])



