import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
# 比较不同size的隐藏层


class CompareLayerSize:
    def __init__(self):
        # set up network
        net_configs = [[1, 3, 0, 1], [1, 10, 0, 1], [1, 30, 0, 1], [1, 100, 0, 1]]
        learning_rate = 0.1
        networks = []
        for net_config in net_configs:
            networks.append(Network(net_config, learning_rate))
        self.networks = networks
        self.l_loss = []
        self.net_configs = net_configs
        pass

    def training(self):
        networks = self.networks
        ## training
        x_train = np.linspace(-np.pi, np.pi, 1000).reshape(1000, -1)
        y_train = np.sin(x_train)

        l_loss = [[], [], [], []]
        for e in range(3000):
            for i in range(len(networks)):
                loss = networks[i].train(x_train, y_train)
                l_loss[i].append(loss)
        self.l_loss = l_loss

    def show_result(self):
        l_loss = self.l_loss
        net_configs = self.net_configs
        x = np.arange(1, len(l_loss[0]) + 1)
        plt.title('loss with different size of 1 hidden layer')
        for i in range(len(l_loss)):
            plt.plot(x[100:], l_loss[i][100:], label=('size of hidden layer = ' + str(net_configs[i][1])))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def testing(self):
        # testing
        networks = self.networks
        net_configs = self.net_configs
        x_test = np.linspace(-np.pi, np.pi, 100).reshape(100, -1)
        y_test = np.sin(x_test)

        for i in range(len(networks)):
            y_predict = networks[i].predict(x_test)
            loss = np.square(y_predict - y_test).sum() / x_test.shape[0]
            print("test loss is ", loss, "with size of hidden layer = ", str(net_configs[i][1]))