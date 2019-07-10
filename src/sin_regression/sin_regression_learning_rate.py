import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network
# 这个类比较不同的学习速率


class CompareLearningRate:
    def __init__(self):
        # set up network
        net_config = [1, 30, 0, 1]
        l_learning_rate = [0.01, 0.03, 0.1, 0.3]
        networks = []
        for learning_rate in l_learning_rate:
            networks.append(Network(net_config, learning_rate))

        self.l_learning_rate = l_learning_rate
        self.activation_str = ["Sigmoid", "Tanh", "ReLU"]
        self.networks = networks
        self.l_loss = []
        pass

    def training(self):
        l_learning_rate = self.l_learning_rate
        networks = self.networks
        # training
        x_train = np.linspace(-np.pi, np.pi, 1000).reshape(1000, -1)
        y_train = np.sin(x_train)

        l_loss = [[], [], [], []]
        for e in range(3000):
            for i in range(len(l_learning_rate)):
                loss = networks[i].train(x_train, y_train)
                l_loss[i].append(loss)
        self.l_loss = l_loss

    def show_result(self):
        l_loss = self.l_loss
        l_learning_rate = self.l_learning_rate
        x = np.arange(1, len(l_loss[0]) + 1)
        plt.title('loss with diffenrent learning rate')
        for i in range(len(l_loss)):
            label_str = ' learning rate = ' + str(l_learning_rate[i])
            plt.plot(x[100:], l_loss[i][100:], label=(label_str))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def testing(self):
        networks = self.networks
        l_learning_rate = self.l_learning_rate
        # testing
        x_test = np.linspace(-np.pi, np.pi, 100).reshape(100, -1)
        y_test = np.sin(x_test)

        for i in range(len(networks)):
            y_predict = networks[i].predict(x_test)
            loss = np.square(y_predict - y_test).sum() / x_test.shape[0]
            print("test loss is ", loss, "with learning rate = ", l_learning_rate[i])