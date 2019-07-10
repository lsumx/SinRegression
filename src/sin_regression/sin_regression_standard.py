import numpy as np
import matplotlib.pyplot as plt
from util.network_elements import Network


# 最佳版本
class StandardResult:
    def __init__(self):
        # set up network
        print("hint: 假如跑崩了，再跑一次就好了。")
        net_config = [1, 30, 2, 1]
        learning_rate = 0.1
        self.network = Network(net_config, learning_rate)
        self.ll = []
        pass

    def training(self):
        network = self.network
        # training
        x_train = np.linspace(-np.pi, np.pi, 600).reshape(600, -1)
        y_train = np.sin(x_train)

        ll = []
        for e in range(3000):
            loss = network.train(x_train, y_train)
            ll.append(loss)
        self.ll = ll

    def show_result(self):
        ll = self.ll
        x = np.arange(1, len(ll) + 1)
        plt.title('loss with three layer, hidden layer = 30')
        label_str = ' activation = tanh'
        plt.plot(x[100:], ll[100:], label=(label_str))

        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()

    def testing(self):
        network = self.network
        ## testing
        x_test = np.linspace(-np.pi, np.pi, 100).reshape(100, -1)
        y_test = np.sin(x_test)

        y_predit = network.predict(x_test)
        loss = np.square(y_predit - y_test).sum() / x_test.shape[0]
        print("test loss is ", loss, "with layer size = ", 30)
