from sin_regression.sin_regression_activation import CompareActivation
from sin_regression.sin_regression_learning_rate import CompareLearningRate
from sin_regression.sin_regression_num_of_hidden import CompareDifferentLayerNum
from sin_regression.sin_regression_size_of_hidden import CompareLayerSize
from sin_regression.sin_regression_standard import StandardResult


if __name__ == '__main__':
    # # 比较不同的激活函数
    # print("========================================================================================")
    # print("=======================Compare Different Activation Function============================")
    # compare = CompareActivation()
    # compare.training()
    # compare.show_result()
    # compare.testing()
    # print("========================================================================================")
    #
    # # 比较不同学习速率
    # print("========================================================================================")
    # print("===========================Compare Different Learning Rate==============================")
    # compare = CompareLearningRate()
    # compare.training()
    # compare.show_result()
    # compare.testing()
    # print("========================================================================================")

    # # 比较不同隐藏层数量
    # print("========================================================================================")
    # print("===========================Compare Different Layer Number===============================")
    # compare = CompareDifferentLayerNum()
    # compare.training()
    # compare.show_result()
    # compare.testing()
    # print("========================================================================================")

    # # 比较不同隐藏层大小
    # print("========================================================================================")
    # print("===========================Compare Different Layer Size=================================")
    # compare = CompareLayerSize()
    # compare.training()
    # compare.show_result()
    # compare.testing()
    # print("========================================================================================")

    # 比较最佳版本
    print("========================================================================================")
    print("=====================================Standard Result====================================")
    compare = StandardResult()
    compare.training()
    compare.show_result()
    compare.testing()
    print("========================================================================================")
