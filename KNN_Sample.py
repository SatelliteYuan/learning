# https://www.cnblogs.com/gezhuangzhuang/p/10765355.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

def ocrHandwriting():
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 我们把图片分成5000张，每张20 x 20
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # 使cells变成一个numpy数组，它的维度为（50， 100， 20， 20）
    x = np.array(cells)

    # 训练数据和测试数据
    train = x[:, :50].reshape(-1, 400).astype(np.float32) # size = (2500, 400)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32) # size = (2500, 400)

    # 为训练集和测试集创建一个label
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    # 初始化KNN，训练数据、测试KNN，k=5
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    # 分类的准确率
    # 比较结果和test_labels
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print(accuracy)


def ocrChar():
    data = np.loadtxt('letter-recognition.data', dtype='float32', delimiter=',', converters={0:lambda ch:ord(ch) - ord('A')})

    # 将数据分成2份，10000个训练，10000个测试
    train, test = np.vsplit(data, 2)

    # 将训练集和测试集分解为数据、label
    # 实际上每一行的第一列是我们的一个字母标记。接下来的 16 个数字是它的不同特征。
    responses, trainData = np.hsplit(train, [1])  # 数据从第二列开始
    labels, testData = np.hsplit(test, [1])  

    # 初始化KNN，训练数据、测试KNN，k=5
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)

    # 分类的准确率
    # 比较结果和test_labels
    correct = np.count_nonzero(result==labels)
    accuracy = correct * 100.0 / result.size
    print(accuracy)    


def saveModel(train, train_labels):
    # 保留数据
    np.savez('knn_data.npz', train=train, train_labels=train_labels)

    # 加载数据
    with np.load('knn_data.npz') as data:
        print(data.files)
        train = data['train']
        train_labels = data['train_labels']    


def pointClassify():
    # 包含25个已知/训练数据的(x,y)值的特征集
    trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

    # 用数字0和1分别标记红色和蓝色
    responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    # 画出红色的点
    red = trainData[responses.ravel() == 0]
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

    # 画出蓝色的点
    blue = trainData[responses.ravel() == 1]
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

    plt.show()


    # newcomer为测试数据
    newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
    plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

    print("result: ", results, "\n")
    print("neighbours: ", neighbours,"\n")
    print("distance: ", dist)

    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')

    plt.show()


if __name__ == "__main__":
    pointClassify()    