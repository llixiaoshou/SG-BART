import csv
# out=[[1,2,3],[1,2,3],[1,4,3]]#锚点，正,负
# with open('output.csv', 'w', newline='') as file:
#     # 创建CSV writer对象
#     writer = csv.writer(file)
#     # 将每个向量写入CSV文件的一行
#     for vector in out:
#         writer.writerow(vector)
# # coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    import pandas as pd

    # 读取CSV文件
    dataframe = pd.read_csv('output.csv')
    dataframe=np.array(dataframe)
    # 查看前几行数据

    # 获取数据形状
    print(dataframe.shape)
    data = dataframe[:,:-1]
    label = dataframe[:,-1]
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    # data, label, n_samples, n_features = get_data()
    # print('data.shape',data.shape)
    # print('label',label)
    # print('label中数字有',len(set(label)),'个不同的数字')
    # print('data有',n_samples,'个样本')
    # print('每个样本',n_features,'维数据')
    # print('Computing t-SNE embedding')
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    # result = tsne.fit_transform(data)
    # print('result.shape',result.shape)
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    # plt.show(fig)
    import numpy as np
    import matplotlib.pyplot as plt

    # 示例向量
    vectors = np.array([[1, 2,0], [3, 4,1], [5, 6,2]])

    # 提取 x 和 y 坐标
    x = vectors[:, 0]
    y = vectors[:, 1]

    # 定义颜色数组
    colors = ['red', 'green', 'blue' ]

    # 绘制散点图，每个向量使用不同的颜色
    plt.scatter(x, y, c=colors)
    plt.title("Vector Scatter Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == '__main__':
    main()




