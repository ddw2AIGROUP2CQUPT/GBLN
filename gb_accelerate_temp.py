import ast
import math
import time
import pandas as pd
import random
import numpy
import scipy.io
import matplotlib.pyplot as plt
# from sklearn.cluster import k_means
from config import args
import numpy as np
random.seed(42)

# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球

# 判断粒球的标签和纯度
def get_label_and_purity(gb):
    # 分离不同标签数据
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 0] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        if(len(gb_label_temp)!=0): 
            max_label = max(gb_label_temp.keys())
            purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数
            label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
        else:
            label = -1
            purity = 1
    # print(label)
    # 标签、纯度
    return label, purity


# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]  # 第2列往后的所有数据
    # print("data no label\n",data_no_label)
    center = data_no_label.mean(axis=0)  # 同一列在所有行之间求平均
    # print("center:\n", center)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius


# 绘制粒球
def gb_plot(gb_dict, plt_type=0):
    color = {0: 'r', 1: 'k', 2: 'b', 3: 'y', 4: 'g', 5: 'c', 6: 'm', 7: 'peru', 8: 'pink', 9: 'gold'}
    # 图像宽高与XY轴范围成比例，绘制粒球才是正圆
    plt.figure(figsize=(5, 4))  # 图像宽高
    # plt.axis([-0, 0.08, 0, 0.3])  # 设置x轴的范围为[-1.2, 1.2]，y轴的范围为[-1, 1]
    for key in gb_dict.keys():
        # print(key)
        gb = gb_dict[key][0]
        # for gb in gb_list:
        # print(granular_ball)
        label, p = get_label_and_purity(gb)
        k = len(np.unique(gb[:, 0], axis=0))
        center, radius = calculate_center_and_radius(gb)
        # 绘制所有点
        for i in range(0, k):
            data0 = gb[gb[:, 0] == i]
            plt.plot(data0[:, 1], data0[:, 2], '.', color=color[i], markersize=5)

        if plt_type == 0 or plt_type == 1:  # 绘制粒球
            theta = numpy.arange(0, 2 * numpy.pi, 0.01)
            x = center[0] + radius * numpy.cos(theta)
            y = center[1] + radius * numpy.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])  # 绘制粒球中心
    plt.show()


def splits(purity, gb_dict):
    gb_len = 1
    # print("分裂前的gbdict\n", gb_dict)  # {球心：[[bs,65],[1,bs个样本到球心距离]]}
    while True:
        ball_number_1 = len(gb_dict)
        # print("ball-number-a\n", ball_number_1)
        gb_dict_single = gb_dict.copy()  # 复制一个临时list，接下来再遍历取值
        for i in range(0, gb_len):  # gb_len为1为啥要写循环 ？？？？
            gb_single = {}
            # 取字典数据，包括键值
            gb_dict_temp = gb_dict_single.popitem()
            # print("取字典数据\n", gb_dict_temp[1])  # array：当前球内样本向量[球个数，65]+[1,各样本距离所在球心的距离]
            gb_single[gb_dict_temp[0]] = gb_dict_temp[1]
            # print("gb_single:", gb_single.keys())  # 当前球心

            # 取出value:粒球数据
            gb = gb_dict_temp[1][0]
            # print("粒球数据\n", gb)  # 当前球内样本向量[球个数，65]

            # 判断纯度是否满足要求，不满足则继续划分
            label, p = get_label_and_purity(gb)
            if p < purity:  # 当前纯度小于设定纯度时，停止划分，更新字典，进行下一步
                # print("gb_single\n", gb_single)
                gb_dict_new = splits_ball(gb_single).copy()
                # print("gbdict_new\n", gb_dict_new)  # {球1：[[球1内样本数，64],[1,各样本到中心球的距离]];.....}
                gb_dict.update(gb_dict_new)  # 如果被更新的字典中己包含对应的键值对，那么原 value 会被覆盖；如果被更新的字典中不包含对应的键值对，则该键值对被添加进去
            else:
                continue
        # print("gb_dict:", gb_dict.keys())
        gb_len = len(gb_dict)
        ball_number_2 = len(gb_dict)
        # print("ball_number_b\n", ball_number_2)
        # 粒球数和上一次划分的粒球数一样，即不再变化
        if ball_number_1 == ball_number_2:
            break

    # 绘制粒球
    epoch=0
    i=0
    return gb_dict


# 计算距离
def calculate_distances(data, p):
    return ((data - p) ** 2).sum(axis=0) ** 0.5


def splits_ball(gb_dict):
    # {center: [gb, distances]}
    center = []
    distances_other_class = []  # 粒球到异类点的距离
    balls = []  # 聚类后的label
    gb_dis_class = []  # 不同标签数据的距离
    center_other_class = []
    center_distances = []  # 新距离
    ball_list = {}  # 最后要返回的字典，键：中心点，值：粒球 + 到中心的距离
    distances_other = []
    distances_other_temp = []

    centers_dict = []  # 中心list
    gbs_dict = []  # 粒球数据list
    distances_dict = []  # 距离list

    # 取出字典中的数据:center,gb,distances
    # 取字典数据，包括键值
    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):
        center.append(float(center_split))
    center = np.array(center)  # 转为array
    centers_dict.append(center)  # 老中心加入中心list
    gb = gb_dict_temp[1][0]  # 取出粒球数据
    distances = gb_dict_temp[1][1]  # 取出到老中心的距离
    # print('center:', center)
    # print('gb:', gb)
    # print('distances:', distances)

    # 分离不同标签数据的距离
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print(len_label)
    for label in len_label.tolist():
        # 分离不同标签距离
        gb_dis_temp = []
        for i in range(0, len(distances)):
            if gb[i, 0] == label:
                gb_dis_temp.append(distances[i])
        if len(gb_dis_temp) > 0:
            gb_dis_class.append(gb_dis_temp)

    # 取新中心
    for i in range(0, len(gb_dis_class)):
        # print('gb_dis_class_i:', gb_dis_class[i])

        # 最远异类点
        # center_other_temp = gb[distances.index(max(gb_dis_class[i]))]

        # 随机异类点
        ran = random.randint(0, len(gb_dis_class[i]) - 1)
        center_other_temp = gb[distances.index(gb_dis_class[i][ran])]

        if center[0] != center_other_temp[0]:
            center_other_class.append(center_other_temp)
        # print(distances.index(max(distances)))
    # print('center_other_class:', center_other_class)
    centers_dict.extend(center_other_class)
    # print('centers_dict:', centers_dict)

    distances_other_class.append(distances)
    # 计算到每个新中心的距离
    for center_other in center_other_class:
        balls = []  # 聚类后的label
        distances_other = []
        for feature in gb:
            # 欧拉距离
            distances_other.append(calculate_distances(feature[1:], center_other[1:]))
        # 新中心list
        # distances_dict.append(distances_other)
        distances_other_temp.append(distances_other)  # 临时存放到每个新中心的距离
        distances_other_class.append(distances_other)
    # print('distances_other_class:', len(distances_other_temp))

    # 某一个数据到原中心和新中心的距离，取最小以分类
    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])
        # print('distances_temp:', distances_temp)
        classification = distances_temp.index(min(distances_temp))  # 0:老中心；1,2...：新中心
        balls.append(classification)
    # 聚类情况
    balls_array = np.array(balls)
    # print("聚类情况：", balls_array)

    # 根据聚类情况，分配数据
    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])
    # print('gbs_dict:', len(gbs_dict))

    # 分配新距离
    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])
    # print('distances_dict:', distances_dict)
    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1
    # print('distances_dict:', distances_dict)

    # 打包成字典
    for i in range(len(centers_dict)):
        gb_dict_key = str(centers_dict[i][0])
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(centers_dict[i][j])
        gb_dict_value = [gbs_dict[i], distances_dict[i]]  # 粒球 + 到中心的距离
        ball_list[gb_dict_key] = gb_dict_value

    # print('ball_list:', ball_list.keys())
    return ball_list


def main(data, pur):  #+pur

    # data [bs,1+64]
    start = time.time()
    times = 0
    num_gb = 0
    for i in range(0, 6):
        

        # print('开始时间', start)
        # print("进入粒球\n", data)
        # data[data[:, 0] == -1, 0] = 0
        # print("进入粒球第一步\n", data.shape)  # [bs,1+64]  no change
        # data = data[:, 0:]
        # print("进入粒球第2步\n", data.shape)  # [bs,1+64]  no change
        # 数组去重
        # data = numpy.unique(data, axis=0)  # 去除data中的重复行 no change
        # print("数组去重\n", data)
        # print("去重后，shape:", data.shape)  # [bs,1+64]  按首列标签顺序排列，其余不变

        data_temp = []
        data_list = data.tolist()
        data = []
        for data_single in data_list:
            if data_single[1:] not in data_temp:
                data_temp.append(data_single[1:])
                data.append(data_single)
        data = np.array(data)
        # print("single_list\n", data)   # data no change

        purity = pur   # ★★★★

        # 初始随机中心
        center_init = data[random.randint(0, len(data)-1), :]  # 任选一行，也就是某个样本作为初始中心
        # center_init = data[:, 1:3].mean(axis=0)
        # print("初始中心\n", center_init)  # 第一维是标签，后64是向量

        distance_init = []
        for feature in data:
            # print("单个样本\n", feature)  # 输入data中的单个样本（shape：第一维是标签，后64是向量）
            # 初始中心距离
            distance_init.append(calculate_distances(feature[1:], center_init[1:]))  # 计算所有样本距离初始中心（后64维向量）的欧拉距离：差的平方和的开方
        # print('distance_init:', len(distance_init))

        # 封装成字典
        gb_dict = {}
        gb_dict_key = str(center_init.tolist()[0])
        for j in range(1, len(center_init)):
            gb_dict_key += '_' + str(center_init.tolist()[j])
        # print("key:\n", gb_dict_key)   # 初始中心所有维度构成的字符串
        gb_dict_value = [data, distance_init]  # 所有样本[bs,65]+该样本到中心的距离[1,bs]
        gb_dict[gb_dict_key] = gb_dict_value
        # print("字典\n", gb_dict)

        # 分类划分
        gb_dict = splits(purity=purity, gb_dict=gb_dict)   # ？？？ 返回 {球1：[[球1内样本数，64],[1,各样本到中心球的距离]];.....}
        num_gb += len(gb_dict)
        # print('粒球数量：', gb_dict.keys())
        end = time.time()
        # print('耗费时间', round((end - start) * 1000, 0))
        times = (end - start) + times
        break

    centers = []
    numbers = []
    radius = []
    # print('总平均耗时：%s' % (round(times / 3 * 1000, 0)))
    index = []
    result = []
    for i in gb_dict.keys():  # 遍历每个球
        a = list(calculate_center_and_radius(gb_dict[i][0])[0])  # a 每个球center [1,64]
        # print("a:\n",a)
        radius1 = calculate_center_and_radius(gb_dict[i][0])[1]  # a 每个半径
        # print("abanjing\n", radius1)
        lab, p = get_label_and_purity(gb_dict[i][0])  # 获取每个球标签、纯度
        a.insert(0, lab)  # 下标为0的位置（首位）插入球标签
        # print("a+label\n", a)  # [1,1(label)+64]
        centers.append(a)
        radius.append(radius1)
        result.append(gb_dict[i][0])
        # print("result:\n", gb_dict[i][0])  # [1,1(ball_label)+64]

        index1 = []
        for j in gb_dict[i][0]:
            index1.append(j[-1])
            # print("index1\n", index1)  # 球心64维向量最后一维
        numbers.append(len(gb_dict[i][-1]))  # 球内部样本数
        # print("ball_samples:\n", len(gb_dict[i][-1]))
        index.append(index1)
    # print("centers\n", centers)
    # print("results\n", result)
    # print(time.time() - start)
    return numbers, result, centers, radius

    """
    numbers : 每个球内部样本数
    result: [ball_samples,ball_label+64维 样本向量]
    centers: [ball_numbers,ball_label+64维 球中心向量]
    radius: [ball_numbers,每个球半径]
    """


if __name__ == '__main__':
    main()
