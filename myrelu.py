"""
GBNR class
Author 'JerryZ'
"""
import csv
# from pandocfilters import Math
from scipy import stats
import torch
import numpy as np
from config import args
# 类需要继承Function类，此处forward和backward都是静态方法
import gb_accelerate_temp as new_GBNR
import random
import time
import os
random.seed(42)

global total_num_noisy
global total_num_sample
total_num_noisy = 0
total_num_sample = 0


def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5
save_pth = "%d_%s_%s_noise%.1f_pur(%.2f,%.2f)_bs%d_ratio%.2f_%s_GB-%s_%s" % (args.num,args.model,args.dataset,args.noisy_pro,args.purity,args.reclust_pur,args.BATCH_SIZE,args.retention_ratio,args.noisy_mode,args.GB_mode,args.noisy_mode)  # 定义保存路径名
if not os.path.isdir(os.path.join('log',save_pth)):
    os.makedirs(os.path.join('log',save_pth))
fw = open(os.path.join('log',save_pth,'g_ball.log'),'w')
def fw_write(str):
    fw.write(str)
    fw.flush()
def ball_statistics(noisy, sample):
    global total_num_sample
    global total_num_noisy
    total_num_noisy += noisy
    total_num_sample += sample
    fw_write("\nsample noisy rate(cumulative): {:.4f}%".format(total_num_noisy/total_num_sample*100.))
    fw_write("\n")
    fw_write("----"*20)
class GBNR(torch.autograd.Function):
    @staticmethod
    def forward(self, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        # t = time.time()
        
        ############## input_.shape :  [bs,68(pur+ori+index+now_label+64维向量)]
        self.batch_size = input_.size(0)
        input_main = input_[:, 3:]      # noise_label+64 [bs,65]
        self.input = input_[:, 4:]      # backward中使用
        self.nlabel = input_[:,3:4]     # noise_label
        self.res = input_[:, 1:2]       # 样本原标签
        self.index = input_[:, 2:3]     # 样本下标

        pur = input_[:, 0].numpy().tolist()[0]  # 从第0维取出纯度
        t1 = time.time()
        numbers, balls, result, _ = new_GBNR.main(input_main, pur)  # 加了pur
        # print("liqiu",time.time()-t1)
        """
            numbers:    每个球内部样本数
            balls:      [ball_samples*array([球内样本数*[ball_label+64维样本向量]])]
            result:     [ball_numbers*list[ball_label+64维球中心向量]]
            radius:     [ball_numbers,每个球半径]
        """
        ######## 用于存放合格球(球内样本数、样本特征、球的中心向量)
        numbers_qualified = []
        balls_qualified = []
        centers_qualified = []

        ######## 用于存放需再次聚类的单样本
        sample_for_recluster = []
        index = 0               # 计数
        for ball in balls:      # 取出每个球里面的样本  array([球内样本数*[ball_label+样本特征64维]])
            if args.noisy_pro != 0:      # 若噪声为0 则不需要删除单样本
                if numbers[index] < 2 :  # 将所有单样本进行再次聚类 
                    for sample in ball:
                        sample_for_recluster.append(sample)
                else:
                    balls_qualified.append(ball)
                    numbers_qualified.append(numbers[index])
                    centers_qualified.append(result[index])
            else:
                balls_qualified.append(ball)
                numbers_qualified.append(numbers[index])
                centers_qualified.append(result[index])
            index += 1
        # print("current purity: ", pur,"first clustering has ball number: ", len(balls), "single ball number: ", len(sample_for_recluster))
        # fw_write("\ncurrent purity: {:.2%} first clustering has ball number: {} single ball number: {}".format(pur, len(balls), len(sample_for_recluster)))
        ######## 二次聚类单样本 纯度降低  
        re_pur = args.reclust_pur
        if len(sample_for_recluster) != 0 and args.noisy_pro != 0:  # 没有单样本点 或者 噪声为0的时候不需要聚两次

            numbers, balls, result, _ = new_GBNR.main(torch.Tensor(np.array(sample_for_recluster)), re_pur)
            index = 0
            for ball in balls:
                if args.noisy_pro != 0: 
                    if numbers[index] >= 2:      # 丢掉数量1的球  ---->有噪声才把单样本球丢弃
                        balls_qualified.append(ball)
                        numbers_qualified.append(numbers[index])
                        centers_qualified.append(result[index])
                else:
                    if numbers[index] >= 1:      # 丢掉数量0的球  ---->无噪声不把单样本球丢弃
                        balls_qualified.append(ball)
                        numbers_qualified.append(numbers[index])
                        centers_qualified.append(result[index])
                index += 1

            # print("No.",re,"current re-purity: ", re_pur, "after re-clustering single ball number: ", len(sample_for_recluster))
            # fw_write("\ncurrent re-purity: {:.2%} after re-clustering single ball number: {}".format(re_pur,len(sample_for_recluster)))

        ######## 依据各球内样本数对number、center、balls进行重新排序(冒泡排序): 从大到小
        
        # t2 = time.time()
        for i in range(1, len(numbers_qualified)):
            for j in range(len(numbers_qualified)-i):
                if numbers_qualified[j] < numbers_qualified[j+1]:
                    numbers_qualified[j], numbers_qualified[j+1] = numbers_qualified[j+1], numbers_qualified[j]
                    balls_qualified[j], balls_qualified[j+1] = balls_qualified[j+1], balls_qualified[j]
                    centers_qualified[j], centers_qualified[j+1] = centers_qualified[j+1], centers_qualified[j]
        # print("paixu haoshi",time.time()-t2)


        ######## 供backward方法调用
        self.balls = balls_qualified        # 各球内样本
        self.numbers = numbers_qualified    # 各球内样本数
        
        ######## 保存一定比例的球内部的样本    ？？？？？这里的超参还需要讨论
        target = []
        data = []
        for k, i in enumerate(centers_qualified):   # i:[ball_label, 64维球心]
            if k < int(args.retention_ratio * len(numbers_qualified)):  
                data.append(i[1:])                      # 球中心向量64
                target.extend(np.array([i[0]]))         # ball—label
            else:
                break

        self.data = np.array(data)  # [ball_number, 64维球心]
        # fw_write("\nremain ball num:{}".format(len(data)))
        # print("zongshi",time.time()-t)
        ######## 下面为统计球标签噪声
        ball_noise = 0
        ballsample_total = 0
        ball_sample_noise = 0
        for m, i in enumerate(numbers_qualified):
            if m < int(args.retention_ratio * len(numbers_qualified)):
                true_label_lis = []  # 用于存放每个球的内部样本的原始标签
                ballsample_total += len(balls_qualified[m])
                max_label_lis = []
                for a in balls_qualified[m]:  # a.shape: [球内样本数, samplelabel+64样本向量]
                    true_label_lis.append(self.res[np.where((np.array(self.input) == a[1:]).all(axis=1))][0])

                # 需要修改，避免出现【2,3,4】的情况判断错误 ex:true=2<->target=4  一定程度减少了噪声球比例  -》无重复样本标签，球标签怎么算
                true_max_label = max(true_label_lis, key=true_label_lis.count)  # 球内样本真实标签列表中最多的样本标签即为球的真实标签
                temp_dic = {}
                for item in true_label_lis:
                    if item in temp_dic:
                        temp_dic[item] += 1
                    else:
                        temp_dic[item] = 1
                for x in temp_dic:
                    if temp_dic[x] >= temp_dic[true_max_label]:
                        max_label_lis.append(x)

                if target[m] not in max_label_lis:
                    ball_noise += 1
                    ball_sample_noise += len(balls_qualified[m])
            else:
                break
        
        
        ball_noise_percent = ball_noise / len(centers_qualified)
        ball_noise_percent_weight = ball_sample_noise / ballsample_total
        fw_write("\n该batch球样本噪声比例: {:.4%}\t, 带权球样本噪声比例: {:.4%}\t, 球样本噪声数:{:.1f}\t,  总球个数:{:.1f}\t".format(ball_noise_percent, ball_noise_percent_weight, ball_noise, len(centers_qualified)))
        ball_statistics(ball_noise,len(centers_qualified))
        ######## 下面为利用粒球修改标签
        # for m, i in enumerate(numbers_qualified):
        #     if m < int(args.retention_ratio * len(numbers_qualified)):
        #         for a in balls_qualified[m]:  # a.shape: [球内样本数, samplelabel+64样本向量]
        #             self.nlabel[np.where((np.array(self.input) == a[1:]).all(axis=1))] = a[0]

        return torch.Tensor(data), torch.Tensor(target), torch.Tensor(self.nlabel)
        


    """
    data1:[ball_number, 64维球心]
    target:[ball_number,ball_label]
    """

    @staticmethod
    # def backward(self, output_grad, input, index, id, new_input_label):
    def backward(self, output_grad, input, _):
        balls = np.array(self.balls,dtype=object)  # [ball_sample_numbers,ball_label+64维样本向量]
        # print("balls:\n", balls)

        result = np.zeros([self.batch_size, 68], dtype='float64')  # [bs,64+1]        11  ->   65  ->33
        # result = np.zeros([self.batch_size, 36], dtype='float64')  # [bs,64+1]        11  ->   65  ->33

        # dic = []
        for  i in range(output_grad.size(0)):  # 每个球内部样本
            for a in balls[i]:  # a:球balls[m]内的每个样本[1(ball_label)+64]
                # np.where(np.array(self.input) == a[1:])[0][0] 找的是行索引,存在重复的a
                # where_list = list(np.where(np.array(self.input) == a[1:])[:][0])
                # index_row = max(where_list, key=where_list.count)
                # result[index_row, 3:] = np.array(output_grad[m, :])
                result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 4:] = np.array(output_grad[i, :])
                # dic.append(index_row)                               # 用每个球心向量的梯度去更新球内所有样本的的梯度

        return torch.Tensor(result)
