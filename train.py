import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim
from tensorboardX import SummaryWriter


from models.preactresnet import PreActResNet18
from models.resnet import *

from densenet import densenet121
from config import args
import dataloader_cifar 
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint, accuracy, load_checkpoint, ThreeCropsTransform




############################# 全局变量
random.seed(42)
best_acc = 0.0  # 记录测试最高准确率
step = 0        # 记录总步数
save_pth = "%d_%s_%s_noise%.1f_pur(%.2f,%.2f)_bs%d_ratio%.2f_%s_GB-%s_%s" % (args.num,args.model,args.dataset,args.noisy_pro,args.purity,args.reclust_pur,args.BATCH_SIZE,args.retention_ratio,args.noisy_mode,args.GB_mode,args.noisy_mode)  # 定义保存路径名
if not os.path.isdir(os.path.join('log',save_pth)):
    os.makedirs(os.path.join('log',save_pth))
fw = open(os.path.join('log',save_pth,'out.log'),'w')
writer = SummaryWriter(os.path.join('log',save_pth))  # tensorboard记录,设定路径


def save_checkpoint(best_acc, model, optimizer, epoch):
    # print('Best Model Saving...')
    fw.write('\nBest Model Saving...')
    model_state_dict = model.state_dict()
    save_dir = os.path.join('checkpoints',save_pth)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'model_state_dict': model_state_dict,  # 网络参数
        'global_epoch': epoch,  # 最优对应epoch
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
        'best_acc': best_acc,  # 最好准确率
    }, os.path.join(save_dir,'checkpoint_model_best_{}.pth'.format(best_acc)))


def create_model(model_name,num_class,device):
    if model_name == 'preactresnet18':
        model = PreActResNet18()
    if model_name == 'resnet20':
        model = resnet20(num_classes=num_class)
    if model_name == 'resnet32':
        model = resnet32(num_classes=num_class)
    if model_name == 'resnet44':
        model = resnet44(num_classes=num_class)
    if model_name == 'resnet56':
        model = resnet56(num_classes=num_class)
    if model_name == 'densenet121':
        model = densenet121(num_classes=num_class)
    model = model.to(device)
    return model

def fw_write(str):
    fw.write(str)
    fw.flush()

############################# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定CPU or cuda
# fw.write("gpu: "device)
fw_write(str(args))

############################# 数据加载
if args.dataset == 'cifar10':
    warm_up = 0
    num_class = 10
elif args.dataset == 'cifar100':
    warm_up = 30
    num_class = 100

loader = dataloader_cifar.cifar_dataloader(args.dataset, 
                                           r=args.noisy_pro, 
                                           noise_mode=args.noisy_mode,
                                           batch_size=args.BATCH_SIZE,
                                           num_workers=8,
                                           root_dir=args.data_path,
                                           noise_file="%s/%.1f_%s.json"%(args.data_path,args.noisy_pro,args.noisy_mode))
############################# 模型载入
fw_write("\nBuilding model")

model = create_model(args.model,num_class, device)
fw_write(str(model))
# data_loader_train = loader.run('train')
train_loader = loader.run('train')
test_loader = loader.run('test')


# 分别提取模型的权重参数和偏置参数
weight_p, bias_p = [], []
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
# Q1 损失函数
cost = nn.CrossEntropyLoss().to(device)
# Q2 优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 5e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.weight_decay)
# Q3 学习率变化策略
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=0.1, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000 // (50000 // args.BATCH_SIZE),
                                                                        48000 // (50000 // args.BATCH_SIZE),
                                                                        64000 // (50000 // args.BATCH_SIZE)], gamma=0.1,
                                                 last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.1, patience=150)



def main():
    global best_acc  # 记录测试最高准确率
    global step      # 记录总步数
    since = time.time()  # 用于计算运行时间
    
    for epoch in tqdm(range(1, args.epochs + 1),total=args.epochs):

        model.train()  # 训练
        training_loss = 0.0
        epoch_optim = epoch
        # adjust_learning_rate(optimizer, epoch_optim, args)
        # print("Epoch {}/{}".format(epoch, args.epochs))
        fw_write("\n")
        fw_write("--" * 20)
        # exp_pool = []   # [[2,3],[4,9],[1,5],[6,8]]
        epoch_s_time = time.time()
        for i, train_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            fw_write("\n")
            fw_write("- -- -- -- -- -- -" * 4)
    
            x, (labels, index), original_labels = train_data
            x, labels, index, original_labels = x.to(device), labels.to(device), index.to(device), original_labels.to(device)
            # 载入样本进入模型    v2.1 加入 warmup 
            if args.GB_mode == 'False':
                y, target, _ = model(x, labels, index, original_labels, GB_layer=False, purity=args.purity)
            else:
                y, target, _ = model(x, labels, index, original_labels, GB_layer=True, purity=args.purity)
            loss = cost(y, target.long())  # 加入粒球后模型返回的什么到底,   每个batch的loss
            _, pred = torch.max(y.data, 1)
            train_correct = (pred == target).sum().item()
            train_accuracy = train_correct / target.size(0) 
            writer.add_scalar('Error/train', 100. - train_accuracy * 100., step)
            writer.add_scalar('Loss/train', loss.item(), step)
            # print("当次batch训练样本总数: ", x.size(0),"loss:",loss.item())
            fw_write('\n[{:d}, {:5d}]\tcurrent batch num: {}\tcurrent lr: {:5f}\tcurrent train accuracy: {:.5f}\ttrain loss:{:.5f}'.format(epoch,i,x.size(0),optimizer.param_groups[0]['lr'],train_accuracy,loss))
            training_loss += loss.item()  # 一个epoch的loss

            # Q4 反向传播
            # writer.add_scalar("Train/Loss", loss.item(), i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.print_intervals == args.print_intervals - 1:  
                model.eval()  # 进入测试，测试时不启用 Batch Normalization 和 Dropout
                num_correct = 0.
                test_total = 0
                with torch.no_grad():
                    for j, test_data in tqdm(enumerate(test_loader), total=len(test_loader)):
                        
                        test_img, test_label = test_data
                        test_img, test_label = test_img.to(device), test_label.to(device)

                        y, _, _ = model(test_img, test_label, [], [], GB_layer=False, purity=args.purity)
                        loss = cost(y, test_label.long())
                        _, pred = torch.max(y.data, 1)  # 返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
                        test_correct = (pred == test_label).sum().item()
                        writer.add_scalar('Error/test',100. - (test_correct / test_label.size(0) * 100.), step)
                        writer.add_scalar('Loss/test', loss.item(), step)
                        num_correct += test_correct
                        test_total += test_label.size(0)

                test_accuracy = num_correct / test_total * 100.
                fw_write('\n[%d, %5d]\ttrain_loss: %.4f\ttest_accuracy: %.2f\ttest_correct:%d\ttest_total:%d'
                      % (epoch, i + 1, training_loss / args.print_intervals, test_accuracy, num_correct, test_total))

                # writer.add_scalar("Test/Accuracy", test_accuracy, i)

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    best_acc_loc = epoch
                    save_checkpoint(best_acc, model, optimizer, epoch)
                training_loss = 0.0

                model.train()  # 返回训练

            step += 1
        
        fw_write('\nCurrent best acc:{:.2f}%\t at epoch{}'.format(best_acc, best_acc_loc))
        fw_write('\nCurrent Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],step)
        time_epoch = time.time() - epoch_s_time
        time_left = ((time.time() - since) / epoch) * (args.epochs - epoch)
        fw_write('\n每epoch时间:  {:.0f}h {:.0f}m {:.0f}s'.format(time_epoch // 60 // 60,(time_epoch // 60) - 60 * (time_epoch // 60 // 60), time_epoch % 60))
        fw_write('\n剩余用时: {:.0f}h {:.0f}m {:.0f}s'.format(time_left // 60 // 60,(time_left // 60) - 60 * (time_left // 60 // 60), time_left % 60))
              

    # writer.close()  # 终止该对象
    fw_write('\nFinal test best acc:{:.2f}%\t at epoch{}'.format(best_acc, best_acc_loc))
    # print('有粒球训练->eval最好精度:{:.4f}'.format(best_acc))
    time_used = time.time() - since
    # fw_write("\n",'-' * 30)
    fw_write('\n总训练用时: {:.0f}h {:.0f}m {:.0f}s'.format(time_used // 60 // 60, (time_used // 60) - 60 * (time_used // 60 // 60), time_used % 60))


if __name__ == '__main__':
    main()

