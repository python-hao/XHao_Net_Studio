#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： XHao
# datetime： 2021/1/14 13:11 
# ide： PyCharm
import numpy as np
from PyQt5.QtCore import QThread
from ResNet_model import resnet50,resnet18,resnet34,resnet101

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
import time

class Resnet_Train():
    def __init__(self,batch_signal,epoch_signal,epoch,batchsize,floor,lr,num_classes,dataset_path,exercise):
        """
        训练参数
        :param batch_signal: 画batch图的信号，用于与主线程UI通信
        :param epoch_signal: 画epoch图的信号，用于与主线程UI通信
        :param epoch: 迭代次数
        :param batchsize: 每次迭代图片数量
        :param num_classes: 分类类别数
        :param lr: 学习率
        :param dataset_path: 数据集路径
        :param exercise: 第几次训练
        """
        self.batch_signal = batch_signal
        self.epoch_signal = epoch_signal
        print(self.batch_signal)
        self.epoch = epoch
        self.batchsize = batchsize
        self.lr = lr
        self.floor = floor
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.exercise = exercise
        self.floors = [resnet18, resnet34, resnet50, resnet101]
        self.train()

    def train(self):
        img_save = r"./exercise_{}/log_01".format(self.exercise)
        if not os.path.exists(img_save):
            os.makedirs(img_save)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        # 图像预处理
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
                                         transforms.RandomHorizontalFlip(),  # 水平翻转
                                         #  transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor(),  # 转化为张量
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            #  transforms.Normalize([0.485],[0.485])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       # transforms.Grayscale(num_output_channels=1),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        # data_root = os.path.abspath(os.path.join(os.getcwd(), "./."))  # get data root path
        image_path = self.dataset_path  # flower data set path
        train_dataset = datasets.ImageFolder(root=image_path + "train",
                                             transform=data_transform["train"])
        train_num = len(train_dataset)
        print("训练集样本数量：", train_num)
        leaves_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in leaves_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('classes_names_fruit.json', 'w') as json_file:
            json_file.write(json_str)

        batch_size = self.batchsize
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=0)

        validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        print("验证集样本数量：", val_num)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=0)
        All_dataset = datasets.ImageFolder(root=image_path)
        ALL_num = len(All_dataset)
        print("样本总数量：", ALL_num)
        print("validate_loader:",len(validate_loader),"train_loader:",len(train_loader)) #628,2217

        # net = self.floors[self.floor](num_classes=self.num_classes)
        net = resnet50(num_classes=self.num_classes)
        net.to(device)

        loss_function = nn.CrossEntropyLoss()
        best_acc = 0.0
        save_path = 'exercise_{}/resNet50_{}.pth'.format(self.exercise, self.exercise)
        epochloss = []
        epochacc = []
        for epoch in range(self.epoch):
            # train
            start = time.perf_counter()
            net.train()
            running_loss = 0.0
            batch_loss = []
            batch_acc = []
            print("  learning--rate: {}".format(self.lr))
            optimizer = optim.Adam(net.parameters(), lr=self.lr)
            # 方便显示
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                # print train process
                rate = (step + 1) / len(train_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                batch_loss.append(loss)
                print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
                x = np.arange(0.0, 0.5, 0.01)
                x_batchloss = range(0, len(train_loader) + 1)
                y_batchloss = batch_loss
                x_batchacc = range(0, len(validate_loader)+ 1)
                y_batchacc = batch_acc
                # self.batch_signal.emit(epoch, x_batchloss, y_batchloss, x_batchacc, y_batchacc)
            print()
            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                for step_val, val_data in enumerate(validate_loader, start=0):
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += (predict_y == val_labels.to(device)).sum().item()
                    batch_acc.append(acc / val_num)

                    x_batchloss = range(0, len(train_loader) + 1)
                    y_batchloss = batch_loss
                    x_batchacc = range(0, len(validate_loader) + 1)
                    y_batchacc = batch_acc
                    # batch_signal.emit(epoch, x_batchloss, y_batchloss, x_batchacc, y_batchacc)

                val_accurate = acc / val_num
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, running_loss / step, val_accurate))
                epochloss.append(round(running_loss / step, 5))
                epochacc.append(round(val_accurate, 5))
                # epoch_signal.emit(epoch,range(self.epoch),epochloss,range(self.epoch),epochacc)
                with open(r'exercise_03_resized/result.txt', 'w', encoding='utf-8') as f:
                    f.write('epochacc:' + str(epochacc))
                    f.write('\n')
                    f.write('epochloss:' + str(epochloss))
                    f.write('\n')
            img_save = r"./exercise_{}/log_01".format(self.exercise)
            if not os.path.exists(img_save):
                os.makedirs(img_save)
            # 画图---自变量、变量
            x_batchloss = range(0, step + 1)
            y_batchloss = batch_loss
            x_batchacc = range(0, step_val + 1)
            y_batchacc= batch_acc
            # self.batch_signal.emit(epoch,x_batchloss,y_batchloss,x_batchacc,y_batchacc)
            # 一次epoch结束
            end = time.perf_counter()
            print('time useage: %.10f' % (end - start))
        print('Finished Training \n Painting.....')
        # # 最终结果绘制
        # x = range(0, 200)
        # y1 = epochacc
        # y2 = epochloss
        # plt.figure(1)
        # ax3 = plt.subplot(211)
        # ax4 = plt.subplot(212)
        # # 选择 ax3
        # plt.sca(ax3)
        # plt.title("val_accuracy---learning rate: {}".format(self.lr))
        # plt.xlabel('Epoch Number')
        # plt.ylabel('val_accuracy')
        # plt.plot(x, y1, color="red")
        # # 选择 ax4
        # plt.sca(ax4)
        # plt.title("train_loss---learning rate: {}".format(self.lr))
        # plt.xlabel('Epoch Number')
        # plt.ylabel('train_loss')
        # plt.plot(x, y2, color="green")
        # plt.tight_layout()
        # plt.savefig(r"./exercise_{}/log_01/result.png".format(self.exercise))
        # plt.show()
        # plt.close()
