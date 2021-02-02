#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： XHao
# datetime： 2021/1/15 12:59 
# ide： PyCharm
# 定义一个类，利用torch卷积神经网络进实现识别检测等功能
import json

import torch
from PIL import Image
from torchvision import transforms
from ResNet_model import resnet50

class Checknet():
    def __init__(self):
        self.device = torch.device("cpu")
        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, img_path,net_path,name_json):
        # load image in rgb
        self.img = Image.open(img_path).convert('RGB')
        # [N, C, H, W]
        self.img = self.data_transform(self.img)
        # expand batch dimension
        self.img = torch.unsqueeze(self.img, dim=0)

        # read class_indict
        try:
            self.json_file = open(name_json, 'r')
            self.class_indict = json.load(self.json_file)
        except Exception as e:
            print(e)
            exit(-1)

        # create model
        self.model = resnet50(num_classes=39)
        # load model weights
        self.model_weight_path = net_path
        self.model.load_state_dict(torch.load(self.model_weight_path))
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            # predict class
            self.output = torch.squeeze(self.model(self.img))
            self.predict = torch.softmax(self.output, dim=0)
            self.predict_cla = torch.argmax(self.predict).numpy()
        print(self.class_indict[str(self.predict_cla)], self.predict[self.predict_cla].numpy())
        return self.class_indict[str(self.predict_cla)]