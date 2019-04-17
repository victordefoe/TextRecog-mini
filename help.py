# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/3/13 16:18
@Contact: victordefoe88@gmail.com

@File: help.py
@Statement:

'''

import os
import torch
import torch.nn as nn
import math
loss = nn.CrossEntropyLoss()

def json():
    import json
    annotation_file = 'D:/BigDesign/data/COCO_Text/Train2014/annotations/COCO_Text.json'
    assert os.path.isfile(annotation_file), "file does not exist"
    dataset = json.load(open(annotation_file, 'r'))

    count = 0
    # for ann in dataset['anns']:
    #     if 'utf8_string' in dataset['anns'][ann]:
    #         print(dataset['anns'][ann])
    #         count += 1
    #     if count > 10:
    #         break
    print(dataset['imgs']['370250'])

json()





def crossEntropy():
    input = torch.randn(1, 5, requires_grad=True)
    target = torch.empty(1, dtype=torch.long).random_(5)


    output = loss(input, target)


    print("输入为5类:")
    print(input)
    print("要计算loss的类别:")
    print(target)
    print("计算loss的结果:")
    print(output)


    first = 0
    for i in range(1):
        first -= input[i][target[i]]  # 目标值的相反数
        second = 0

    for i in range(1):
        for j in range(5):
            second += math.exp(input[i][j])    # exp(Sigma 0->n {pred[i])})
    res = 0
    res += first +math.log(second)
    print("自己的计算结果：")
    print(res)


    # first = [0,0,0,0]
    # for i in range(4):
    #     first[i] -= input[i][target[i]]  # 目标值的相反数
    #     second = [0,0,0,0]
    #
    # for i in range(4):
    #     for j in range(5):
    #         second[i] += math.exp(input[i][j])    # exp(Sigma 0->n {pred[i])})
    # res = 0
    # for i in range(4):
    #     res += first[i] +math.log(second[i])
    # print("自己的计算结果：")
    # print(res/4)




