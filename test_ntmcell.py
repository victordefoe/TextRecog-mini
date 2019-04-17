# encoding: utf-8
'''
@Author: 刘琛
@Time: 2018/11/28 22:30
@Contact: victordefoe88@gmail.com

@File: test_ntmcell.py
@Statement:

'''

import torch
from torch.autograd import Variable
from ext.ntm.aio import EncapsulatedNTM as NTMCell

ntmcell = NTMCell(num_inputs=100, num_outputs=256, controller_size=64, controller_layers=2, num_heads=1, N=60, M=130)
ntmcell.init_sequence(92)

inputs = torch.zeros((92,100)).type(torch.FloatTensor)
# inputs = Variable(inputs)
a = Variable(inputs)
hidden, state = ntmcell(inputs)
print(hidden)
print('shape: ', state[0][0].size(), state[1][0].size(), state[2][0].size())