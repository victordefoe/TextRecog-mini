import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append('../')
from ext.ntm.aio import EncapsulatedNTM as NTMCell


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, nclass):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size, hidden_size)
        # self.ntm = NTMCell(num_inputs=256, num_outputs=256, controller_size=64, controller_layers=2, num_heads=1, N=60, M=130)


        self.out_gen_h = nn.Linear(hidden_size, nclass)
        self.out_gen_c = nn.Linear(hidden_size, nclass)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.processed_batches = 0

    def forward(self, prev_hidden, feats, conv_feats):
        self.processed_batches = self.processed_batches + 1
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        # self.ntm.init_sequence(nB)

        ##-------------------
        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        alpha = F.softmax(emition) # nB * nT

        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))

        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0)
        # pytorch 0.4 version
        if alpha.size(0) == 1:
            # 不知道为什么，换了pytorch版本0.4以后， 当batch size为1的时候， context的第一个维度会消失掉
            context = context.unsqueeze(0)
        #-----------------------------------

        state = ['read_list', 'controller_state', ['heads_state']]

        # next_in_feat = F.softmax(self.out_gen_h(prev_hidden) + self.out_gen_c(context))
        cur_hidden = self.rnn(context, prev_hidden)
        # cur_hidden, _ = self.ntm(context)

        return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.generator = nn.Linear(hidden_size, num_classes)
        self.processed_batches = 0
        self.end_prob = 31
        self.max_length = 20
        self.num_classes = num_classes

    def forward(self, feats, conv_feats, lengths=None):
        self.processed_batches = self.processed_batches + 1
        # print(feats.size(), text_length)
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        assert(input_size == nC)


        # assert (nB == text_length)
        if lengths is not None:
            max_length, _ = lengths.max(0)
            self.max_length = max_length.cpu().data.numpy().item()
            num_labels = lengths.data.sum()
        # num_steps = text_length.data.max()
        # num_labels = text_length.data.sum()

        # output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
        # output_hiddens = []
        output_hiddens = Variable(torch.zeros(self.max_length, nB, hidden_size).type_as(feats.data))
        hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
        # max_locs = torch.zeros(num_steps, nB)
        # max_vals = torch.zeros(num_steps, nB)
        # for i in range(num_steps):
        key_value = 0
        probs = torch.zeros([nB, self.max_length, self.num_classes])
        tstep = 0


        while tstep < self.max_length:
            hidden, alpha = self.attention_cell(hidden, feats, conv_feats)

            # output_hiddens.append(hidden)
            output_hiddens[tstep] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                # max_locs[i] = max_loc.cpu()
                # max_vals[i] = max_val.cpu()
            prob = self.generator(hidden)
            key = prob.max(1)[1]

            key = torch.squeeze(key)
            key_value = key.data


            if lengths is not None:
                new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))

                # if probs is None:
                #     probs = key.data.type(torch.IntTensor)
                # else:
                #     probs = torch.cat([probs, key.data.type(torch.IntTensor)], 0)

                start = 0
                b = 0
                for length in lengths.data:
                    new_hiddens[start:start + length] = output_hiddens[0:length, b, :]
                    start = start + length
                    b = b + 1

            else:
                probs[:, tstep, :] = prob.data
                if key_value.cpu().numpy().item() == 112:
                    break
            # key_value = key.data.cpu().numpy().item()
            # print('key:', key)
            tstep += 1

        if lengths is not None:
            probs = self.generator(new_hiddens)
        else:
            probs = Variable(probs)
        # if self.processed_batches % 500 == 0:
        #     print('max_locs', max_loc)
        #     print('max_vals', max_val)
        #

        return probs

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False ):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        #self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))
        self.attention = Attention(nh, nh, nclass)

    def forward(self, input, length=None):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]   [width, batch_size, channels]


        # rnn features
        #
        rnn = self.rnn(conv)
        # print(rnn.size())
        output = self.attention(rnn, conv, length)

        return output
