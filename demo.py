import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import time

import models.crnn as crnn


# diction = {'<nul>': 30, '1': 32, '0': 29, '3': 34, '2': 31, '5': 36, '4': 33, '7': 38, '6': 35,
#         '9': 46, '8': 37, 'A': 45, 'C': 47, 'B': 48, 'E': 49, 'D': 50, 'G': 51, 'F': 52, 'I': 53,
#         'H': 54, 'K': 41, 'J': 17, 'M': 19, 'L': 18, 'O': 13, 'N': 20, 'Q': 15, 'P': 14, 'S': 25,
#         'R': 16, 'U': 56, 'T': 26, 'W': 63, 'V': 55, 'Y': 60, 'X': 61, '[': 7, 'Z': 59, ']': 9, '\\': 8,
#         'a': 58, 'c': 42, 'b': 57, 'e': 27, 'd': 40, 'g': 2, 'f': 28, 'i': 23, 'h': 4, 'k': 21, 'j': 24,
#         'm': 3, 'l': 22, 'o': 65, 'n': 10, 'q': 11, 'p': 39, 's': 1, 'r': 12, 'u': 0, 't': 62, 'w': 44,
#         'v': 64, 'y': 5, 'x': 43, 'z': 6, '#': 66}
#
# dict_reverse = dict((v,k) for k,v in diction.items())

dict_reverse = dataset.read_alphabet('./data/charset_size=134.txt')
diction = dict((v,k) for k,v in dict_reverse.items())


char_show_dict ={u'京': 'A', u'沪': 'B', u'津': 'C', u'渝': 'D', u'冀': 'E', u'晋': 'F',
                 u'蒙': 'G', u'辽': 'H', u'吉': 'I', u'黑': 'J',u'苏': 'K', u'浙': 'L',
                 u'皖': 'M', u'闽': 'N', u'赣': 'O', u'鲁': 'P', u'豫': 'Q', u'鄂': 'R',
                 u'湘': 'S', u'粤': 'T',u'桂': 'U', u'琼': 'V', u'川': 'W', u'贵': 'X',
                 u'云': 'Y', u'藏': 'Z', u'陕': 'o', u'甘': 'i', u'青': '[', u'宁': '\\',u'新': ']'}



char_show_dict_reverse = dict((v,k) for k,v in char_show_dict.items())



print('dict_reverse', dict_reverse)


def transfer(dic, inp_str):
    new_l = []
    for each in inp_str:
        if each in dic:
            new = dic[each]
        else:
            new = each
        new_l.append(new)
    new_str = ''.join(new_l)
    return new_str



def read_image(path):
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    return image


def test(model, data_path, max_iter=100):
    test_dataset = dataset.listDataset(list_file=data_path, transform=dataset.resizeNormalize((100, 32)))

    image = torch.FloatTensor(64, 3, 32, 32)
    text = torch.LongTensor(64 * 5)
    length = torch.IntTensor(64)
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    print('Start test')
    # for p in crnn.parameters():
    #     p.requires_grad = False

    length = torch.IntTensor(1)
    length[0] = 7
    length = Variable(length)

    model.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=64, num_workers=int(2))
    test_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = test_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        print('cpu_image:', cpu_images.size())
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)


        preds = model(image, length)



        _, preds = preds.max(1)
        preds = preds.view(-1)
        sim_preds = converter.decode(preds.data, length.data)
        for pred, target in zip(sim_preds, cpu_texts):
            target = ''.join(target.split(':'))
            if pred == target:
                n_correct += 1

    for pred, gt in zip(sim_preds, cpu_texts):
        gt = ''.join(gt.split(':'))
        print('%-20s, gt: %-20s' % (pred, gt))

    accuracy = n_correct / float(max_iter * 64)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


if __name__ == '__main__':

    model_path = './expr_basic/netCRNN_varleng_190_800.pth'
    test_img_dir = './data/test_test/'
    seq_leng = 7
    report_file = './data/report3.txt'

    img_paths = [os.path.join(test_img_dir, p) for p in os.listdir(test_img_dir)]

    # alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    # charlist = dataset.read_alphabet('./data/charset_size=66.txt').values()
    # alphbet_default = ':'.join(charlist)

    charlist = dict_reverse.values()
    alphabet = ':'.join(charlist)

    model = crnn.CRNN(32, 1, len(diction), 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(diction, dict_reverse, '$')
    # converter = utils.strLabelConverterForAttention(alphbet_default, ':')

    # model.cpu()
    model.eval()
    length = torch.IntTensor(1)
    length[0]=seq_leng
    length = Variable(length)

    dataset.create_list(test_img_dir, 'test')



    idx = 0
    # 输出报告

    print("============开始测试==========")

    strict_prec, easy_prec, recall, total, seq_prec = 0, 0, 0, 0.0, 0
    dur_times = []
    fid = open(report_file, 'w')
    fid.write('========这是运行模型生成的测试报告==========\r\n创建时间：%s\r\n\r\n' % time.asctime())
    fid.write('编号          真实结果         模型预测结果         正确字符数           运行时间\r\n')
    fid.write('___________________________________________________________________________________________________\r\n')

    #
    for img_path in img_paths:
        start_time = time.time()
        image = read_image(img_path)
        preds = model(image)
        cor = img_path[-13:-6]
        name = os.path.split(img_path)[1]
        # print('pred_raw:', preds)

        _, preds = preds.max(2)


        preds = preds.squeeze(0)
        #pytorch 0.4 version
        preds = preds.unsqueeze(0)

        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=True)


        # print('%-20s => %-20s' % (raw_pred, sim_pred))

        end_time = time.time()
        dur_time = end_time - start_time
        dur_times.append(dur_time)

        # cor = transfer(char_show_dict_reverse, cor).upper()
        # sim_pred = transfer(char_show_dict_reverse, sim_pred).upper()



        print('编号： %d' % (idx), '真实结果： ', name, '/  模型预测结果：', sim_pred, '     | 运行时间：', dur_time)
        try:
            s, e, r, t = utils.compare(cor, sim_pred, ignore_case=True)
            strict_prec += s
            easy_prec += e
            recall += r
            total += t
            if s == seq_leng:
                seq_prec += 1
            fid.write(' %-03d     |     %-8s    |     %-10s     |      %-6d  |       %03.6f s   \r\n' % (
            idx, cor.upper(), sim_pred, s, dur_time))
        except Exception as error:
            print('因为错误：{%s} \n编号{%d}号测试文件，因为无法比较被排除在统计结果外' % (error, idx))

        idx += 1

    print('--------运行结果报告----------')
    print('* 效果测算: ')
    print('共计: %d \n严格：%d \n精确：%d \n召回：%d' % (total, strict_prec, easy_prec, recall))
    print('\n严格：%.3f%%   |   精确率：%.3f%%   |   召回率：%.3f%% ' % (
    strict_prec / total * 100, easy_prec / total * 100, recall / total * 100))


    print('\n** 运行时间测算：')

    tot_time = 0.0
    for each in dur_times:
        tot_time += each
    avg_time = tot_time / len(dur_times)
    print('共计 %d 项， \n共用时 %.6f 秒 \n单个样本预测平均用时：%.6f 秒' % (len(dur_times), tot_time, avg_time))
    print('序列正确率：%.6f%%     (%d/%d) \r\n' % (float(seq_prec) / len(dur_times) * 100, seq_prec, len(dur_times)))

    fid.write('___________________________________________________________________________________________________\r\n')
    fid.write('\r\n\r\n--------运行结果报告----------\r\n* 效果测算: \r\n')
    fid.write('序列正确率：%.6f%%     (%d/%d) \r\n' % (float(seq_prec) / len(dur_times) * 100, seq_prec, len(dur_times)))
    fid.write('共计: %d \r\n严格：%d \r\n精确：%d \r\n召回：%d\r\n' % (total, strict_prec, easy_prec, recall))
    fid.write('\r\n严格：%.3f%%   |   精确率：%.3f%%   |   召回率：%.3f%% \r\n' % (
    strict_prec / total * 100, easy_prec / total * 100, recall / total * 100))
    fid.write('\r\n** 运行时间测算：\r\n')
    fid.write('共计 %d 项， \r\n共用时 %.6f 秒 \r\n单个样本预测平均用时：%.6f 秒\r\n' % (len(dur_times), tot_time, avg_time))
    fid.close()

