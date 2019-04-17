#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os
import random

def create_list(dir=None, set=None, end_token='#'):
    if set == None:
        set = 'train'
    if dir == None:
        dir = './data/%s/' % set
    allfiles = os.listdir(dir)
    list_path = './data/%s_list.txt' % set

    with open(list_path, 'w') as f:
        for file in allfiles:
            path = os.path.join(dir, file)
            label = file.split('_')[0]
            label = list(label)
            label = ':'.join(label)
            label = label + ':' + end_token
            f.write('%s %s \n' % (path, label))



def read_alphabet(dir):
    with open(dir, 'r', encoding='utf-8') as f:
        cont = f.readline()
        final_dict = {}
        while True:
            if cont == '':
                break

            cont = cont.split('\t')
            try:
                cont[0] = cont[0].strip()
                cont[1] = cont[1].strip()
            except:
                print(cont)
            # print(cont)
            if cont[0] not in final_dict:
                final_dict[int(cont[0])]=cont[1]
            cont = f.readline()
    # print('read_alphbet:', final_dict)
    return final_dict



class listDataset(Dataset):
    def __init__(self, list_file=None, transform=None, target_transform=None):
        with open(list_file) as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split(' ')
        imgpath = line_splits[0]
        try:
            img = Image.open(imgpath).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = line_splits[1]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels



class reformdatasets(object):

    def __init__(self, datasetname=None):
        assert datasetname != None , 'Please specific the dataset\'s name'
        self.names_dict = {'car_plate_rec':self.car_plate, 'synth_90k':self.synth90k,
                           'coco_text':self.coco_text}
        assert datasetname in self.names_dict.keys(), 'Invalid dataset name!'
        self.dataset = self.names_dict[datasetname]

    def __call__(self, data_dir, opt_sep='$', end_char='#'):
        return self.dataset(data_dir, opt_sep, end_char)

    def coco_text(self, data_dir, space_char, end_char):
        image_dir = os.path.join(data_dir, 'image')
        text_dir = os.path.join(data_dir, 'text')
        all_images = os.listdir(image_dir)
        all_text = os.listdir(text_dir)
        num_instances = len(all_text)
        seperate_num = num_instances // 10
        list_dirs = ['train_list.txt', 'val_list.txt', 'test_list.txt']
        ranges = [0, seperate_num * 8, seperate_num * 9, num_instances]

        error_num = 0
        for ind in range(len(list_dirs)):
            with open(os.path.join(data_dir, list_dirs[ind]), 'w', encoding='utf-8') as f_list:
                for i in range(ranges[ind], ranges[ind+1]):
                    im_path = os.path.join(image_dir, str(i)+'.jpg')
                    text_path = os.path.join(text_dir, str(i)+'.txt')
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f_txt:
                            label = f_txt.readline()
                        label = space_char.join(label) + space_char + end_char
                        f_list.write('%s %s \n' % (im_path, label))
                    except:
                        error_num += 1
                        continue
                print('Totally %d instances in %s' % (i - ranges[ind] - error_num, list_dirs[ind]))

    def car_plate(self, datadir, space_char, end_char):
        raise NotImplementedError

    def synth90k(self, datadir, space_char, end_char):
        root_dir = './data/synth_90k/'
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        ann_train_paths = ['annotation_train.txt', 'annotation_val.txt', 'annotation_test.txt']
        list_dirs = ['train_list.txt', 'val_list.txt', 'test_list.txt']
        total_samples = [7224612, 802734,891917]
        num_samples = [120000, 1200, 1200]
        for i in range(len(list_dirs)):
            writing_file = os.path.join(root_dir, list_dirs[i])
            ann_train_path = os.path.join(datadir, ann_train_paths[i])
            if os.path.exists(writing_file):
                continue
            rs = random.sample(range(total_samples[i]), num_samples[i])
            with open(writing_file, 'w') as fw:
                with open(ann_train_path, 'r') as fr:
                    for iters in range(int(12e5)):
                        cont = fr.readline()
                        if iters in rs:
                            label = cont.split(' ')[0].split('_')[1]
                            label = label + end_char
                            label = space_char.join(label)
                            path = os.path.join(datadir, cont.split(' ')[0][2:])
                            fw.write('%s %s \n' % (path, label))
                        if (iters+1) % 10000 == 0 :
                            print('dealing: %d k / %.3f' % ((iters+1)//1000, (iters+1)/12e5))



if __name__ == '__main__':
    dataset = reformdatasets('synth_90k')
    dataset('D:/BigDesign/data/synth90k/mjsynth/mnt/ramdisk/max/90kDICT32px')





