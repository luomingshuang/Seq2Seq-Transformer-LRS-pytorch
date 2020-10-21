import os
from tqdm import tqdm
import glob
import pickle
import random
import cv2
import torch
import os,math

import numpy as np
from torch.utils.data import Dataset

import options as opt

char_list = ['sos', 'eos', ' ', '!', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 
'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^']

##train samples:
# max length_npy:  154
# max length_word:  24

##val samples:
# max length_npy:  153
# max length_word:  21

# train data number: 30881
# val data number: 1082
# test data number: 1243

# 6 words: 

##pretrain samples: 
# 4 words: 453211   max frames:   max characters length: 
# 6 words: 286885   max frames: 480--88  max characters length: 76--60
# 8 words: 203964   max frames: 497--112  max characters length: 79--69
# 10 words: 154282  max frames: 498--135 max characters length: 95--83
# 12 words: 120792 max frames: 524--146  max characters length: 114--95
# 14 words: 95759 max frames: 532--181   max characters length: 127--95
# 16 words: 76148 max frames: 551--204   max characters length: 139--105
# 18 words: 60880 max frames: 642--227   max characters length: 146--110
# 20 words: 49004 max frames: 602--250   max characters length: 163--113
# 22 words: 40182 max frames: 626--270   max characters length: 182--113
# 24 words: 33181 max frames: 649--290   max characters length: 194--115

class Mydataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.samples = []
        self.max_characters_length = []
        for each_path in self.path:
          if 'pretrain' in each_path:
             with open(each_path, 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip('\n') for line in lines]
                for line in lines:
                    items = line.strip().split(' ')
                    if int(items[1].split('/')[1])-int(items[1].split('/')[0]) <= 181:
                       self.max_characters_length.append(len(list(items[2])))
                       #print(len(list(items[2])))
                       self.samples.append(items)
          else:
             with open(each_path, 'r') as f:
                lines = f.readlines()
                lines = [line.rstrip('\n') for line in lines]
                for line in lines:
                    items = line.strip().split(' ')
                    if int(items[1]) <= 181:
                       self.max_characters_length.append(len(list(items[2])))
                       #print(len(list(items[2])))
                       self.samples.append(items)
        #self.samples = list(filter(lambda x: int(x[1].split('/')[1])-int(x[1].split('/')[0]) <= 88, self.samples))
        self.samples = list(filter(lambda x: int(x[3]) <= opt.length_words, self.samples))
        print('max_characters_length: ', max(self.max_characters_length))
        print('the number of train data is: ', str(len(self.samples)))
    def __getitem__(self, i):
        sample = self.samples[i]
        npy_path = sample[0]
        feats = np.load(npy_path)
        #print(feats.shape)
        st = 0
        ed = 0
        if '/' in sample[1]:
           st, ed = sample[1].split('/')
           feats = self._load_boundary(feats, st, ed)
        else:
           feats = feats
        text = sample[2]
        text_length = sample[3]

        #feats = np.load(npy_path)
        #feats = self._load_boundary(feats, st, ed)
        feats = self._padding(feats, 184)

        trn = [char_list.index(c) for c in list(text.replace('/', ' '))]

        labels = np.pad(trn, (0, 105 - len(trn)), 'constant', constant_values=-1)
    
        return torch.FloatTensor(feats), torch.LongTensor(labels)

    def __len__(self):
        return len(self.samples)

    def _load_vid(self, array): 
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (120, 120)) for im in array]
        array = np.stack(array, axis=0)

        return array
    
    def _load_boundary(self, arr, st, ed):
        st = int(st)
        ed = int(ed)
        if st <= 2:
            st = 0
        return arr[st:ed]

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    def _padding_ones(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.ones(size))
        return np.stack(array, axis=0)


if __name__ == "__main__":
    import torch  
    from tqdm import tqdm

    #train_dataset = AiShellDataset(args, 'train')

    train_dataset = Mydataset(['pretrain_6_word_samples.txt', 'pretrain_8_word_samples.txt', 'pretrain_10_word_samples.txt', 'pretrain_12_word_samples.txt', 'pretrain_14_word_samples.txt'])
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, pin_memory=True, shuffle=True, num_workers=8)
    for i,data in enumerate(train_loader):
        if i==0:
          print(data[0].size())
          break
    print('train_dataset: ', len(train_dataset), len(train_loader))
