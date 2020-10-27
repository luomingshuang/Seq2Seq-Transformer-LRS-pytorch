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
from torch.utils.data.dataloader import default_collate

from config import pickle_file, IGNORE_ID, imgs_path, word_length, frequency
from utils import extract_feature

char_list = ['sos', 'eos', ' ', '!', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^']

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img):
    for i in range(batch_img.shape[0]):
        if(random.random() < 0.05 and 0 < i):
            batch_img[i] = batch_img[i - 1]
    return batch_img
    
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

class AiShellDataset_pretrain(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.samples = []
        length = 0
        duration_length = 0
        with open('pretrain_six_word_samples.txt', 'r') as f:
            lines = f.readlines()
            print('the number of samples: ', len(lines))
            #for key in dicts.keys():
            for line in lines:
                
                length_word = len(list(line.rstrip('\n').split(' ')[1]))
                duration = line.rstrip('\n').split(' ')[2].split('/')
                #duration_frames = int(float(duration)*25)
                if length_word <= 60:
                    duration_frames = float(duration[1])*25 - float(duration[0])*25
                    
                    if length_word >= length:
                        length = length_word   
                    #print(duration_frames)
                    #print(duration)
                    if duration_frames >= duration_length:
                        duration_length = duration_frames
                    if duration_frames <= 80:
                        self.samples.append((line.rstrip('\n').split(' ')[0], line.rstrip('\n').split(' ')[1], duration[0], duration[1]))
                    
        print('max length is: ', length)
        print('max frames is: ', duration_length)
       
        print('samples for {} : '.format(split), len(self.samples))

    def __getitem__(self, i):
        sample = self.samples[i]
        name = sample[0]
        images = np.load(name)
        images = self._load_boundary(images, sample[2], sample[3])
        #print(sample[1])
        trn = [char_list.index(c) for c in list(sample[1].replace('/', ' '))]
        #print(sample[1].replace('/', ' '))
        labels = np.pad(trn, (0, 60- len(trn)), 'constant', constant_values=IGNORE_ID)
        #print(trn)
        vid = self._load_vid(images)
        vid = self._padding(vid, 82)
    
        return torch.FloatTensor(vid), torch.LongTensor(labels)


    def __len__(self):
        return len(self.samples)

    def _load_vid(self, array): 
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (120, 120)) for im in array]
        #print(p, len(array))
        array = np.stack(array, axis=0)
        return array
    
    def _load_boundary(self, arr, st, ed):
        st = math.floor(float(st) * 25)
        ed = math.ceil(float(ed) * 25)
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
    from utils import parse_args
    from tqdm import tqdm

    args = parse_args()
    #train_dataset = AiShellDataset(args, 'train')

    train_dataset = AiShellDataset_pretrain(args, 'pretrain')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    for i,data in enumerate(train_loader):
        if i==0:
          print(data[0].size())
          break
    print('train_dataset: ', len(train_dataset), len(train_loader))


