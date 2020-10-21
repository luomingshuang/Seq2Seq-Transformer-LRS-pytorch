import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance

from config import device, print_freq, vocab_size, sos_id, eos_id, IGNORE_ID, word_length
#from data_gen import AiShellDataset, pad_collate
from data_gen_pretrain import AiShellDataset_pretrain

from transformer.video_frontend import visual_frontend
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence


visual_model = visual_frontend(hiddenDim=512, embedSize=256)

model = Transformer(visual_model)

pretrained_dict = torch.load('BEST_checkpoint_1_words.tar')
pretrained_dict = pretrained_dict['model']

pretrained_dict = pretrained_dict.state_dict()

model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
print('miss matched params:{}'.format(missed_params))
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

global args
args = parse_args()

#pretrain = '/data/lip/LRS2/crop_npy/pretrain'
#path = '/data/lip/LRS2/crop_npy/train'
path = '/data/lip/LRS2/crop_npy/test'

import glob
import os
files = glob.glob(os.path.join(path, '*', '*.npy'))
       
print(len(files))

for data in tqdm(files):
        save_name = data.replace('test', 'lms_test_feats')
        save_dir = save_name[:-10]

        if(not os.path.exists(save_dir)): os.makedirs(save_dir)
        # Move to GPU, if available
        padded_input = np.load(data)   
        #padded_input = padded_input.to(device)
        padded_input = torch.FloatTensor(padded_input)
        padded_input = padded_input.unsqueeze(0)
        
        frontend_feats = model(padded_input)
        frontend_feats = frontend_feats.squeeze(0)
        np.save(save_name, frontend_feats.detach().numpy())