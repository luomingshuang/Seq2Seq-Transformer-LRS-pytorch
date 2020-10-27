import argparse 
import numpy as np
import torch
import os
import torch.nn as nn
# from torch import nn
from tqdm import tqdm
import editdistance

from options import device, vocab_size, sos_id, eos_id, print_freq
#from data_gen import AiShellDataset, pad_collate
from data_load import Mydataset
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.video_frontend import visual_frontend
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

char_list = ['sos', 'eos', ' ', '!', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 
'?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
'V', 'W', 'X', 'Y', 'Z', '^']

#torch.cuda.set_device(0)
#os.environ['CUDA_VISIBLE_DEVICES']='1'
def cer_compute(predict, truth):
    word_pairs = [(list(p[0]), list(p[1])) for p in zip(predict, truth)]
    #print(word_pairs)
    wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
    return np.array(wer).mean()

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        visual_model = visual_frontend(hiddenDim=512, embedSize=256)
        
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder, decoder, visual_model)

        optimizer = TransformerOptimizer(
            torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        visual_model = visual_frontend(hiddenDim=512, embedSize=256)
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder, decoder, visual_model)
       
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    #model = model.cuda()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0])

    valid_dataset = Mydataset(['val_samples.txt'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    test_dataset = Mydataset(['test_samples.txt'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    # Epochs
    k = 0
    for epoch in range(0, 1):
        # One epoch's validation
        if epoch % 1 == 0:
            #wer, cer = valid(valid_loader=valid_loader,model=model,logger=logger)
            wer, cer = valid(valid_loader=test_loader, model=model, logger=logger)

def parse_args_decode():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=1, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=1, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args

args_decode = parse_args_decode()

def valid(valid_loader, model, logger):
    model = model.module
    model.eval()

    losses = AverageMeter()
    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    #pred_all_txt = []
    #gold_all_txt = []
    # Batches
    wer = float(0)
    a = 0    
    for data in tqdm(valid_loader):

        # Move to GPU, if available
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        #input_lengths = input_lengths.to(device)
        #if padded_target.size(1) <= word_length:
        a += 1
        with torch.no_grad():
                # Forward prop.
                preds = model.recognize(padded_input, char_list, args_decode)
                #print(preds)
                pred_txt = []
                gold_txt = []
                #print(padded_target)
                golds = [char_list[one] for one in padded_target.cpu().numpy()[0] if one not in (sos_id, eos_id, -1)]
                changdu = len(golds)
                print(''.join(golds)) 
                pred = [char_list[one] for one in preds[0]['yseq'][:changdu+1] if one not in (sos_id, eos_id, -1)] 
                print(''.join(pred))
                
                pred_txt.append(''.join(pred))  
                gold_txt.append(''.join(golds))
                    
                pred_all_txt.extend(pred_txt)
                gold_all_txt.extend(gold_txt)
       
        #if a >=10:
         #  break
           
    wer = wer_compute(pred_all_txt, gold_all_txt)
    cer = cer_compute(pred_all_txt, gold_all_txt)

    print('wer: ', wer)
    print('cer: ', cer)

    return wer, cer

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
