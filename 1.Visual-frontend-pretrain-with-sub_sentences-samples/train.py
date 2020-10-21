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
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.video_frontend import visual_frontend
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

# import pickle
# pklfile = 'char_list.pkl'
# with open(pklfile, 'rb') as file:
#     data = pickle.load(file)
# char_list = data
char_list = ['sos', 'eos', ' ', '!', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '^']

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
    
    writer = SummaryWriter()
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

        # print(model)
        #model = nn.DataParallel(model, device_ids=[0,1])
        # pretrained_dict = torch.load('tf2torch_TM.tar')
        # model_dict = model.state_dict()
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        # print('loading TM successful!')

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        #start_epoch = checkpoint['epoch'] + 1
        print('loading model successful!')
        #epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        
        optimizer = checkpoint['optimizer']
        optimizer._update_lr()
        #lr = 0.0002
        #optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    #model = model.cuda()
    
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

    # Custom dataloaders
    
    
    # train_dataset = AiShellDataset(args, 'train')
    # #print(train_dataset[0][0].size())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    
    pretrain_dataset = AiShellDataset_pretrain(args, 'pretrain')
    #print(train_dataset[0][0].size())
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = AiShellDataset_pretrain(args, 'pretrain')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        
        train_loss, n = train(pretrain_loader=pretrain_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger, k=k)
        k = n
        print('train_loss: ', train_loss)
        writer.add_scalar('model_{}/train_loss'.format(word_length), train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(word_length), lr, epoch)

        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        if epoch % 1 == 0:
            wer, cer = valid(valid_loader=valid_loader,model=model,logger=logger)
            #writer.add_scalar('model_{}/valid_loss'.format(word_length), valid_loss, epoch)
            writer.add_scalar('model_{}/valid_wer'.format(word_length), wer, epoch)
            writer.add_scalar('model_{}/valid_cer'.format(word_length), cer, epoch)

            # Check if there was an improvement
            is_best = wer < best_loss
            #is_best = train_loss < best_loss
            best_loss = min(wer, best_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(pretrain_loader, model, optimizer, epoch, logger, k):
    #writer = SummaryWriter()
    
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    length = len(pretrain_loader)
    for i, (data) in enumerate(pretrain_loader):
        # Move to GPU, if available
        padded_input, padded_target = data
        #padded_input, padded_target = data

        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred, gold = model(padded_input, padded_target)
        #print(pred.size(), gold.size())
        #print(pred.size(), gold.size())
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        
        #writer.add_scalar('model_{}/train_iteration_loss'.format(word_length), loss.item(), n)
        n += 1
        
        #if n == 10:
         # break
        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(pretrain_loader), loss=losses))
        
        

    return losses.avg, n


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
                preds = model.recognize(padded_input)
                #print(pred.size())
                #loss, n_correct = cal_performance(pred, padded_target, smoothing=args.label_smoothing)
                #pred_argmax = pred.argmax(-1)
                #preds = pred_argmax
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                gold_txt = []
                length = preds.size(0)
                #length_r2l = predss_r2l.size(0)
                for n in range(length):
                #changdu = len(gold[n].cpu().numpy())
                    golds = [char_list[one] for one in padded_target[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    #print(preds[n].cpu().numpy())
                    pred = [char_list[one] for one in preds[n].cpu().numpy()[:changdu+1] if one not in (sos_id, eos_id, -1)]
                    
                    print('golds: ', ''.join(golds))
                    print('preds: ', ''.join(pred))
                    
                    pred_txt.append(''.join(pred))
                    #pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    #gold_phonemes.append(golds)
                    
                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)
        #if a >2000:
        if a > 2000:
            break
    wer = wer_compute(pred_all_txt, gold_all_txt)
    cer = cer_compute(pred_all_txt, gold_all_txt)
               
    #losses.update(loss.item())
    print('wer: ', wer)
    print('cer: ', cer)
    # Print status
    #logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return wer, cer


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
