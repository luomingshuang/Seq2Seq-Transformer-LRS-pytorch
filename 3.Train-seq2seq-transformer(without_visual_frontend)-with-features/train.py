import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance

from options import device, vocab_size, sos_id, eos_id, print_freq, length_words
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
    
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        # visual_model = visual_frontend(hiddenDim=512, embedSize=256)
        
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder, decoder)

        optimizer = TransformerOptimizer(
            torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder, decoder)

        checkpoint = torch.load(checkpoint)
        print('loading model successful!')
        pre_model = checkpoint['model']
        pretrained_dict = pre_model.state_dict()

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        #optimizer = checkpoint['optimizer']
        #optimizer._update_lr()
        #lr = 0.0002
        optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    #model = model.cuda()
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    
    pretrain_dataset = Mydataset(['pretrain_6_word_samples.txt','pretrain_8_word_samples.txt','pretrain_10_word_samples.txt'])
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = Mydataset(['val_samples.txt'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    
    test_dataset = Mydataset(['test_samples.txt'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
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
        writer.add_scalar('model_{}/train_loss'.format(length_words), train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(length_words), lr, epoch)

        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        if epoch % 1 == 0:
            wer, cer = valid(valid_loader=valid_loader,model=model,logger=logger)
            #writer.add_scalar('model_{}/valid_loss'.format(word_length), valid_loss, epoch)
            writer.add_scalar('model_{}/valid_wer'.format(length_words), wer, epoch)
            writer.add_scalar('model_{}/valid_cer'.format(length_words), cer, epoch)
            
            wer, cer = valid(valid_loader=test_loader,model=model,logger=logger)
            writer.add_scalar('model_{}/test_wer'.format(length_words), wer, epoch)
            writer.add_scalar('model_{}/test_cer'.format(length_words), cer, epoch)

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
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    length = len(pretrain_loader)
    for i, (data) in enumerate(pretrain_loader):
        # Move to GPU, if available
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred, gold = model(padded_input, padded_target)
    
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        
        n += 1
        
        losses.update(loss.item())

#        if n >= 200:
 #          break        

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
        if a > 30:
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
