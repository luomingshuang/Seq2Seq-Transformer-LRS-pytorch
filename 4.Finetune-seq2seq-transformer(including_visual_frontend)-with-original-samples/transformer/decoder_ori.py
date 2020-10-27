import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from options import IGNORE_ID, device
from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from .utils import get_attn_key_pad_mask, get_attn_pad_mask, get_non_pad_mask, get_subsequent_mask, pad_list


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, sos_id, eos_id,
            n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=True)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
       # print('ys: ', ys)
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
       # print('ys_in: ', ys_in)
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        #print('ys_out: ', ys_out)
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        #print('ys_in_pad: ', ys_in_pad)
       # print('ys_out_pad: ', ys_out_pad)
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     pad_idx=self.eos_id)
        #print('subseq: ', slf_attn_mask_subseq, 'keypad: ', slf_attn_mask_keypad)
        #slf_attn_mask_keypad = slf_attn_mask_keypad.astype(int)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        output_length = ys_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length)

        # Forward
        dec_output = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        #print(self.tgt_word_prj)
        #print('seq_logit size: ', dec_output.size())
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        #print('seq_logit size: ', seq_logit.size())
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold

    def recognize_beam(self, encoder_outputs):
        #print(encoder_outputs.size())
        #maxlen = encoder_outputs.size(1)
        maxlen= 101
        #print(maxlen)
        # prepare sos
        ys_l2r = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long()
        #ys_r2l = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long()
        #print(ys)
        
        for i in range(maxlen):
                #last_id = ys.cpu().numpy()[0][-1]
                # -- Prepare masks
                non_pad_mask_l2r = torch.ones_like(ys_l2r).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask_l2r = get_subsequent_mask(ys_l2r)
                
                dec_output_l2r = self.dropout(self.tgt_word_emb(ys_l2r) * self.x_logit_scale +
                                  self.positional_encoding(ys_l2r))
                
                for n in range(self.n_layers):
                    dec_output_l2r, dec_slf_attn, dec_enc_attn = self.layer_stack[n](
                              dec_output_l2r, encoder_outputs,
                              non_pad_mask=non_pad_mask_l2r,
                              slf_attn_mask=slf_attn_mask_l2r,
                              #slf_attn_mask=None,
                              dec_enc_attn_mask=None)
                              
                pred_l2r = self.tgt_word_prj(dec_output_l2r[:,-1])
                
                pred_argmax_l2r = pred_l2r.argmax(-1)
                
                pred_argmax_l2r = pred_argmax_l2r.unsqueeze(-1)
    
                ys_l2r = torch.cat((ys_l2r, pred_argmax_l2r), 1)
           
        return ys_l2r


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
