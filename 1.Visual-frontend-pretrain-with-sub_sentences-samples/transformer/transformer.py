import torch.nn as nn
#from .video_frontend import visual_model

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder, visual_model):
        super(Transformer, self).__init__()
        #self.visual_frontend = visual_frontend(hiddenDim=512, embedSize=256)
        self.visual_frontend = visual_model
        # for p in self.parameters():
        #     p.requires_grad=False

        self.encoder = encoder
        # for p in self.parameters():
        #     p.requires_grad=False
        
        self.decoder = decoder
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #print(padded_input.size())
        # padded_input = padded_input.permute(0,4,1,2,3)
        padded_input = padded_input.unsqueeze(1)
        #print(padded_input.size())
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        #print('length is: ', length)
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        #print(padded_input.size(), input_lengths)
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        #print(encoder_padded_outputs.size())
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        #print(pred.size(), max(pred.size(1)))
        return pred, gold

    def recognize(self, input):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        padded_input = input.unsqueeze(1)
        #print(padded_input.size())
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        #print('length is: ', length)
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        #print(padded_input.size(), input_lengths)
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        #print(encoder_padded_outputs.size())
        pred = self.decoder.recognize_beam(encoder_padded_outputs)
        #print(pred.size(), max(pred.size(1)))
        return pred