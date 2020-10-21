import torch.nn as nn
#from .video_frontend import visual_model

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, visual_model):
        super(Transformer, self).__init__()
        #self.visual_frontend = visual_frontend(hiddenDim=512, embedSize=256)
        self.visual_frontend = visual_model
        # for p in self.parameters():
        #     p.requires_grad=False
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input):
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
       
        return padded_input

