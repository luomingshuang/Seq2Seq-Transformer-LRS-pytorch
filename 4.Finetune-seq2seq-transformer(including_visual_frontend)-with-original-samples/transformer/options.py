import torch

length_words = 24  ##max length words = 24

max_length_feats = 154

max_length_words = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

vocab_size = 47

sos_id = 0

eos_id = 1

frontend = False

print_freq = 1

IGNORE_ID = -1