import glob
import os
import numpy as np

train_npy = '/data/lip/LRS2/crop_npy/train'
train_txt = '/data/lip/LRS2/text/train'

val_npy = '/data/lip/LRS2/crop_npy/val'
val_txt = '/data/lip/LRS2/text/val'

test_npy = '/data/lip/LRS2/crop_npy/test'
test_txt = '/data/lip/LRS2/text/test'

train_file = 'train_samples.txt'
val_file = 'val_samples.txt'
test_file = 'test_samples.txt'

def data_process(npy_path, txt_path, write_file, mode):
    length_words = []
    length_characters = []
    length_npys = []
    npy_files = glob.glob(os.path.join(npy_path, '*', '*.npy'))
    with open(write_file, 'w') as f_w:
        for npy in npy_files:
            items = npy.split('/')
            items[4] = 'text'
            items[5] = mode
            items[-1] = items[-1].split('.')[0]+'.txt'
            txt_file = '/'.join(items)
            length_npy = len(np.load(npy))
            length_npys.append(length_npy)

            with open(txt_file, 'r') as f:
                text = f.readlines()[0].split('  ')[1].rstrip('\n')

                length_words.append(len(text.split(' ')))

                length_characters.append(len(list(text)))

                sample = npy+' '+str(length_npy)+' '+'/'.join(text.split(' '))+' '+str(len(text.split(' ')))
            
                f_w.write(sample)
                f_w.write('\n')

            f.close()
        
    print('max length_npy: ', max(length_npys))
    print('max length_word: ', max(length_words))
    print('max length character: ', max(length_characters))

data_process(train_npy, train_txt, train_file, 'train')

data_process(val_npy, val_txt, val_file, 'val')

data_process(test_npy, test_txt, test_file, 'test')
