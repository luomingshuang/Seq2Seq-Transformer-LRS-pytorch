###3 word length samples(no overlap): 618971 

import glob
import os

npy_path = '/data/lip/LRS2/crop_npy/pretrain'
txt_path = '/data/lip/LRS2/text/pretrain'

npy_files = glob.glob(os.path.join(npy_path, '*', '*.npy'))

print(len(npy_files))
word_length = 1
k=0
with open('pretrain_{}_word_samples.txt'.format(word_length), 'w') as f1:
  for npy in npy_files:
    items = npy.split('/')
    #print(items)
    items[4] = 'text'
    items[-1] = items[-1].split('.')[0]+'.txt'
    txt_file = '/'.join(items)
    #print(txt_file)
    with open(txt_file, 'r') as f:
        lines = f.readlines()[4:]
        #for line in lines[4:]
        length = len(lines)
        N = int(length / word_length)

        for i in range(N):
            text = []
            duration = []
            for one in range(i*word_length, (i+1)*word_length):
                items = lines[one].rstrip('\n').strip(' ').split(' ')
                word = items[0]

                st = float(items[1])
                ed = float(items[2])
                duration.append(st)
                duration.append(ed) 
                
                text.append(word)
            
            sample = npy+' '+'/'.join(text)+' '+str('/'.join([str(duration[0]), str(duration[-1])]))
            k += 1
            print(sample)
            f1.write(sample)
            f1.write('\n')
            
f1.close()
print('the {} word length samples is: '.format(word_length), k)
