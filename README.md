# Seq2Seq-Transformer-LRS-pytorch
Introduction
----
This is a project for seq2seq lip reading on a sentences-level lip-reading dataset called LRS2 (published by VGG, Oxford University) with 
transformer model. In this project, we implemented it with Pytorch.

Dependencies
----
* Python: 3.6+
* Pytorch: 1.3+
* Others

Dataset
----
This project is trained on LRS2 (grayscale).

##Training And Testing
We divide our whole training process into four main stages.
###Stage 1: Visual-frontend-pretrain-with-sub_sentences-samples
In this stage, we mainly train the visual frontend model to get a strong strength to 
extract effective feature. We use three kinds (1 word length, 2 word length, 3 word length) 
short sub-sentences samples by cropping the long pretrained samples to train the 
model (including the visual-frontend and seq2seq transformer model).
Focusing the model to adapt the three kinds of short sub-sentences samples, the 
visual-frontend can get a good ability to extract features. 
(We can also crop more short sub-sentences samples such as 4 word length, 5 word length and more.)

Preprocess the pretrained data to three kinds short sub-sentences samples:
```
cd Visual-frontend-pretrain-with-sub_sentences-samples
vim crop_sentences.py
#set word length to 1, 2, 3
python crop_sentences.py
#get pretrain_1_word_samples.txt, pretrain_2_word_samples.txt, pretrain_3_word_samples.txt
```

Train the whole model by the three-kinds short sub-sentences samples:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py
#get the final model BEST_checkpoint_pretrained_words.tar.
```
###Stage 2: Extract-all-samples-feature-with-pretrained-visual-frontend
In this stage, we use the pretrained visual frontend model trained by stage 1 to extract 
the original train samples. And we save these feature vector as a npy file by the following 
command:
```
cp -r Visual-frontend-pretrain-with-sub_sentences-samples/BEST_checkpoint_pretrained_words.tar 
Extract-all-samples-feature-with-pretrained-visual-frontend/
python extract_feats_from_frontend.py
#including pretrain and train data
``` 
###Stage 3: Train-seq2seq-transformer(without_visual_frontend)-with-features
Based on stage 1 and stage 2, we get a pretrained seq2seq transformer model 
(trained in stage 1 and not including the visual frontend part)
and samples which have been extracted features.
So in this stage, our main task is to train the seq2seq transformer model.
We also load the pretrained seq2seq transformer model and continue to train the seq2seq 
transformer model with feature samples.
```
cp -r Visual-frontend-pretrain-with-sub_sentences-samples/BEST_checkpoint_pretrained_words.tar 
Extract-all-samples-feature-with-pretrained-visual-frontend/
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py
#get BEST_checkpoint_seq2seq_TM.tar.
```
###Stage 4: Finetune-seq2seq-transformer(including_visual_frontend)-with-original-samples
After stage 3, we can get a seq2seq transformer model which has a good performance. So in this 
stage, we combine the visual frontend model (trained in stage 1) and the seq2seq transformer model 
(trained in stage 3) to a whole pretrained model. By loading the whole pretrained model, we train the model with 
original train samples (not be extracted to feature).
```
cp -r Visual-frontend-pretrain-with-sub_sentences-samples/BEST_checkpoint_pretrained_words.tar 
Finetune-seq2seq-transformer(including_visual_frontend)-with-original-samples/
cp -r Train-seq2seq-transformer(without_visual_frontend)-with-features/BEST_checkpoint_seq2seq_TM.tar
Finetune-seq2seq-transformer(including_visual_frontend)-with-original-samples/
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py
```
###TEST
Last, when our training loss is converged, we can get the model called "BEST_checkpoint_final_words.tar".
Here, we provide a final model which is available at [GoogleDrive].
And copy the checkpoint to this folder. Here, we provide the beam search method with ngram_language model.
We can test the model as follows:
```
##When we test the model without language model, we set the beam size in "test_LM.py" to 1.
python test_LM.py
##When we test the model with language model, we set the beam size in "test_LM.py" to 2 (3,4,5).
python test_LM.py
```
Our testing results as follows (waiting...):
```
beam size=1, WER=% (Baseline)
beam size=2, WER=%
beam size=3, WER=%
beam size=4, WER=%
beam size=5, WER=%
```


 
