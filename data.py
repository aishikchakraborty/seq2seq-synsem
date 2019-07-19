import torch
import unicodedata
from torchtext import data
from torchtext import datasets
import time
import re
import spacy
import os
import _pickle as pickle
from tqdm import tqdm
from collections import Counter
import collections
import random
random.seed(1234)
import mmap
import numpy as np

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'

def dd():
    return defaultdict(int)

def dd3():
    return 3

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Vocab():
    def __init__(self, max_vocab=10000):
        self.stoi = {}
        self.itos = []
        self.itos.append(SOS_WORD)
        self.itos.append(EOS_WORD)
        self.itos.append(PAD_WORD)
        self.itos.append(UNK_WORD)
        self.max_vocab = max_vocab


    def create_vocab_for_translation(self, all_translation_files_path):
        v = []
        f = open(all_translation_files_path, 'r')
        for lines in f:
            # lines = normalizeString(lines).split()
            lines = (lines).split()
            # print(lines)
            v.extend(lines)
        v = Counter(v).most_common(self.max_vocab)
        print('10 most common words: ' + str(v[:10]))
        v = [w[0] for w in v]
        self.itos.extend(v)
        self.stoi = collections.defaultdict(dd3, {v:k for k,v in enumerate(self.itos)})



class Preprocess():
    def __init__(self, train_src_file, train_tgt_file, val_src_file, val_tgt_file, vocab_generation_file, max_len=50):
        # self.vocab = vocab
        self.train_src_file = train_src_file
        self.train_tgt_file = train_tgt_file
        self.val_src_file = val_src_file
        self.val_tgt_file = val_tgt_file
        self.max_len = max_len

        self.lang2idx = {'en':0, 'fr':1, 'de':2, 'sos':3}

        self.vocab = Vocab()
        self.vocab.create_vocab_for_translation(vocab_generation_file)

        self.train_src = []
        self.train_tgt = []
        self.val_src = []
        self.val_tgt = []
        self.train_srclng = []
        self.train_tgtlng = []
        self.val_srclng = []
        self.val_tgtlng = []

        self.train_src_sentlen = []
        self.train_tgt_sentlen = []
        self.train_word_pair = []
        self.train_word_pair_y = []

        self.val_src_sentlen = []
        self.val_tgt_sentlen = []
        self.val_word_pair = []
        self.val_word_pair_y = []

    def preprocess_train(self):
        with open(self.train_src_file) as fsrc, open(self.train_tgt_file) as ftgt:
            for src_line, tgt_line in tqdm(zip(fsrc, ftgt), total=get_num_lines(self.train_src_file)):
                # src_line, tgt_line = normalizeString(src_line).split(), normalizeString(tgt_line).split()
                src_line, tgt_line = (src_line).split(), (tgt_line).split()
                # print(src_line);print(tgt_line);
                src_line = src_line
                tgt_line = tgt_line 
                src_seq = [self.vocab.stoi[w] for w in src_line[1:]][:self.max_len]
                tgt_seq = [self.vocab.stoi[w] for w in ['<SOS>'] + tgt_line[1:]][:self.max_len]
                # print(src_seq);print(tgt_seq);
                # print(self.vocab.itos);break;


                if len(src_seq)<5 or len(tgt_seq)<5:
                    continue

                self.train_src_sentlen.append((len(src_seq)-2)//15)
                self.train_tgt_sentlen.append((len(tgt_seq)-2)//15)

                x1 = range(1, len(src_seq)-2)
                # print(x1)
                src_words_idx = random.sample(x1, 2)


                if random.random() < 0.5:
                    self.train_word_pair.append([src_seq[src_words_idx[1]], src_seq[src_words_idx[0]]])
                    self.train_word_pair_y.append(0)
                else:
                    self.train_word_pair.append([src_seq[src_words_idx[0]], src_seq[src_words_idx[1]]])
                    self.train_word_pair_y.append(1)

                self.train_src.append(src_seq)
                self.train_tgt.append(tgt_seq)

                self.train_srclng.append(self.lang2idx[src_line[0]])
                self.train_tgtlng.append(self.lang2idx[tgt_line[0]])

    def preprocess_val(self):
        with open(self.val_src_file) as fsrc, open(self.val_tgt_file) as ftgt:
            for src_line, tgt_line in tqdm(zip(fsrc, ftgt), total=get_num_lines(self.val_src_file)):
                # src_line, tgt_line = normalizeString(src_line).split(), normalizeString(tgt_line).split()
                src_line, tgt_line = (src_line).split(), (tgt_line).split()
                src_line = src_line
                tgt_line = tgt_line

                src_seq = [self.vocab.stoi[w] for w in src_line[1:]][:self.max_len]
                tgt_seq = [self.vocab.stoi[w] for w in ['<SOS>'] + tgt_line[1:]][:self.max_len]

                if len(src_seq)<5 or len(tgt_seq)<5:
                    continue
                self.val_src_sentlen.append((len(src_seq)-2)//15)
                self.val_tgt_sentlen.append((len(tgt_seq)-2)//15)
                x1 = range(1, len(src_seq)-2)
                src_words_idx = random.sample(x1, 2)


                if random.random() < 0.5:
                    self.val_word_pair.append([src_seq[src_words_idx[1]], src_seq[src_words_idx[0]]])
                    self.val_word_pair_y.append(0)
                else:
                    self.val_word_pair.append([src_seq[src_words_idx[0]], src_seq[src_words_idx[1]]])
                    self.val_word_pair_y.append(1)

                self.val_src.append(src_seq)
                self.val_tgt.append(tgt_seq)

                self.val_srclng.append(self.lang2idx[src_line[0]])
                self.val_tgtlng.append(self.lang2idx[tgt_line[0]])



#
# class MaxlenTranslationDataset(data.Dataset):
# 	# Code modified from
# 	# https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py
# 	# to be able to control the max length of the source and target sentences
#
#     def __init__(self, path, exts, fields, max_len=None, **kwargs):
#
#         if not isinstance(fields[0], (tuple, list)):
#             fields = [('src', fields[0]), ('trg', fields[1])]
#
#         src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
#
#         examples = []
#         with open(src_path) as src_file, open(trg_path) as trg_file:
#             for src_line, trg_line in tqdm(zip(src_file, trg_file)):
#                 src_line, trg_line = src_line.split(' '), trg_line.split(' ')
#                 if max_len is not None:
#                 	src_line = src_line[:max_len]
#                     src_line = src_line + exts[0].split('.')[1]
#                 	src_line = str(' '.join(src_line))
#                 	trg_line = trg_line[:max_len]
#                     trg_line = trg_line.insert(exts[1].split('.')[1], 0)
#                 	trg_line = str(' '.join(trg_line))
#
#                 if src_line != '' and trg_line != '':
#                     examples.append(data.Example.fromlist(
#                         [src_line, trg_line], fields))
#
#         super(MaxlenTranslationDataset, self).__init__(examples, fields, **kwargs)
#
#
# class DataPreprocessor(object):
# 	def __init__(self):
# 		self.src_field, self.trg_field = self.generate_fields()
#
# 	def preprocess(self, train_path, val_path, train_file, val_file, src_lang, trg_lang, max_len=None):
# 		# Generating torchtext dataset class
# 		print ("Preprocessing vocab dataset...")
# 		train_dataset = self.generate_data(train_path, src_lang, trg_lang, max_len)
#
# 		print ("Saving train dataset...")
# 		self.save_data(train_file, train_dataset)
#
# 		print ("Preprocessing validation dataset...")
# 		val_dataset = self.generate_data(val_path, src_lang, trg_lang, max_len)
#
# 		print ("Saving validation dataset...")
# 		self.save_data(val_file, val_dataset)
#
# 		# Building field vocabulary
# 		self.src_field.build_vocab(train_dataset, max_size=30000)
# 		self.trg_field.build_vocab(train_dataset, max_size=30000)
#
# 		src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
#
# 		vocabs = {'src_vocab': src_vocab, 'trg_vocab':trg_vocab,
# 			  'src_inv_vocab':src_inv_vocab, 'trg_inv_vocab':trg_inv_vocab}
#
# 		return train_dataset, val_dataset, vocabs
#
# 	def load_data(self, vocab_generation_file, train_file, val_file):
#
# 		# Loading saved data
#         vocab_dataset = torch.load(vocab_generation_file)
# 		vocab_examples = vocab_dataset['examples']
#
#         train_dataset = torch.load(train_file)
# 		train_examples = train_dataset['examples']
#
# 		val_dataset = torch.load(val_file)
# 		val_examples = val_dataset['examples']
#
# 		# Generating torchtext dataset class
# 		fields = [('src', self.src_field), ('trg', self.trg_field)]
#
#         vocab_dataset = data.Dataset(fields=fields, examples=train_examples)
#         train_dataset = data.Dataset(fields=fields, examples=train_examples)
# 		val_dataset = data.Dataset(fields=fields, examples=val_examples)
#
# 		# Building field vocabulary
# 		self.src_field.build_vocab(train_dataset, max_size=30000)
# 		self.trg_field.build_vocab(train_dataset, max_size=30000)
#
# 		src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
# 		vocabs = {'src_vocab': src_vocab, 'trg_vocab':trg_vocab,
# 			  'src_inv_vocab':src_inv_vocab, 'trg_inv_vocab':trg_inv_vocab}
#
# 		return train_dataset, val_dataset, vocabs
#
#
# 	def save_data(self, data_file, dataset):
#
# 		examples = vars(dataset)['examples']
# 		dataset = {'examples': examples}
#
# 		torch.save(dataset, data_file)
#
# 	def generate_fields(self):
# 	    src_field = data.Field(tokenize=data.get_tokenizer('spacy'),
# 	                           eos_token=EOS_WORD,
# 	                           pad_token=PAD_WORD,
# 	                           include_lengths=True,
# 	                           batch_first=True)
#
# 	    trg_field = data.Field(tokenize=data.get_tokenizer('spacy'),
# 	                           eos_token=EOS_WORD,
# 	                           pad_token=PAD_WORD,
# 	                           include_lengths=True,
# 	                           batch_first=True)
#
# 	    return src_field, trg_field
#
# 	def generate_data(self, data_path, src_lang, trg_lang, max_len=None):
# 	    exts = ('.'+src_lang, '.'+trg_lang)
#
# 	    dataset = MaxlenTranslationDataset(
# 	        path=data_path,
# 	        exts=(exts),
# 	        fields=(self.src_field, self.trg_field),
# 	        max_len=max_len)
#
# 	    return dataset
#
# 	def generate_vocabs(self):
# 	    # Define string to index vocabs
# 	    src_vocab = self.src_field.vocab.stoi
# 	    trg_vocab = self.trg_field.vocab.stoi
#
# 	    # Define index to string vocabs
# 	    src_inv_vocab = self.src_field.vocab.itos
# 	    trg_inv_vocab = self.trg_field.vocab.itos
#
# 	    return src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab
