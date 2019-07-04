import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *

import numpy as np
import math
import time
import os

from logger import Logger
from tqdm import tqdm

from data import *
from utils import *
from model.Seq2Seq import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from bleu import *


class Trainer(object):
    def __init__(self, dp, args):

        # Language setting
        self.max_len = args.max_len
        self.args = args

        # Data Loader
        self.dp = dp
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self.device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

        # Path
        self.data_path = args.data_path
        self.sample_path = os.path.join('./samples/' + args.sample)
        self.log_path = os.path.join('./logs/' + args.log)

        if not os.path.exists(self.sample_path): os.makedirs(self.sample_path)
        if not os.path.exists(self.log_path): os.makedirs(self.log_path)

        # Hyper-parameters
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer

        # Training setting
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        # self.iter_per_epoch = len(train_loader)

        # Log
        self.logger = open(self.log_path+'/log.txt','w')
        self.sample = open(self.sample_path+'/sample.txt','w')
        self.tf_log = Logger(self.log_path)

        self.build_model(self.dp.vocab)

    def pad_sequences(self, s):
        pad_token = self.dp.vocab.stoi['<PAD>']
        # print(s)
        lengths = [len(s1) for s1 in s]
        longest_sent = max(lengths)
        padded_X = np.ones((self.args.batch_size, longest_sent), dtype=np.int64) * pad_token
        for i, x_len in enumerate(lengths):
            sequence = s[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
        # print(padded_X)
        return padded_X


    def build_model(self, vocabs):
        # build dictionaries
        self.vocab = self.dp.vocab
        # print(len(self.vocab.itos))
        self.encoder = Encoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)
        self.decoder = Decoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)

        # build the model
        self.model = Seq2Seq(self.encoder, self.decoder, self.device, len(self.vocab.itos)).to(self.device)

        # set the criterion and optimizer
        self.criterion = nn.NLLLoss(ignore_index=self.dp.vocab.stoi['<PAD>'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)

        if torch.cuda.is_available():
            self.model.cuda()

        print (self.model)
        print (self.criterion)
        print (self.optimizer)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate(self):
        self.model.eval()
        start_time = time.time()
        epoch_loss = 0
        num_batches = len(self.dp.val_src)//self.args.batch_size
        epoch_bleu = 0
        tgt = []
        pred = []
        with torch.no_grad():
            for i in range(num_batches):
                src = self.pad_sequences(self.dp.val_src[i*self.args.batch_size : (i+1)*self.args.batch_size])
                tgt = self.pad_sequences(self.dp.val_tgt[i*self.args.batch_size : (i+1)*self.args.batch_size])
                lng = self.dp.val_srclng[i*self.args.batch_size : (i+1)*self.args.batch_size]

                src = torch.LongTensor(src).to(self.device).transpose(0, 1)
                tgt = torch.LongTensor(tgt).to(self.device).transpose(0, 1)
                lng = torch.LongTensor(lng).to(self.device)

                output = self.model(src, tgt, lng)
                # print(output.size())
                output_ = output[1:].view(-1, output.shape[-1])
                # print(tgt[1:, :].size())
                tgt_ = (tgt[1:, :]).contiguous().view(-1)
                loss = self.criterion(output_, tgt_)
                epoch_loss += loss.item()

                pred_sents = []
                trg_sents = []
                output = output.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

                for j in range(self.args.batch_size):
                    # print(output.size())
                    pred_sent = self.get_sentence(output[j, :, :].data.cpu().numpy().argmax(axis=-1).tolist(), 'tgt')
                    trg_sent = self.get_sentence(tgt[j].data.cpu().numpy().tolist(), 'tgt')
                    pred_sents.append(pred_sent)
                    trg_sents.append(trg_sent)
                epoch_bleu += get_bleu(pred_sents, trg_sents)
            message = "Val loss: %1.3f  val_bleu:  %1.3f , val_ppl: %4.3f elapsed: %1.3f " % (
            epoch_loss/num_batches, epoch_bleu/num_batches, np.exp(epoch_loss/num_batches), time.time() - start_time)
            print(message)

        return epoch_bleu/num_batches

    def train(self):
        self.best_bleu = .0
        patience = 0
        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

        for epoch in range(1, self.num_epoch):
            #self.scheduler.step()
            self.train_loss = 0
            self.train_bleu = 0
            start_time = time.time()

            num_batches = len(self.dp.train_src)//self.args.batch_size

            for i in tqdm(range(num_batches)):
                self.model.train()

                src = self.pad_sequences(self.dp.train_src[i*self.args.batch_size : (i+1)*self.args.batch_size])
                tgt = self.pad_sequences(self.dp.train_tgt[i*self.args.batch_size : (i+1)*self.args.batch_size])
                lng = self.dp.train_srclng[i*self.args.batch_size : (i+1)*self.args.batch_size]

                src = torch.LongTensor(src).to(self.device).transpose(0, 1)
                tgt = torch.LongTensor(tgt).to(self.device).transpose(0, 1)
                lng = torch.LongTensor(lng).to(self.device)
                # print(src.size())
                # print(tgt.size())

                output = self.model(src, tgt, lng)
                # print(output.size())
                output_ = output[1:].view(-1, output.shape[-1])
                # print(tgt[1:, :].size())
                tgt_ = (tgt[1:, :]).contiguous().view(-1)

                self.optimizer.zero_grad()
                loss = self.criterion(output_, tgt_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.train_loss += loss.item()

                pred_sents = []
                trg_sents = []
                output = output.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

                for j in range(self.args.batch_size):
                    pred_sent = self.get_sentence(output[j].data.cpu().numpy().argmax(axis=-1).tolist(), 'tgt')
                    trg_sent = self.get_sentence(tgt[j].data.cpu().numpy().tolist(), 'tgt')
                    pred_sents.append(pred_sent)
                    trg_sents.append(trg_sent)
                bleu_value = get_bleu(pred_sents, trg_sents)
                self.train_bleu += bleu_value

                if i%self.args.log_interval == 0 and i>0:
                    message = "Train epoch: %d  iter: %d  train loss: %1.3f  train_bleu:  %1.3f , train_ppl: %4.3f elapsed: %1.3f " % (
                    epoch, i, self.train_loss/self.args.log_interval, self.train_bleu/self.args.log_interval, np.exp(self.train_loss/self.args.log_interval), time.time() - start_time)
                    print(message)
                    self.train_loss = 0
                    self.train_bleu = 0
                    start_time = time.time()

            val_bleu = self.evaluate()
            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                torch.save(self.model, 'models/model.pb')
                patience = 0
            else:
                patience +=1
            if patience > 3:
                break

    def get_sentence(self, sentence, side):
        def _eos_parsing(sentence):
            if '<EOS>' in sentence:
                return sentence[:sentence.index('<EOS>')+1]
            else:
                return sentence

        # index sentence to word sentence
        sentence = [self.dp.vocab.itos[s] for s in sentence]

        return _eos_parsing(sentence)


    def print_train_result(self, epoch, train_iter, start_time):
        mode = ("=================================        Train         ====================================")
        print (mode, '\n')
        self.logger.write(mode+'\n')
        self.sample.write(mode+'\n')

        message = "Train epoch: %d  iter: %d  train loss: %1.3f  train bleu: %1.3f  elapsed: %1.3f " % (
        epoch, train_iter, self.train_loss.avg, self.train_bleu.avg, time.time() - start_time)
        print (message, '\n\n')
        self.logger.write(message+'\n\n')


    def print_valid_result(self, epoch, train_iter, val_bleu, start_time):
        mode = ("=================================        Validation         ====================================")
        print (mode, '\n')
        self.logger.write(mode+'\n')
        self.sample.write(mode+'\n')

        message = "Train epoch: %d  iter: %d  train loss: %1.3f  train_bleu:  %1.3f  val bleu score: %1.3f  elapsed: %1.3f " % (
        epoch, train_iter, self.train_loss.avg, self.train_bleu.avg, val_bleu, time.time() - start_time)
        print (message, '\n\n' )
        self.logger.write(message+'\n\n')


    def print_sample(self, batch_size, epoch, train_iter, source, target, pred):

        def _write_and_print(message):
            for x in message:
                self.sample.write(x+'\n')
            print ((" ").join(message))

        random_idx = randomChoice(batch_size)
        src_sample = self.get_sentence(tensor2np(source)[random_idx], 'src')
        trg_sample = self.get_sentence(tensor2np(target)[random_idx], 'trg')
        pred_sample = self.get_sentence(tensor2np(pred[random_idx]).argmax(axis=-1), 'trg')

        src_message = ["Source Sentence:    ", (" ").join(src_sample), '\n']
        trg_message = ["Target Sentence:    ", (" ").join(trg_sample), '\n']
        pred_message =  ["Generated Sentence: ", (" ").join(pred_sample), '\n']

        message = "Train epoch: %d  iter: %d " % (epoch, train_iter)
        self.sample.write(message+'\n')
        _write_and_print(src_message)
        _write_and_print(trg_message)
        _write_and_print(pred_message)
        self.sample.write('\n\n\n')
