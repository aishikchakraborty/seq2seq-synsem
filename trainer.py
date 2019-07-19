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
from model.Seq2Seq import *
from model.Encoder import *
from model.Decoder import *
from model.ProbingTasks import ProbingTasks
from model.LabelSmoothing import *
from bleu import *

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))



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

        self.pad_idx = self.dp.vocab.stoi['<PAD>']

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

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def build_model(self, vocabs):
        # build dictionaries
        self.vocab = self.dp.vocab
        # print(len(self.vocab.itos))
        self.encoder = Encoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)
        self.decoder = Decoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)

        # build the model
        self.model = Seq2Seq(self.encoder, self.decoder, self.device, len(self.vocab.itos)).to(self.device)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        if self.args.mode == 'probe':
            self.probe = ProbingTasks(self.encoder, self.hidden_dim, 4, 2, False).to(self.device)
        elif self.args.mode == 'train':
            self.probe = ProbingTasks(self.encoder, self.hidden_dim, 4, 2, True).to(self.device)

        # set the criterion and optimizer
        self.criterion = nn.NLLLoss(ignore_index=self.dp.vocab.stoi['<PAD>'])
        self.criterion_ppl = nn.NLLLoss(ignore_index=self.dp.vocab.stoi['<PAD>'])

        self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.probe.parameters()), lr=self.lr)

        # self.optimizer = NoamOpt(self.model.embed_dim, 1, 400,
        # torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.probe.parameters()), lr=self.lr)

        self.optimizer2 = optim.Adam(self.probe.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)

        if torch.cuda.is_available():
            self.model.cuda()

        print(self.model)
        print(self.criterion)
        print(self.optimizer)

    def build_transmodel(self, vocabs):
        # build dictionaries
        self.vocab = self.dp.vocab
        # print(len(self.vocab.itos))
        self.encoder = TransEncoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)
        self.decoder = TransDecoder(len(self.vocab.itos), self.embed_dim, self.hidden_dim, self.num_layer)

        # build the model
        self.model = TransSeq2Seq(self.encoder, self.decoder, self.device, self.embed_dim, len(self.vocab.itos)).to(self.device)
        self.model.apply(self.init_weights)

        if self.args.mode == 'probe':
            self.probe = ProbingTasks(self.encoder, self.hidden_dim, 4, 2, False).to(self.device)
        elif self.args.mode == 'train':
            self.probe = ProbingTasks(self.encoder, self.hidden_dim, 4, 2, True).to(self.device)

        # set the criterion and optimizer
        self.criterion = LabelSmoothing(len(self.dp.vocab.itos), self.dp.vocab.stoi['<PAD>'], 0.0)
        self.criterion_ppl = nn.NLLLoss(ignore_index=self.dp.vocab.stoi['<PAD>'])
        self.criterion2 = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.probe.parameters()), lr=self.lr)

        self.optimizer2 = optim.Adam(self.probe.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)

        if torch.cuda.is_available():
            self.model.cuda()

        print(self.model)
        print(self.criterion)
        print(self.optimizer)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def greedy_decode(self, model, src, max_len, start_symbol):
        with torch.no_grad():
            # memory = model.encoder(src)
            ys = (torch.ones(1, src.size(1)) * start_symbol).type_as(src.data)
            probs = torch.FloatTensor(torch.zeros(max_len, src.size(1), len(self.dp.vocab.itos))).to(self.device)
            for i in range(max_len):
                prob = model(src, ys, 0)
                _, next_word = torch.max(prob[-1, :, :], dim=-1)
                # print(next_word.size())
                next_word = next_word.data
                # print(next_word)
                probs[i] = prob[-1, :, :]
                ys = torch.cat([ys,
                                torch.ones(1, src.size(1)).type_as(src.data)*next_word], dim=0)
            return ys, probs

    def evaluate(self):
        self.model.eval()
        start_time = time.time()
        epoch_loss = 0
        epoch_loss_ppl = 0
        num_batches = len(self.dp.val_src)//self.args.batch_size
        epoch_bleu = 0
        epoch_sentlen = 0
        epoch_wp = 0
        tgt = []
        pred = []
        with torch.no_grad():
            for i in range(num_batches):
                # hidden = self.model.encoder.init_hidden(self.args.batch_size)
                src = self.pad_sequences(self.dp.val_src[i*self.args.batch_size : (i+1)*self.args.batch_size])
                tgt = self.pad_sequences(self.dp.val_tgt[i*self.args.batch_size : (i+1)*self.args.batch_size])
                lng = self.dp.val_tgtlng[i*self.args.batch_size : (i+1)*self.args.batch_size]
                sentlen = self.dp.val_src_sentlen[i*self.args.batch_size : (i+1)*self.args.batch_size]
                word_pairs = self.dp.val_word_pair[i*self.args.batch_size : (i+1)*self.args.batch_size]
                word_pairs_y = self.dp.val_word_pair_y[i*self.args.batch_size : (i+1)*self.args.batch_size]

                src = torch.LongTensor(src).to(self.device).transpose(0, 1)
                tgt = torch.LongTensor(tgt).to(self.device).transpose(0, 1)
                lng = torch.LongTensor(lng).to(self.device)
                sentlen = torch.LongTensor(sentlen).to(self.device)
                word_pairs = torch.LongTensor(word_pairs).to(self.device)
                word_pairs_y = torch.LongTensor(word_pairs_y).to(self.device)


                output, logits = self.greedy_decode(self.model, src, src.size(0), self.dp.vocab.stoi['<SOS>'])
                # sentlen_out, wp_out = self.probe(src, lng, word_pairs)
                # loss_syn1 = self.criterion2(sentlen_out, sentlen)
                # loss_syn2 = self.criterion2(wp_out, word_pairs_y)

                output_ = logits.view(-1, logits.shape[-1])
                # print(tgt[1:, :].size())
                tgt_ = (tgt[1:, :]).contiguous().view(-1)
                loss = self.criterion(output_, tgt_)
                # loss /= (output.size(0)*output.size(1))
                epoch_loss_ppl += self.criterion_ppl(output_, tgt_).item()
                epoch_loss += loss.item()
                # epoch_sentlen += loss_syn1.item()
                # epoch_wp += loss_syn2.item()

                pred_sents = []
                trg_sents = []
                output = output[:].transpose(0, 1)
                tgt = tgt.transpose(0, 1)
                # print(output.size())
                # print(tgt.size())

                for j in range(self.args.batch_size):
                    # print(output.size())
                    pred_sent = self.get_sentence(output[j, 1:].data.cpu().numpy().tolist(), 'tgt')
                    trg_sent = self.get_sentence(tgt[j, 1:].data.cpu().numpy().tolist(), 'tgt')
                    pred_sents.append(pred_sent)
                    trg_sents.append(trg_sent)
                    if i == 1:
                        print('Pred: ' + str(' '.join(pred_sent)))
                        print('Target: ' + str(' '.join(trg_sent)))
                epoch_bleu += get_bleu(pred_sents, trg_sents)
            message = "Val loss: %1.3f  val_bleu:  %1.3f , val_ppl: %4.3f, val_sentlen: %1.3f, val_wp: %1.3f, elapsed: %1.3f " % (
            epoch_loss/num_batches, epoch_bleu/num_batches, np.exp(epoch_loss_ppl/num_batches), epoch_sentlen/num_batches, epoch_wp/num_batches, time.time() - start_time)
            print(message)

        return epoch_bleu/num_batches

    def train(self):
        self.best_bleu = .0
        patience = 0
        print(f'The model has {self.count_parameters(self.model):,} trainable parameters')

        for epoch in range(1, self.num_epoch):
            #self.scheduler.step()

            self.train_loss = 0
            self.train_loss_ppl = 0
            self.train_bleu = 0
            self.train_sentlen = 0
            self.train_wp = 0
            start_time = time.time()

            num_batches = len(self.dp.train_src)//self.args.batch_size

            for i in tqdm(range(num_batches)):
                self.model.train()
                # hidden = self.model.encoder.init_hidden(self.args.batch_size)
                src = self.pad_sequences(self.dp.train_src[i*self.args.batch_size : (i+1)*self.args.batch_size])
                tgt = self.pad_sequences(self.dp.train_tgt[i*self.args.batch_size : (i+1)*self.args.batch_size])
                tgtlng = self.dp.train_tgtlng[i*self.args.batch_size : (i+1)*self.args.batch_size]
                srclng = self.dp.train_srclng[i*self.args.batch_size : (i+1)*self.args.batch_size]
                sentlen = self.dp.train_src_sentlen[i*self.args.batch_size : (i+1)*self.args.batch_size]
                word_pairs = self.dp.train_word_pair[i*self.args.batch_size : (i+1)*self.args.batch_size]
                word_pairs_y = self.dp.train_word_pair_y[i*self.args.batch_size : (i+1)*self.args.batch_size]

                # import pdb; pdb.set_trace();
                src = torch.LongTensor(src).to(self.device).transpose(0, 1)
                tgt = torch.LongTensor(tgt).to(self.device).transpose(0, 1)

                tgtlng = torch.LongTensor(tgtlng).to(self.device)
                srclng = torch.LongTensor(srclng).to(self.device)
                sentlen = torch.LongTensor(sentlen).to(self.device)
                word_pairs = torch.LongTensor(word_pairs).to(self.device)
                word_pairs_y = torch.LongTensor(word_pairs_y).to(self.device)
                # print(src.size())
                self.optimizer.zero_grad()
                self.model.zero_grad()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                # print(src[:, 0])
                # print(tgt[:, 0])
                # print(tgt[:-1, 0])
                # print(tgt[1:, 0])
                output = self.model(src, tgt[:-1, :], tgtlng)

                # sentlen_out, wp_out = self.probe(src, srclng, word_pairs)
                # loss_syn1 = self.criterion2(sentlen_out, sentlen)
                # loss_syn2 = self.criterion2(wp_out, word_pairs_y)

                # loss_syn = loss_syn1 + loss_syn2

                output_ = output.view(-1, output.shape[-1])
                # print(output.size())
                # print(tgt.size())
                # print(tgt[1:, :].size())
                tgt_ = (tgt[1:, :]).contiguous().view(-1)

                loss = self.criterion(output_, tgt_)
                # loss /= (output.size(0)*output.size(1))

                loss_ppl = self.criterion_ppl(output_, tgt_)
                self.train_loss_ppl += loss_ppl.item()
                if self.args.mode == 'train':
                    loss = loss + loss_syn

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                self.train_loss += loss.item()

                # if self.args.mode == 'probe':
                #     self.optimizer2.zero_grad()
                #     loss_syn.backward()
                #     self.optimizer2.step()

                # self.train_sentlen += loss_syn1.item()
                # self.train_wp += loss_syn2.item()

                pred_sents = []
                trg_sents = []
                output = output.transpose(0, 1)
                tgt = tgt.transpose(0, 1)

                for j in range(self.args.batch_size):
                    pred_sent = self.get_sentence(output[j].data.cpu().numpy().argmax(axis=-1).tolist(), 'tgt')
                    trg_sent = self.get_sentence(tgt[j, 1:].data.cpu().numpy().tolist(), 'tgt')
                    pred_sents.append(pred_sent)
                    trg_sents.append(trg_sent)
                bleu_value = get_bleu(pred_sents, trg_sents)
                self.train_bleu += bleu_value

                if i%self.args.log_interval == 0 and i>0:
                    message = "Train epoch: %d  iter: %d  train loss: %1.3f  train_bleu:  %1.3f , train_ppl: %4.3f, train_sentlen: %1.3f, train_wp: %1.3f, elapsed: %1.3f " % (
                    epoch, i, self.train_loss/self.args.log_interval, self.train_bleu/self.args.log_interval, np.exp(self.train_loss_ppl/self.args.log_interval), self.train_sentlen/self.args.log_interval, self.train_wp/self.args.log_interval,time.time() - start_time)
                    print(message)
                    self.train_loss = 0
                    self.train_bleu = 0
                    self.train_loss_ppl = 0
                    self.train_sentlen = 0
                    self.train_wp = 0
                    start_time = time.time()

            val_bleu = self.evaluate()
            if val_bleu > self.best_bleu:
                self.best_bleu = val_bleu
                torch.save(self.model, 'models/model_' + self.args.mode + '.pb')
                patience = 0
            else:
                patience +=1
            # if patience > 3:
            #     break

    def get_sentence(self, sentence, side):
        def _eos_parsing(sentence):
            # if '<EOS>' in sentence:
            #     return sentence[:sentence.index('<EOS>')+1]
            # else:
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
