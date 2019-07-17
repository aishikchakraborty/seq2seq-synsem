import numpy as np
import torch
from data import *

import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import model
# we will use CUDA if it is available
USE_CUDA = True
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if os.path.isfile('data/prepro.pb'):
    dp = pickle.load(open('data/prepro.pb', 'rb'))
else:
    dp = Preprocess(args.train_path + 'train.src', args.train_path + 'train.tgt', args.val_path + 'val.src', args.val_path + 'val.tgt', args.train_path + 'train_all')
    dp.preprocess_train()
    dp.preprocess_val()
    pickle.dump(dp, open(args.data_path + 'prepro.pb', 'wb'))

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = model.BahdanauAttention(hidden_size)

    mdl = model.EncoderDecoder(
        model.Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        model.Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        model.Generator(hidden_size, tgt_vocab))

    return mdl.cuda() if USE_CUDA else mdl


# # Training
#
# This section describes the training regime for our models.

# We stop for a quick interlude to introduce some of the tools
# needed to train a standard encoder decoder model. First we define a batch object that holds the src and target sentences for training, as well as their lengths and masks.

# ## Batches and Masking

# In[9]:


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg
            self.trg_lengths = trg_lengths
            self.trg_y = trg
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()



# ## Training Loop
# The code below trains the model for 1 epoch (=1 pass through the training data).

# In[10]:


def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):

        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


# ## Training Data and Batching
#
# We will use torch text for batching. This is discussed in more detail below.

# ## Optimizer
#
# We will use the [Adam optimizer](https://arxiv.org/abs/1412.6980) with default settings ($\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon=10^{-8}$).
#
# We will use $0.0003$ as the learning rate here, but for different problems another learning rate may be more appropriate. You will have to tune that.

# # A First  Example
#
# We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.

# ## Synthetic Data

# In[11]:

def pad_sequences(s, dp, batch_size):
    pad_token = dp.vocab.stoi['<PAD>']
    # print(s)
    lengths = torch.LongTensor([len(s1) for s1 in s])
    longest_sent = max(lengths)
    padded_X = np.ones((batch_size, longest_sent), dtype=np.int64) * pad_token
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    # print(seq_lengths)
    for i, x_len in enumerate(lengths):
        sequence = s[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
    if batch_size != 1:
        padded_X = padded_X[perm_idx]
    return padded_X, seq_lengths.data.cpu().numpy().tolist()

def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=2, sos_index=1, mode='train'):
    """Generate random data for a src-tgt copy task."""
    # training_dataset = list(zip(dp.train_src, dp.train_tgt, dp.train_srclng, dp.train_tgtlng))
    training_dataset = list(zip(dp.train_src, dp.train_tgt, dp.train_srclng, dp.train_tgtlng))
    import random; random.seed(1234);
    random.shuffle(training_dataset)
    MAX_SAMPLES = 40040
    dp.train_src, dp.src_tgt, dp.train_srclng, dp.train_tgtlng = zip(*training_dataset)
    dp.train_src, dp.src_tgt, dp.train_srclng, dp.train_tgtlng = dp.train_src[:MAX_SAMPLES], dp.src_tgt[:MAX_SAMPLES], dp.train_srclng[:MAX_SAMPLES], dp.train_tgtlng[:MAX_SAMPLES]

    if mode == 'train':
        len_src = [len(w) for w in dp.train_src]
        len_tgt = [len(w) for w in dp.train_tgt]

        num_batches = len(dp.train_src)//batch_size
        for i in range(num_batches):
            src, src_lengths = pad_sequences(dp.train_src[i*batch_size : (i+1)*batch_size], dp, batch_size)
            tgt, tgt_lengths = pad_sequences(dp.train_tgt[i*batch_size : (i+1)*batch_size], dp, batch_size)

            # src_lengths = len_src[i*batch_size : (i+1)*batch_size]
            # tgt_lengths = len_tgt[i*batch_size : (i+1)*batch_size]

            # print(src_lengths)
            src = torch.LongTensor(src).to(DEVICE)
            tgt = torch.LongTensor(tgt).to(DEVICE)
            # src_lengths = torch.LongTensor(src_lengths)
            # tgt_lengths = torch.LongTensor(tgt_lengths)

            yield Batch((src, src_lengths), (tgt, tgt_lengths), pad_index=pad_index)
    else:
        len_src = [len(w) for w in dp.val_src]
        len_tgt = [len(w) for w in dp.val_tgt]

        num_batches = len(dp.val_src)//batch_size
        for i in range(num_batches):
            src, src_lengths = pad_sequences(dp.val_src[i*batch_size : (i+1)*batch_size], dp, batch_size)
            tgt, tgt_lengths = pad_sequences(dp.val_tgt[i*batch_size : (i+1)*batch_size], dp, batch_size)

            # src_lengths = len_src[i*batch_size : (i+1)*batch_size]
            # tgt_lengths = len_tgt[i*batch_size : (i+1)*batch_size]

            # print(src_lengths)
            src = torch.LongTensor(src).to(DEVICE)
            tgt = torch.LongTensor(tgt).to(DEVICE)
            # src_lengths = torch.LongTensor(src_lengths)
            # tgt_lengths = torch.LongTensor(tgt_lengths)

            yield Batch((src, src_lengths), (tgt, tgt_lengths), pad_index=pad_index)

# ## Loss Computation

# In[12]:


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


# ### Printing examples
#
# To monitor progress during training, we will translate a few examples.
#
# We use greedy decoding for simplicity; that is, at each time step, starting at the first token, we choose the one with that maximum probability, and we never revisit that choice.

# In[13]:


def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


# In[14]:


def print_examples(example_iter, model, n=2, max_len=100,
                   sos_index=1,
                   src_eos_index=None,
                   trg_eos_index=None,
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = dp.vocab.stoi[EOS_TOKEN]
        trg_sos_index = dp.vocab.stoi[SOS_TOKEN]
        trg_eos_index = dp.vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = 0
        trg_sos_index = 0
        trg_eos_index = 1

    for i, batch in enumerate(example_iter):

        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg

        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=dp.vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=dp.vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=dp.vocab)))
        print()

        count += 1
        if count == n:
            break


# ## Training the copy task

# In[15]:


def train_copy_task():
    """Train the simple copy task."""
    num_words = len(dp.vocab.itos)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100, mode='eval'))

    dev_perplexities = []

    if USE_CUDA:
        model.cuda()

    for epoch in range(10):

        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad():
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=50)

    return dev_perplexities


# In[16]:


# train the copy task
dev_perplexities = train_copy_task()

# def plot_perplexity(perplexities):
#     """plot perplexities"""
#     plt.title("Perplexity per Epoch")
#     plt.xlabel("Epoch")
#     plt.ylabel("Perplexity")
#     plt.plot(perplexities)
#
# plot_perplexity(dev_perplexities)
#
#
# # You can see that the model managed to correctly 'translate' the two examples in the end.
# #
# # Moreover, the perplexity of the development data nicely went down towards 1.
#
# # # A Real World Example
# #
# # Now we consider a real-world example using the IWSLT German-English Translation task.
# # This task is much smaller than usual, but it illustrates the whole system.
# #
# # The cell below installs torch text and spacy. This might take a while.
#
# # In[17]:
#
#
# #!pip install git+git://github.com/pytorch/text spacy
# #!python -m spacy download en
# #!python -m spacy download de
#
#
# # ## Data Loading
# #
# # We will load the dataset using torchtext and spacy for tokenization.
# #
# # This cell might take a while to run the first time, as it will download and tokenize the IWSLT data.
# #
# # For speed we only include short sentences, and we include a word in the vocabulary only if it occurs at least 5 times. In this case we also lowercase the data.
# #
# # If you have **issues** with torch text in the cell below (e.g. an `ascii` error), try running `export LC_ALL="en_US.UTF-8"` before you start `jupyter notebook`.
#
# # In[18]:
#
#
# # For data loading.
# from torchtext import data, datasets
#
# if True:
#     import spacy
#     spacy_de = spacy.load('de')
#     spacy_en = spacy.load('en')
#
#     def tokenize_de(text):
#         return [tok.text for tok in spacy_de.tokenizer(text)]
#
#     def tokenize_en(text):
#         return [tok.text for tok in spacy_en.tokenizer(text)]
#
#     UNK_TOKEN = "<unk>"
#     PAD_TOKEN = "<pad>"
#     SOS_TOKEN = "<s>"
#     EOS_TOKEN = "</s>"
#     LOWER = True
#
#     # we include lengths to provide to the RNNs
#     SRC = data.Field(tokenize=tokenize_de,
#                      batch_first=True, lower=LOWER, include_lengths=True,
#                      unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
#     TRG = data.Field(tokenize=tokenize_en,
#                      batch_first=True, lower=LOWER, include_lengths=True,
#                      unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
#
#     MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
#     train_data, valid_data, test_data = datasets.IWSLT.splits(
#         exts=('.de', '.en'), fields=(SRC, TRG),
#         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
#             len(vars(x)['trg']) <= MAX_LEN)
#     MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
#     SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
#     TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)
#
#     PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]
#
#
# # ### Let's look at the data
# #
# # It never hurts to look at your data and some statistics.
#
# # In[19]:
#
#
# def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
#     """ This prints some useful stuff about our data sets. """
#
#     print("Data set sizes (number of sentence pairs):")
#     print('train', len(train_data))
#     print('valid', len(valid_data))
#     print('test', len(test_data), "\n")
#
#     print("First training example:")
#     print("src:", " ".join(vars(train_data[0])['src']))
#     print("trg:", " ".join(vars(train_data[0])['trg']), "\n")
#
#     print("Most common words (src):")
#     print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
#     print("Most common words (trg):")
#     print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")
#
#     print("First 10 words (src):")
#     print("\n".join(
#         '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
#     print("First 10 words (trg):")
#     print("\n".join(
#         '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")
#
#     print("Number of German words (types):", len(src_field.vocab))
#     print("Number of English words (types):", len(trg_field.vocab), "\n")
#
#
# print_data_info(train_data, valid_data, test_data, SRC, TRG)
#
#
# # ## Iterators
# # Batching matters a ton for speed. We will use torch text's BucketIterator here to get batches containing sentences of (almost) the same length.
# #
# # #### Note on sorting batches for RNNs in PyTorch
# #
# # For effiency reasons, PyTorch RNNs require that batches have been sorted by length, with the longest sentence in the batch first. For training, we simply sort each batch.
# # For validation, we would run into trouble if we want to compare our translations with some external file that was not sorted. Therefore we simply set the validation batch size to 1, so that we can keep it in the original order.
#
# # In[20]:
#
#
# train_iter = data.BucketIterator(train_data, batch_size=64, train=True,
#                                  sort_within_batch=True,
#                                  sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
#                                  device=DEVICE)
# valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False,
#                            device=DEVICE)
#
#
# def rebatch(pad_idx, batch):
#     """Wrap torchtext batch into our own Batch class for pre-processing"""
#     return Batch(batch.src, batch.trg, pad_idx)
#
#
# # ## Training the System
# #
# # Now we train the model.
# #
# # On a Titan X GPU, this runs at ~18,000 tokens per second with a batch size of 64.
#
# # In[21]:
#
#
# def train(model, num_epochs=10, lr=0.0003, print_every=100):
#     """Train a model on IWSLT"""
#
#     if USE_CUDA:
#         model.cuda()
#
#     # optionally add label smoothing; see the Annotated Transformer
#     criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#
#     dev_perplexities = []
#
#     for epoch in range(num_epochs):
#
#         print("Epoch", epoch)
#         model.train()
#         train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter),
#                                      model,
#                                      SimpleLossCompute(model.generator, criterion, optim),
#                                      print_every=print_every)
#
#         model.eval()
#         with torch.no_grad():
#             print_examples((rebatch(PAD_INDEX, x) for x in valid_iter),
#                            model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)
#
#             dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter),
#                                        model,
#                                        SimpleLossCompute(model.generator, criterion, None))
#             print("Validation perplexity: %f" % dev_perplexity)
#             dev_perplexities.append(dev_perplexity)
#
#     return dev_perplexities
#
#
#
# # In[22]:
#
#
# model = make_model(len(SRC.vocab), len(TRG.vocab),
#                    emb_size=256, hidden_size=256,
#                    num_layers=1, dropout=0.2)
# dev_perplexities = train(model, print_every=100)
#
#
# # In[23]:
#
#
# plot_perplexity(dev_perplexities)
#
#
# # ## Prediction and Evaluation
# #
# # Once trained we can use the model to produce a set of translations.
# #
# # If we translate the whole validation set, we can use [SacreBLEU](https://github.com/mjpost/sacreBLEU) to get a [BLEU score](https://en.wikipedia.org/wiki/BLEU), which is the most common way to evaluate translations.
# #
# # #### Important sidenote
# # Typically you would use SacreBLEU from the **command line** using the output file and original (possibly tokenized) development reference file. This will give you a nice version string that shows how the BLEU score was calculated; for example, if it was lowercased, if it was tokenized (and how), and what smoothing was used. If you want to learn more about how BLEU scores are (and should be) reported, check out [this paper](https://arxiv.org/abs/1804.08771).
# #
# # However, right now our pre-processed data is only in memory, so we'll calculate the BLEU score right from this notebook for demonstration purposes.
# #
# # We'll first test the raw BLEU function:
#
# # In[24]:
#
#
# import sacrebleu
#
#
# # In[25]:
#
#
# # this should result in a perfect BLEU of 100%
# hypotheses = ["this is a test"]
# references = ["this is a test"]
# bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
# print(bleu)
#
#
# # In[26]:
#
#
# # here the BLEU score will be lower, because some n-grams won't match
# hypotheses = ["this is a test"]
# references = ["this is a fest"]
# bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
# print(bleu)
#
#
# # Since we did some filtering for speed, our validation set contains 690 sentences.
# # The references are the tokenized versions, but they should not contain out-of-vocabulary UNKs that our network might have seen. So we'll take the references straight out of the `valid_data` object:
#
# # In[27]:
#
#
# len(valid_data)
#
#
# # In[28]:
#
#
# references = [" ".join(example.trg) for example in valid_data]
# print(len(references))
# print(references[0])
#
#
# # In[29]:
#
#
# references[-2]
#
#
# # **Now we translate the validation set!**
# #
# # This might take a little bit of time.
# #
# # Note that `greedy_decode` will cut-off the sentence when it encounters the end-of-sequence symbol, if we provide it the index of that symbol.
#
# # In[30]:
#
#
# hypotheses = []
# alphas = []  # save the last attention scores
# for batch in valid_iter:
#   batch = rebatch(PAD_INDEX, batch)
#   pred, attention = greedy_decode(
#     model, batch.src, batch.src_mask, batch.src_lengths, max_len=25,
#     sos_index=TRG.vocab.stoi[SOS_TOKEN],
#     eos_index=TRG.vocab.stoi[EOS_TOKEN])
#   hypotheses.append(pred)
#   alphas.append(attention)
#
#
# # In[31]:
#
#
# # we will still need to convert the indices to actual words!
# hypotheses[0]
#
#
# # In[32]:
#
#
# hypotheses = [lookup_words(x, TRG.vocab) for x in hypotheses]
# hypotheses[0]
#
#
# # In[33]:
#
#
# # finally, the SacreBLEU raw scorer requires string input, so we convert the lists to strings
# hypotheses = [" ".join(x) for x in hypotheses]
# print(len(hypotheses))
# print(hypotheses[0])
#
#
# # In[34]:
#
#
# # now we can compute the BLEU score!
# bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references], .01).score
# print(bleu)
#
#
# # ## Attention Visualization
# #
# # We can also visualize the attention scores of the decoder.
#
# # In[47]:
#
#
# def plot_heatmap(src, trg, scores):
#
#     fig, ax = plt.subplots()
#     heatmap = ax.pcolor(scores, cmap='viridis')
#
#     ax.set_xticklabels(trg, minor=False, rotation='vertical')
#     ax.set_yticklabels(src, minor=False)
#
#     # put the major ticks at the middle of each cell
#     # and the x-ticks on top
#     ax.xaxis.tick_top()
#     ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
#     ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
#     ax.invert_yaxis()
#
#     plt.colorbar(heatmap)
#     plt.show()
#
#
# # In[71]:
#
#
# # This plots a chosen sentence, for which we saved the attention scores above.
# idx = 5
# src = valid_data[idx].src + ["</s>"]
# trg = valid_data[idx].trg + ["</s>"]
# pred = hypotheses[idx].split() + ["</s>"]
# pred_att = alphas[idx][0].T[:, :len(pred)]
# print("src", src)
# print("ref", trg)
# print("pred", pred)
# plot_heatmap(src, pred, pred_att)
#
#
# # # Congratulations! You've finished this notebook.
# #
# # What didn't we cover?
# #
# # - Subwords / Byte Pair Encoding [[paper]](https://arxiv.org/abs/1508.07909) [[github]](https://github.com/rsennrich/subword-nmt) let you deal with unknown words.
# # - You can implement a [multiplicative/bilinear attention mechanism](https://arxiv.org/abs/1508.04025) instead of the additive one used here.
# # - We used greedy decoding here to get translations, but you can get better results with beam search.
# # - The original model only uses a single dropout layer (in the decoder), but you can experiment with adding more dropout layers, for example on the word embeddings and the source word representations.
# # - You can experiment with multiple encoder/decoder layers.- Experiment with a benchmarked and improved codebase: [Joey NMT](https://github.com/joeynmt/joeynmt)
#
# # If this was useful to your research, please consider citing:
# #
# # > Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. https://bastings.github.io/annotated_encoder_decoder/
# #
# # Or use the following `Bibtex`:
# # ```
# # @misc{bastings2018annotated,
# #   title={The Annotated Encoder-Decoder with Attention},
# #   author={Bastings, Joost},
# #   journal={https://bastings.github.io/annotated\_encoder\_decoder/},
# #   year={2018}
# # }```
