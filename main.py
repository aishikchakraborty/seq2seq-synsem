import argparse
import os
import time
from torch.backends import cudnn

from data import *
from trainer import *
import _pickle as pickle

def main(args):
	cuda.set_device(int(args.gpu_num))
	cudnn.benchmark = True

	start_time = time.time()

	# if os.path.isfile(args.data_path + 'vocab.pb'):
	# 	vocab = pickle.load(open(args.data_path, 'rb'))
	# else:
	# vocab = Vocab()
	# vocab.create_vocab_for_translation(args.train_path + 'train_all')
	# pickle.dump(vocab, open(args.data_path + 'vocab.pb', 'wb'))


	# Load dataset
	if os.path.isfile(args.data_path + 'prepro.pb'):
		dp = pickle.load(open(args.data_path + 'prepro.pb', 'rb'))
	else:
		dp = Preprocess(args.train_path + 'train.src', args.train_path + 'train.tgt', args.val_path + 'val.src', args.val_path + 'val.tgt', args.train_path + 'train_all')
		dp.preprocess_train()
		dp.preprocess_val()
		pickle.dump(dp, open(args.data_path + 'prepro.pb', 'wb'))


	print ("Elapsed Time: %1.3f \n"  %(time.time() - start_time))

	# print(dp.vocab.itos)
	# print(dp.train_src[0])
	# print(dp.train_tgt[0])
	# training_dataset = list(zip(dp.train_src, dp.train_tgt, dp.train_srclng, dp.train_tgtlng))
	# import random; random.seed(1234);
	# random.shuffle(training_dataset)
	MAX_SAMPLES = 40040
	# dp.train_src[:], dp.train_tgt[:], dp.train_srclng[:], dp.train_tgtlng[:] = zip(*training_dataset)
	dp.train_src, dp.train_tgt, dp.train_srclng, dp.train_tgtlng = dp.train_src[:MAX_SAMPLES], dp.train_tgt[:MAX_SAMPLES], dp.train_srclng[:MAX_SAMPLES], dp.train_tgtlng[:MAX_SAMPLES]

	print(dp.vocab.itos)
	print(dp.train_src[0])
	print(dp.train_tgt[0])
	print ("=========== Data Stat ===========")
	print ("Train: ", len(dp.train_src))
	print ("val: ", len(dp.val_tgt))
	print ("val: ", len(set(dp.train_src_sentlen)))
	print ("=================================")


	trainer = Trainer(dp, args)
	trainer.train()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# Language setting
	parser.add_argument('--dataset', type=str, default='europarl')
	parser.add_argument('--src-lang', action="append", type=str, default=[], dest='src_lang',
	                    help='list of type of lexical relations to capture. Options | syn | hyp | mer')
	parser.add_argument('--trg_lang', type=str, default='en')
	parser.add_argument('--mode', type=str, default='probe', help='probe|train')
	parser.add_argument('--max_len', type=int, default=20)
	parser.add_argument('--cuda', action='store_true',
	                    help='use CUDA')

	parser.add_argument('--gpu', type=int, default=0,
	                    help='use gpu x')
	parser.add_argument('--log-interval', type=int, default=100)
	# Model hyper-parameters
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--grad_clip', type=float, default=1)
	parser.add_argument('--num_layer', type=int, default=2)
	parser.add_argument('--embed_dim',  type=int, default=32)
	parser.add_argument('--hidden_dim', type=int, default=128)

	# Training setting
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_epoch', type=int, default=100)

	# Path
	parser.add_argument('--data_path', type=str, default='./data/')
	parser.add_argument('--train_path', type=str, default='./data/training/')
	parser.add_argument('--val_path', type=str, default='./data/dev/')

	# Dir.
	parser.add_argument('--log', type=str, default='log')
	parser.add_argument('--sample', type=str, default='sample')

	# Misc.
	parser.add_argument('--gpu_num', type=int, default=0)

	args = parser.parse_args()
	print (args)
	main(args)
