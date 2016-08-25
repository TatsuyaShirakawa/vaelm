# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import math
import random
import numpy as np

random.seed(0)
np.random.seed(0)

import chainer
from chainer import Variable, functions as F, cuda, optimizers, serializers

from dataset import Vocab, MinibatchFeeder
#from vaelm import VAELM
from vaelm2 import VAELM as VAELM
from util import maybe_create_dir

def parse_args(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train_file', '-t', type=str, required=True,
                        help='Train File (.pack)')
    parser.add_argument('--valid_file', '-v', type=str, required=True,
                        help='Validation File (.pack)')
    parser.add_argument('--test_file', '-T', type=str, required=False, default=None,
                        help='Validation File (.pack)')
    parser.add_argument('--vocab_file', '-V', type=str, required=True,
                        help='Vacab File (.pack)')
    parser.add_argument('--save_dir', '-s', type=str, action='store', default="./save",
                        help='save directory')
    parser.add_argument('--encoding', '-e', type=str, action='store', default='utf-8',
                        help='encoding')
    parser.add_argument('--on_memory', '-o', type=bool, action='store', default=False,
                        help='if this flag is set true, dataset was loaded on memory')
    parser.add_argument('--num_samples', '-n', type=int, action='store', default=1,
                        help='number of samples generated for compute loss')
    parser.add_argument('--word_keep_rate', '-w', type=float, action='store', default=0.8,
                        help='conditioned-on word will be replaced by <UNK> w.p. this rate at decoding phase')
    result = parser.parse_args()
    assert(result.word_keep_rate >= 0 and result.word_keep_rate <= 1)

    return result

args = parse_args(sys.argv)

gpu = args.gpu

hidden_size = 256
z_size = 16
num_layers = 2
num_infer_layers = 1

batch_size = 128

save_every_batches = 250000//batch_size # save model, optimizers every this batches
eval_valid_every_batches = 25000//batch_size # evaluate model on valid data every this batches
eval_train_every_batches = 50000//batch_size # evaluate model on train data every this batches
max_epoch = 10000
max_line_length = 100

train_file = args.train_file
valid_file = args.valid_file
test_file = args.test_file
vocab_file = args.vocab_file
save_dir = args.save_dir

encoding = args.encoding

on_memory = args.on_memory

num_samples = args.num_samples

word_keep_rate = args.word_keep_rate

print( "settings:" )
print( "    gpu                     : {}".format(gpu) )
print( "    hidden_size             : {}".format(hidden_size) )
print( "    z_size                  : {}".format(z_size) )
print( "    num_layers              : {}".format(num_layers) )
print( "    num_infer_layers        : {}".format(num_infer_layers) )
print( "    batch_size              : {}".format(batch_size) )
print( "    save_every_batches      : {}".format(save_every_batches) )
print( "    eval_valid_every_batches: {}".format(eval_valid_every_batches) )
print( "    eval_train_every_batches: {}".format(eval_train_every_batches) )
print( "    max_epoch               : {}".format(max_epoch) )
print( "    max_line_length         : {}".format(max_line_length) )
print( "    train_file              : {}".format(train_file) )
print( "    valid_file              : {}".format(valid_file) )
print( "    test_file               : {}".format(test_file) )
print( "    vocab_file              : {}".format(vocab_file) )
print( "    save_dir                : {}".format(save_dir) )
print( "    encoding                : {}".format(encoding) )
print( "    on_memory               : {}".format(on_memory) )
print( "    num_samples             : {}".format(num_samples) )
print( "    word_keep_rate          : {}".format(word_keep_rate) )
    

if gpu >= 0:
    cuda.get_device(gpu).use()

xp = np if gpu < 0 else cuda.cupy

maybe_create_dir(save_dir)

print(' load vocab from {} ...'.format(vocab_file) )
vocab = Vocab().load_pack(open(vocab_file, 'rb'), encoding=encoding)

vocab_size = len(vocab)
print(' vocab size: {}'.format(vocab_size) )

UNK = vocab.unk_id

train_batches = MinibatchFeeder(open(train_file, 'rb'), 
                                batch_size=batch_size, 
                                max_line_length=max_line_length,
                                on_memory = on_memory)

train_head_batches = MinibatchFeeder(open(train_file, 'rb'), 
                                     batch_size=1, # prevent 'backeting'
                                     max_line_length=max_line_length, 
                                     max_num_lines=100,
#                                     max_num_lines=1000,
                                     on_memory = on_memory)
valid_batches = MinibatchFeeder(open(valid_file, 'rb'), 
                                batch_size=1, # prevent 'backeting'
                                max_line_length=max_line_length,
                                max_num_lines = 100,
                                on_memory = on_memory)

print( "train      : {} lines".format(train_batches.num_epoch_lines) )
print( "train(head): {} lines".format(train_head_batches.num_epoch_lines) )
print( "valid      : {} lines".format(valid_batches.num_epoch_lines) )


ignore_label = vocab.padding_id

model = VAELM(vocab_size, hidden_size, z_size,
              num_layers=num_layers, 
              num_infer_layers=num_infer_layers,
              ignore_label=ignore_label)
if gpu >= 0:
    model.to_gpu()

optimizer = optimizers.Adam() # beta1 = 0.5 may do better 
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(2.))

def forward(model, batch, num_samples, word_keep_rate, UNK, train=True):
    
    batch_size = batch.shape[0]

    xp = model.xp
    use_gpu = (xp == cuda.cupy)
    if use_gpu:
        batch = cuda.to_gpu(batch)

    model.reset_state()
    model.zerograds()

    # encode
    batch_length = len(batch[0])-1
    for i in range(batch_length):
        w = Variable(batch[:, i])
        model.encode(w, train=train)
    
    # infer q(z|x)
    model.infer(train=train)

    # compute KL
    KL = 0
    for i in range(model.num_layers):
        # h
        mu, sigma = model.hmus[i], model.hsigmas[i]
        KL += -F.sum((1 + 2 * F.log(sigma) - sigma*sigma - mu*mu) / 2)

        # c
        mu, sigma = model.cmus[i], model.csigmas[i]
        KL += -F.sum((1 + 2 * F.log(sigma) - sigma*sigma - mu*mu) / 2)

    KL /= batch_size
    # draw and decode
    cross_entropies = []
    if not train:
        ys, ts = [], []

    UNKs = np.array([UNK for _ in range(batch_size)], dtype=np.int32)
    if use_gpu:
        UNKs = cuda.to_gpu(UNKs)
    for _ in range(num_samples):

        cross_entropies.append(0)
        if not train:
            ys.append([])
            ts.append([])

        if train == True:
            model.set_by_sample(train=train)
        else:
            model.set_by_MLE(train=train)

        last_w = None
        for i in range(batch_length):
            w, next_w = Variable(batch[:, i]), Variable(batch[:, i+1])
            # word dropout
            masked_w = batch[:, i]
            if np.random.uniform() > word_keep_rate:
                enable = (masked_w != -1)
                masked_w = F.where(enable, masked_w, UNKs)
            y = model.decode(masked_w, train=train)
            cross_entropies[-1] += F.softmax_cross_entropy(y, next_w)
            if not train:
                ys[-1].append(xp.argmax(y.data, axis=1))
                ts[-1].append(next_w.data)
            last_w = next_w

        if not train:
            ys[-1] = xp.vstack(ys[-1]).T
            ts[-1] = xp.vstack(ts[-1]).T
            if use_gpu:
                ys[-1] = cuda.to_cpu(ys[-1])
                ts[-1] = cuda.to_cpu(ts[-1])

    if train:
        return (KL, cross_entropies)
    else:
        assert(len(cross_entropies) == 1 and len(ys) == 1 and len(ts) == 1)
        return (KL, (cross_entropies, ys, ts))


def evaluate(model, batches, vocab, word_keep_rate, UNK, alpha):

    xp = model.xp
    use_gpu = (xp == cuda.cupy)

    ignore_label = vocab.padding_id

    KL = 0
    xent, ys, ts = 0, [], []    

    sum_sentence_length_minus_1 = 0
    num_batches = 0
    for batch in batches:
        num_batches += 1
        assert( len(batch) == 1 )
        cur_sentence_length = (batch != ignore_label).sum(axis=1).max()
        sum_sentence_length_minus_1 += cur_sentence_length - 1
        cur_KL, (cur_xents, cur_ys, cur_ts) = forward(model, batch, num_samples=1, word_keep_rate=word_keep_rate, UNK=UNK, train=False)
        assert(cuda.to_cpu(cur_KL.data) >= 0)
        assert(all(cuda.to_cpu(_.data) for _ in cur_xents))
        assert(len(cur_xents) == 1 and len(cur_ys) == 1 and len(cur_ts) == 1)
        cur_xent = cur_xents[0]
        cur_ys = cur_ys[0]
        cur_ts = cur_ts[0]

        cur_KL.unchain_backward()
        cur_xent.unchain_backward()        

        KL += cur_KL.data
        xent += cur_xent.data
        ys.extend(cur_ys)
        ts.extend(cur_ts)

    if use_gpu:
        KL = cuda.to_cpu(KL)
        xent = cuda.to_cpu(xent)

    KL /= num_batches
    xent_per_word = xent / sum_sentence_length_minus_1

    n = len(ys) // 10 
    if n > 0:
        ys = [ys[i*n] for i in range(10)]
        ts = [ts[i*n] for i in range(10)]

    assert( len(ys) == len(ts) )

    for i in range(min(len(ys), 10)):
        assert( ys[i].shape == ts[i].shape )
        length = len(ts[i])
        print( "actual:" )
        print( " ".join([vocab.get_word(ts[i][j]).encode('utf-8') for j in range(length)]) )
        print( "decode:" )
        print( " ".join([vocab.get_word(ys[i][j]).encode('utf-8') for j in range(length)]) )
        print( " ".join([[".", "x"][ ts[i][j] != -1 and ts[i][j] != ys[i][j] ] for j in range(length)]) )
        print()

    print( "average length    : {}".format( (sum_sentence_length_minus_1 + 1) / num_batches) )
    print( "loss              : {} (alpha = {})".format( alpha*KL + xent/num_batches, alpha ) )
    print( "KL divergence     : {} (alpha = {})".format( KL, alpha ) )
    print( "xentropy          : {}".format( xent/num_batches ) )
    print( "xentropy / word   : {}".format( xent_per_word ) )
    print( "perplexity / word : {}".format( math.pow(2, math.log(math.e, 2) * xent_per_word) ) )
    
def train(model, batch, num_samples, word_keep_rate, UNK, alpha):

    xp = model.xp
    use_gpu = (xp == cuda.cupy)

    if use_gpu:
        batch = cuda.to_gpu(batch)

    KL, xents = forward(model, batch, num_samples=num_samples, word_keep_rate=word_keep_rate, UNK=UNK, train=True)
    loss = alpha * KL + sum(xents) / num_samples
    loss.backward()
    optimizer.update()
    loss.unchain_backward()
    if alpha == 0: KL.unchain_backward()

def save_hdf5(filename, obj):
    gpu = (hasattr(obj, "xp") and obj.xp == cuda.cupy)
    if gpu: obj.to_cpu()
    serializers.save_hdf5(filename, obj)
    if gpu: obj.to_gpu()


next_save_batch = save_every_batches
next_eval_valid_batch = 0 # eval initial model
next_eval_train_batch = 0 # eval initial model
num_saved = 0
num_trained_sentences = 0
num_trained_batches = 0

alpha = 0.0
for epoch in range(max_epoch):

    print( "epoch {}/{}".format( epoch + 1, max_epoch ) )

    for batch in train_batches:

        if num_trained_batches == next_save_batch:
            print( "saving model and optimizer ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            prefix = '{}_{}_{}'.format(epoch+1, num_saved+1, num_trained_sentences)

            model_file = os.path.join(save_dir, prefix + '.model.hdf5')
            print( "save model to {} ...".format(model_file) )
            save_hdf5(model_file, model)

            optimizer_file = os.path.join(save_dir, prefix + '.optimizer.hdf5')
            print( "save optimizer to {} ...".format(optimizer_file) )
            save_hdf5(optimizer_file, optimizer)

            next_save_batch += save_every_batches
            num_saved += 1
            
        if num_trained_batches == next_eval_valid_batch:

            print( "eval on validation dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            evaluate(model, valid_batches, vocab, word_keep_rate=word_keep_rate, UNK=UNK, alpha=alpha)
            print()

            next_eval_valid_batch += eval_valid_every_batches

        if num_trained_batches == next_eval_train_batch:

            print( "eval on training dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
            evaluate(model, train_head_batches, vocab, word_keep_rate=word_keep_rate, UNK=UNK, alpha=alpha)
            print()

            next_eval_train_batch += eval_train_every_batches

        train(model, batch, num_samples=num_samples, word_keep_rate=word_keep_rate, UNK=UNK, alpha=alpha)

        num_trained_batches += 1
        num_trained_sentences += len(batch.data)

        if (num_trained_batches * batch_size)  % 2000 == 0:
            prev_alpha = alpha
            alpha = min(alpha + 0.01, 1.0)
            if prev_alpha != alpha:
                print("alpha {} -> {}".format(prev_alpha, alpha))

        
print( "saving model and optimizer (last) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )

model_file = os.path.join(save_dir, 'model.hdf5')
print( "save model to {} ...".format(model_file) )
save_hdf5(model_file, model)

optimizer_file = os.path.join(save_dir, 'optimizer.hdf5')
print( "save optimizer to {} ...".format(optimizer_file) )
save_hdf5(optimizer_file, optimizer)

print( "eval on validation dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
evaluate(model, valid_batches, word_keep_rate=word_keep_rate, UNK=UNK, alpha=1.0)
print()

print( "eval on training dataset ({}/{}) ...".format(num_trained_batches, train_batches.num_epoch_batches ) )
evaluate(model, train_head_batches, word_keep_rate=word_keep_rate, UNK=UNK, alpha=1.0)
print()        
