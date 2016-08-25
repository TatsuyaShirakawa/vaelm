# -*- coding:utf-8 -*-

import os
import sys
import codecs
import argparse

from dataset import create_vocab, encode_and_pack
from util import maybe_create_dir

def parse_args(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--train_file', '-t', type=str, required=True, help='training text file')
    parser.add_argument('--valid_file', '-v', type=str, required=False, default=None, help='validation text file')
    parser.add_argument('--test_file', '-T', type=str, required=False, default=None, help='testing text file')        
    parser.add_argument('--save_dir', '-s', type=str, required=True, help='savedir')
    parser.add_argument('--min_count', '-m', type=int, required=False, default=0, help='min count')
    parser.add_argument('--max_vocab', '-M', type=int, required=False, default=-1, help='max vocab')
    parser.add_argument('--encoding', '-e', type=str, required=False, default='utf-8', help='encoding')
    result = parser.parse_args()
    assert( result.min_count >= 0 )
    return result

args = parse_args(sys.argv)
train_file = args.train_file
valid_file = args.valid_file
test_file = args.test_file
save_dir = args.save_dir

min_count = args.min_count
max_vocab = args.max_vocab
encoding = args.encoding

maybe_create_dir(os.path.dirname(train_file))

print( "collect vocabulary ..." )
vocab = create_vocab([codecs.open(train_file, encoding=encoding)],
                     min_count=min_count, max_vocab=max_vocab)

vocab_pack_file = os.path.join(save_dir, 'vocab.pack')
print( "save vocab to {} ...".format(vocab_pack_file) )
vocab.save_pack(open(vocab_pack_file, 'wb'))

train_pack_file = os.path.join(os.path.dirname(train_file), 'train.pack')
print( "encode and pack train file {} to {} ...".format(train_file, train_pack_file) )
encode_and_pack(vocab, fin=codecs.open(train_file, 'r', encoding=encoding),
                fout=open(train_pack_file, 'wb'))

if valid_file:
    valid_pack_file = os.path.join(os.path.dirname(valid_file), 'valid.pack')
    print( "encode and pack valid file {} to {} ...".format(valid_file, valid_pack_file) )
    encode_and_pack(vocab, fin=codecs.open(valid_file, 'r', encoding=encoding),
                    fout=open(valid_pack_file, 'wb'))

if test_file:
    test_pack_file = os.path.join(os.path.dirname(test_file), 'test.pack')
    print( "encode and pack test file {} to {} ...".format(test_file, test_pack_file) )
    encode_and_pack(vocab, fin=codecs.open(test_file, 'r', encoding=encoding),
                    fout=open(test_pack_file, 'wb'))



