# -*- coding:utf-8 -*-

from __future__ import print_function, division

import collections
import sys

import nltk
from nltk.tokenize import word_tokenize

import numpy as np

import msgpack

def sepline(line):
    return word_tokenize(line.strip().lower())

class Vocab(object):

    PADDING_ID = -1

    def __init__(self, padding='<pad>', sos='<sos>', eos='<eos>', unk='<unk>'):

        self.__padding_id = self.PADDING_ID
        self.__padding_word = padding

        self.__sos_word = sos
        self.__eos_word = eos
        self.__unk_word = unk
        
        self._init()

        
    def _init(self):
        self.__word2id = {}
        self.__id2word = {}
        self.__id2count = {}
        self.__set(id=0, word='<sos>', count=0)
        self.__set(id=1, word='<eos>', count=0)
        self.__set(id=2, word='<unk>', count=0)                
        

    def __len__(self):
        return len(self.__word2id)

    def entry(self, word, count=1):
        id = self.__word2id.get(word, len(self.__word2id))
        self.__set(id=id, word=word, count=self.__id2count.get(id, 0) + count)

    def __set(self, id, word, count):
        self.__word2id[word] = id
        self.__id2word[id] = word
        self.__id2count[id] = count

    @property
    def padding_word(self):
        return self.__padding_word

    @property
    def padding_id(self):
        return self.__padding_id

    @property
    def sos_word(self):
        return self.__sos_word

    @property
    def sos_id(self):
        return self.get_id(self.sos_word)
    
    @property
    def eos_word(self):
        return self.__eos_word

    @property
    def eos_id(self):
        return self.get_id(self.eos_word)
    
    @property
    def unk_word(self):
        return self.__unk_word

    @property
    def unk_id(self):
        return self.get_id(self.unk_word)
    
    def get_id(self, word):
        if word == self.__padding_word:
            return self.__padding_id
        elif word not in self.__word2id:
            return self.__word2id[self.unk_word]
        return self.__word2id[word] if word != self.__padding_word else self.__padding_id

    def get_word(self, id):
        return self.__id2word[id] if id != self.__padding_id else self.__padding_word

    def get_count(self, id):
        assert( id >= 0 and id <= len(self.__id2word) )
        return self.__id2count[id]

    def items(self):
        for id, word in self.__id2word.items():
            yield (id, word, self.__id2count[id])
        raise StopIteration

    def load_pack(self, fin, encoding='utf-8'):
        self._init()
        unpacker = msgpack.Unpacker()
        BUFSIZE = 1024 * 1024
        while True:
            buf = fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for id, word, count in unpacker:
                word = word.decode(encoding)
                self.__set(id=id, word=word, count=count)
        return self

    def save_pack(self, fout, encoding='utf-8'):
        packer = msgpack.Packer()
        for id, word, count in sorted(self.items()):
            fout.write(packer.pack((id, word.encode(encoding), count)))


def create_vocab(fins, min_count=0, max_vocab=None, sepline=sepline):

    result = Vocab()
    SOS = '<sos>'
    EOS = '<eos>'

    if max_vocab == None or max_vocab < 0:
        max_vocab = float('+inf')

    # count words
    print( "word couting ..." )
    counter = collections.Counter()
    num_lines = 0
    for fin in fins:
        for line in fin:
            words = sepline(line)
            counter[result.sos_word] += 1
            for word in words:
                counter[word] += 1
            counter[result.eos_word] += 1

            num_lines += 1
            if num_lines % 1000 == 0:
                print( "\rreaded {} lines.".format(num_lines), end="" )
                sys.stdout.flush()

    print( "\rreaded {} lines.".format(num_lines))
    print( "{} words.".format(sum(counter.values())) )
    sys.stdout.flush()

    vocab_size_before = len(counter) + 1 # added unk

    print( "reconstructing ...")
    
    # replace unfrequent words with UNK
    for word, count in counter.items():
        if count < min_count and word not in (result.sos_word, result.eos_word, result.unk_word):
            del counter[word]
            counter[result.unk_word] += count

    if max_vocab != None:
        # merge disfrequent words (frequency order less than max_vocab) to UNK
        for i, (word, count) in enumerate(sorted(counter.items(), key=lambda x : x[1], reverse=True)):  
            if i >= max_vocab and word not in (result.sos_word, result.eos_word, result.unk_word):
                del counter[word]
                counter[result.unk_word] += count

    result._init()
    
    for i, (word, count) in enumerate(sorted(counter.items(), key=lambda x : x[1], reverse=True)):
        result.entry(word, count=count)

    vocab_size_after = len(result)
    print("reducing vocab {} -> {} (min_count={}, max_vocab={})".format(vocab_size_before,
                                                                        vocab_size_after,
                                                                        min_count,
                                                                        max_vocab))
    return result

def encode_and_pack(vocab, fin, fout, input_sepline=sepline):

    packer = msgpack.Packer()
    for line in fin:
        words = input_sepline(line)
        encoded = [vocab.sos_id]
        encoded.extend([vocab.get_id(word) for word in words])
        encoded.append(vocab.eos_id)
        fout.write(packer.pack(encoded))


def pad(sentence, length, padding=Vocab.PADDING_ID):
    return sentence + [padding] * (length - len(sentence))
        
def as_mat(batch):
    max_length = max(len(words) for words in batch)
    result = np.vstack( [ pad(words, max_length) for words in batch] ).astype(np.int32)
    return result

def as_mat(batch):
    max_length = max(len(words) for words in batch)
    result = np.vstack( [ pad(words, max_length) for words in batch] ).astype(np.int32)
    return result

class MinibatchFeeder(object):

    def __init__(self, fin, batch_size, sepline = sepline, 
                 max_num_lines=None, max_line_length=100,
                 feed_callback=as_mat, on_memory=False):
        
        self.fin = fin
        self.batch_size = batch_size
        self.sepline = sepline
        self.max_num_lines = max_num_lines
        self.max_line_length = max_line_length
        self.feed_callback = feed_callback
        self.on_memory = on_memory
        self.lines = [] # enabled if on_memory

        self.batch = []

        self.num_epochs = 0
        self.num_batches = 0
        self.num_lines = 0
        self.line_length_counts = collections.Counter()
        self.line_length_cumcounts = collections.Counter()
        
        self.__init_load()

    def __init_load(self):
        self.lines = []
        self.line_length_counts = collections.Counter()
        self.line_length_cumcounts = collections.Counter()

        unpacker = msgpack.Unpacker()
        self.fin.seek(0)

        BUFSIZE = 1024 * 1024
            
        while True:
            buf = self.fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for words in unpacker:
                if self.on_memory:
                    self.lines.append(words)
                self.line_length_counts[len(words)] += 1

        cum = 0
        for length, count in sorted(self.line_length_counts.items()):
            cum += count
            self.line_length_cumcounts[length] = cum

        
        self.fin.seek(0)

    def __length_lb(self, length):
        result = None
        for cur_length in sorted(self.line_length_counts):
            if cur_length <= length:
                result = cur_length
            else:
                break
        return result

    @property
    def num_epoch_lines(self):
        if self.max_num_lines is not None:
            return min(self.line_length_cumcounts[self.__length_lb(self.max_line_length)], self.max_num_lines)
        else:
            return self.line_length_cumcounts[self.__length_lb(self.max_line_length)]

    @property
    def num_epoch_batches(self):
        return self.num_epoch_lines // self.batch_size

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        if self.on_memory:
            return self.__next_on_memory()
        else:
            return self.__next()

    def __next_on_memory(self):
        self.batch = []
        num_lines = 0
        for words in self.lines:
            if len(words) > self.max_line_length:
                continue
            num_lines += 1
            if self.max_num_lines != None and num_lines == self.max_num_lines:
                break
            self.batch.append(words)
            self.num_lines += 1
            if len(self.batch) == self.batch_size:
                self.num_batches += 1
                yield self.feed_callback(self.batch)
                self.batch = []
        self.num_epochs += 1
        raise StopIteration


    def __next(self):
        self.batch = []
        unpacker = msgpack.Unpacker()
        BUFSIZE = 1024 * 1024
        num_lines = 0
        while True:
            buf = self.fin.read(BUFSIZE)
            if not buf:
                break
            unpacker.feed(buf)
            for words in unpacker:
                if len(words) > self.max_line_length:
                    continue
                num_lines += 1
                if self.max_num_lines != None and num_lines == self.max_num_lines:
                    break
                self.batch.append(words)
                self.num_lines += 1
                if len(self.batch) == self.batch_size:
                    self.num_batches += 1
                    yield self.feed_callback(self.batch)
                    self.batch = []
            if self.max_num_lines != None and num_lines == self.max_num_lines:
                break
        self.num_epochs += 1
        self.fin.seek(0)
        raise StopIteration
