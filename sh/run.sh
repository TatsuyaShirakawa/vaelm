#!/usr/bin/env python

# if you would like to load entire data on memory, add `--on_memory true`
python -u ../vaelm/train.py --gpu 0 --train_file $1/train.pack --test_file $1/test.pack --valid_file $1/valid.pack --vocab_file $1/vocab.pack --save_dir $1/save --encoding utf-8 --num_samples 1 $2 $3 $4 $5 $6 $7 $8 $9
