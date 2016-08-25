#!/usr/bin/env bash
# download Pen Tree Bank Dataset

if [ ! -e ./ptb ]; then
    mkdir ptb
fi

wget 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt' -O ./ptb/train.txt
wget 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt' -O ./ptb/test.txt
wget 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt' -O ./ptb/valid.txt

echo download completed!



