## vaelm

Chainer-implementation of the language model introduced in the following paper:

    [Samuel R. Bowman+, "Generating Sentences from a Continuous Space", ICLR2016](https://arxiv.org/pdf/1511.06349.pdf)

## Requirements (for Python)

- Python 2.7.5+, 3.5+, 
- chainer v1.11.0+ (highly recommended to enable GPU + CUDNN environment)
- six
- numpy
- msgpack-python
- nltk.punkt (do nltk.download() and select d)ownload and punkt)

## Example (Training)

```
cd /path/to/this_directory/
mkdir work && cd work
bash ../sh/download_ptb.sh
bash ../sh/prepare.sh ./ptb
bash ../sh/run.sh
```

