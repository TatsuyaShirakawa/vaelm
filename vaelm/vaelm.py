# -*- coding:utf-8 -*-

import numpy as np

import chainer
from chainer import Variable, Chain, functions as F, links as L, cuda


class Encoder(Chain):

    def __init__(self, vocab_size, hidden_size, num_layers, ignore_label=-1):
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ignore_label = ignore_label

        args = {'embed': L.EmbedID(vocab_size, hidden_size, ignore_label=ignore_label)}
        for i in range(self.num_layers):
            args.update({'l{}'.format(i): L.StatelessLSTM(hidden_size, hidden_size)})
            setattr(self, 'h{}'.format(i), None)
            setattr(self, 'c{}'.format(i), None)

        super(Encoder, self).__init__(**args)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def get_l(self, i):
        return getattr(self, "l{}".format(i))

    def get_h(self, i):
        return getattr(self, "h{}".format(i))

    def get_c(self, i):
        return getattr(self, "c{}".format(i))

    def set_h(self, i, h):
        return setattr(self, "h{}".format(i), h)

    def set_c(self, i, c):
        return setattr(self, "c{}".format(i), c)

    def to_cpu(self):
        super(Encoder, self).to_cpu()
        for i in range(self.num_layers):
            h = self.get_h(i)
            c = self.get_c(i)
            if h is not None: h.to_cpu()
            if c is not None: c.to_cpu()

    def to_gpu(self):
        super(Encoder, self).to_gpu()
        for i in range(self.num_layers):
            h = self.get_h(i)
            c = self.get_c(i)
            if h is not None: h.to_gpu()
            if c is not None: c.to_gpu()

    def reset_state(self):
        for i in range(self.num_layers):
            self.set_h(i, None)
            self.set_c(i, None)

    def maybe_init_state(self, batch_size, dtype):
        
        for i in range(self.num_layers):
            if self.get_h(i) is None:
                xp = self.xp
                self.set_h(i, Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype)))
            if self.get_c(i) is None:
                xp = self.xp
                self.set_c(i, Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype)))

    def __call__(self, w, train=True, dpratio=0.5):

        x = self.embed(w)
        self.maybe_init_state(len(x.data), x.data.dtype)

        for i in range(self.num_layers):

            if self.ignore_label is not None:
                enable = (x.data != 0)

            c = F.dropout(self.get_c(i), train=train, ratio=dpratio)
            h = F.dropout(self.get_h(i), train=train, ratio=dpratio)
            x = F.dropout(x, train=train, ratio=dpratio)
            c, h = self.get_l(i)(c, h, x)

            if self.ignore_label != None:
                self.set_c(i, F.where(enable, c, self.get_c(i)))
                self.set_h(i, F.where(enable, h, self.get_h(i)))
            else:
                self.set_c(i, c)
                self.set_h(i, h)

            x = self.get_h(i)
        


class RNNLM(Chain):

    def __init__(self, vocab_size, hidden_size, num_layers, ignore_label=-1):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ignore_label = ignore_label

        args = {'embed': L.EmbedID(vocab_size, hidden_size, ignore_label=ignore_label),        
                'hy': L.Linear(hidden_size, vocab_size)}

        for i in range(self.num_layers):
            args.update({'l{}'.format(i): L.StatelessLSTM(hidden_size, hidden_size)})
            setattr(self, 'h{}'.format(i), None)
            setattr(self, 'c{}'.format(i), None)

        super(RNNLM, self).__init__(**args)
        
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.reset_state()

    def get_l(self, i):
        return getattr(self, "l{}".format(i))

    def get_h(self, i):
        return getattr(self, "h{}".format(i))

    def get_c(self, i):
        return getattr(self, "c{}".format(i))

    def set_h(self, i, h):
        return setattr(self, "h{}".format(i), h)

    def set_c(self, i, c):
        return setattr(self, "c{}".format(i), c)

    def to_cpu(self):
        super(RNNLM, self).to_cpu()
        for i in range(self.num_layers):
            h = self.get_h(i)
            c = self.get_c(i)
            if h is not None: h.to_cpu()
            if c is not None: c.to_cpu()

    def to_gpu(self):
        super(RNNLM, self).to_gpu()
        for i in range(self.num_layers):
            h = self.get_h(i)
            c = self.get_c(i)
            if h is not None: h.to_gpu()
            if c is not None: c.to_gpu()

    def reset_state(self):
        for i in range(self.num_layers):
            self.set_h(i, None)
            self.set_c(i, None)

    def maybe_init_state(self, batch_size, dtype):
        for i in range(self.num_layers):
            if self.get_h(i) is None:
                xp = self.xp
                self.set_h(i, Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype)))
            if self.get_c(i) is None:
                xp = self.xp
                self.set_c(i, Variable(xp.zeros((batch_size, self.hidden_size), dtype=dtype)))

    def __call__(self, w, train=True, dpratio=0.5):

        x = self.embed(w)
        self.maybe_init_state(len(x.data), x.data.dtype)

        for i in range(self.num_layers):

            if self.ignore_label is not None:
                enable = (x.data != 0)

            c = F.dropout(self.get_c(i), train=train, ratio=dpratio)
            h = F.dropout(self.get_h(i), train=train, ratio=dpratio)
            x = F.dropout(x, train=train, ratio=dpratio)
            c, h = self.get_l(i)(c, h, x)

            if self.ignore_label != None:
                self.set_c(i, F.where(enable, c, self.get_c(i)))
                self.set_h(i, F.where(enable, h, self.get_h(i)))
            else:
                self.set_c(i, c)
                self.set_h(i, h)

            x = self.get_h(i)
            
        x = F.dropout(x, train=train, ratio=dpratio)
        return self.hy(x)

class Transformer(Chain):

    def __init__(self, hidden_size, z_size, num_layers):

        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_layers = num_layers

        args = {'lmu': L.Linear(hidden_size, z_size),
                'lsigma': L.Linear(hidden_size, z_size)}

        for i in range(self.num_layers):
            args.update({'l{}'.format(i): L.Linear(hidden_size, hidden_size)})

        super(Transformer, self).__init__(**args)

    def get_l(self, i):
        return getattr(self, "l{}".format(i))

    def __call__(self, h, train=True, dpratio=0.5):
        h = F.dropout(h, train=train, ratio=dpratio)
        for i in range(self.num_layers):
            h = F.tanh(self.get_l(i)(h))
        return (self.lmu(h), F.exp(self.lsigma(h)))
            

class VAELM(Chain):
    
    def __init__(self, vocab_size, hidden_size, z_size, num_layers, num_infer_layers=0, ignore_label=-1):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_layers = num_layers
        self.num_infer_layers = num_infer_layers
        self.ignore_label = ignore_label

        args = {'encoder': Encoder(vocab_size, hidden_size, num_layers=num_layers, ignore_label=ignore_label),
                'decoder': RNNLM(vocab_size, hidden_size, num_layers=num_layers, ignore_label=ignore_label),
            }

        for i in range(num_layers):
            args.update({'htrans{}'.format(i) : Transformer(hidden_size, z_size, num_infer_layers)})
            args.update({'ctrans{}'.format(i) : Transformer(hidden_size, z_size, num_infer_layers)})
            args.update({'zh{}'.format(i) : L.Linear(z_size, hidden_size)})
            args.update({'zc{}'.format(i) : L.Linear(z_size, hidden_size)})

        self.hmus = [None for i in range(num_layers)]
        self.hsigmas = [None for i in range(num_layers)]
        self.cmus = [None for i in range(num_layers)]
        self.csigmas = [None for i in range(num_layers)]
    
        super(VAELM, self).__init__(**args)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def get_htrans(self, i):
        return getattr(self, "htrans{}".format(i))

    def get_ctrans(self, i):
        return getattr(self, "ctrans{}".format(i))

    def get_zh(self, i):
        return getattr(self, "zh{}".format(i))

    def get_zc(self, i):
        return getattr(self, "zc{}".format(i))

    def to_cpu(self):
        super(VAELM, self).to_cpu()
        self.encoder.to_cpu()
        self.decoder.to_cpu()
        for i in range(self.num_layers):
            self.get_htrans(i).to_cpu()
            self.get_ctrans(i).to_cpu()

    def to_gpu(self):
        super(VAELM, self).to_gpu()
        self.encoder.to_gpu()
        self.decoder.to_gpu()
        for i in range(self.num_layers):
            self.get_htrans(i).to_gpu()
            self.get_ctrans(i).to_gpu()
        
    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        for i in range(self.num_layers):
            self.hmus[i] = None
            self.hsigmas[i] = None
            self.cmus[i] = None
            self.csigmas[i] = None

    def infer(self, train=True):

        for i in range(self.num_layers):

            h = self.encoder.get_h(i)
            hmu, hsigma = self.get_htrans(i)(h)
            self.hmus[i] = hmu
            self.hsigmas[i] = hsigma

            c = self.encoder.get_c(i)
            cmu, csigma = self.get_ctrans(i)(c, train=train)
            self.cmus[i] = cmu
            self.csigmas[i] = csigma

    def set_by_MLE(self, train=True):
        for i in range(self.num_layers):
            self.decoder.set_h(i, self.get_zh(i)(self.hmus[i]))
            self.decoder.set_c(i, self.get_zc(i)(self.cmus[i]))

    def set_by_sample(self, train=True):
        xp = self.xp
        use_gpu = (xp == cuda.cupy)
        for i in range(self.num_layers):
            # h
            mu, sigma = self.hmus[i], self.hsigmas[i]
            e = np.random.normal(0., 1., self.z_size).astype(np.float32)
            if use_gpu:
                e = cuda.to_gpu(e)
            self.decoder.set_h(i, self.get_zh(i)(mu + e * sigma))

            # c
            mu, sigma = self.cmus[i], self.csigmas[i]
            e = np.random.normal(0., 1., self.z_size).astype(np.float32)
            if use_gpu:
                e = cuda.to_gpu(e)
            self.decoder.set_c(i, self.get_zc(i)(mu + e * sigma))

    def encode(self, w, train=True):
        return self.encoder(w, train=train)

    def decode(self, w, train=True):
        return self.decoder(w, train=train)
