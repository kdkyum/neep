import torch
import torch.nn as nn
import torch.nn.functional as F


class NEEP(nn.Module):
    def __init__(self, opt):
        super(NEEP, self).__init__()
        self.n_layer = opt.n_layer
        tmp = nn.Sequential()
        tmp.add_module("fc", nn.Linear(2 * opt.n_input, opt.n_hidden))
        tmp.add_module("relu", nn.ReLU(inplace=True))
        setattr(self, "layer1", tmp)
        for i in range(opt.n_layer - 1):
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))
            tmp.add_module("relu", nn.ReLU(inplace=True))
            setattr(self, "layer%d" % (i + 2), tmp)

        self.out = nn.Linear(opt.n_hidden, 1)

    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=-1)
        x_r = torch.cat([s2, s1], dim=-1)

        for i in range(self.n_layer):
            f = getattr(self, "layer%d" % (i + 1))
            x = f(x)
        out_f = self.out(x)

        for i in range(self.n_layer):
            f = getattr(self, "layer%d" % (i + 1))
            x_r = f(x_r)
        out_r = self.out(x_r)

        return out_f - out_r


class EmbeddingNEEP(nn.Module):
    def __init__(self, opt):
        super(EmbeddingNEEP, self).__init__()
        self.encoder = nn.Embedding(opt.n_token, opt.n_hidden)
        self.h = nn.Sequential()
        for i in range(opt.n_layer):
            self.h.add_module(
                "fc%d" % (i + 1), nn.Linear(2 * opt.n_hidden, 2 * opt.n_hidden)
            )
            self.h.add_module("relu%d" % (i + 1), nn.ReLU())
        self.h.add_module("out", nn.Linear(2 * opt.n_hidden, 1))

    def forward(self, s1, s2):
        s1 = self.encoder(s1)
        s2 = self.encoder(s2)
        x = torch.cat([s1, s2], dim=-1)
        _x = torch.cat([s2, s1], dim=-1)
        return self.h(x) - self.h(_x)


class RNEEP(nn.Module):
    def __init__(self, opt):
        super(RNEEP, self).__init__()
        self.encoder = nn.Embedding(opt.n_token, opt.n_hidden)
        self.rnn = nn.GRU(opt.n_hidden, opt.n_hidden, opt.n_layer)
        self.fc = nn.Linear(opt.n_hidden, 1)

        self.nhid = opt.n_hidden
        self.nlayers = opt.n_layer

    def forward(self, x):
        bsz = x.size(1)
        h_f = self.init_hidden(bsz)
        emb_forward = self.encoder(x)
        output_f, _ = self.rnn(emb_forward, h_f)

        h_r = self.init_hidden(bsz)
        x_r = torch.flip(x, [0])
        emb_reverse = self.encoder(x_r)
        output_r, _ = self.rnn(emb_reverse, h_r)

        S = self.fc(output_f.mean(dim=0)) - self.fc(output_r.mean(dim=0))
        return S

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid).detach()
