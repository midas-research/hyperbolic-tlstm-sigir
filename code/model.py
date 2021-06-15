import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import geoopt.manifolds.poincare.math as pmath_geo
import math, itertools


def one_rnn_transform(W, h, U, x, c):
    W_otimes_h = pmath_geo.mobius_matvec(W, h, c=c)
    U_otimes_x = pmath_geo.mobius_matvec(U, x, c=c)
    Wh_plus_Ux = pmath_geo.mobius_add(W_otimes_h, U_otimes_x, c=c)
    return Wh_plus_Ux

class TimeLSTMHyp(nn.Module):
    def __init__(self, input_size, hidden_size, device, cuda_flag=False, bidirectional=False):
        super(TimeLSTMHyp, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.U_all = torch.nn.Parameter(torch.Tensor(hidden_size * 4, input_size))
        self.W_d = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bidirectional = bidirectional
        self.c = torch.tensor([1.0]).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.W_all, self.U_all, self.W_d]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, timestamps, hidden_states, reverse=False):
        b, seq, embed = inputs.size()
        h = hidden_states[0]
        _c = hidden_states[1]
        if self.cuda_flag:
            h = h.cuda()
            _c = _c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = pmath_geo.expmap0(torch.tanh(pmath_geo.logmap0(pmath_geo.mobius_matvec(self.W_d, _c, c=self.c), c=self.c))) # short term mem
            c_s2 = pmath_geo.mobius_pointwise_mul(c_s1, timestamps[:, s: s + 1].expand_as(c_s1), c=self.c) # discounted short term mem
            c_l = pmath_geo.mobius_add(-c_s1, _c, c=self.c) # long term mem
            c_adj = pmath_geo.mobius_add(c_l, c_s2, c=self.c)
            
            W_f, W_i, W_o, W_c_tmp = self.W_all.chunk(4, dim=1)
            U_f, U_i, U_o, U_c_tmp = self.U_all.chunk(4, dim=0)
            # print ('WF: ', W_f.shape)
            # print ('H: ', h.shape)
            # print ('UF: ', U_f.shape)
            # print ('X: ', inputs[:, s].shape)
            f = pmath_geo.logmap0(one_rnn_transform(W_f, h, U_f, inputs[:, s], self.c), c=self.c).sigmoid()
            i = pmath_geo.logmap0(one_rnn_transform(W_i, h, U_i, inputs[:, s], self.c), c=self.c).sigmoid()
            o = pmath_geo.logmap0(one_rnn_transform(W_o, h, U_o, inputs[:, s],self.c), c=self.c).sigmoid()
            c_tmp = pmath_geo.logmap0(one_rnn_transform(W_c_tmp, h, U_c_tmp, inputs[:, s], self.c), c=self.c).sigmoid()
            
            f_dot_c_adj = pmath_geo.mobius_pointwise_mul(f, c_adj, c=self.c)
            i_dot_c_tmp = pmath_geo.mobius_pointwise_mul(i, c_tmp, c=self.c)
            _c = pmath_geo.mobius_add(i_dot_c_tmp, f_dot_c_adj, c=self.c)

            h = pmath_geo.mobius_pointwise_mul(o, pmath_geo.expmap0(torch.tanh(_c), c=self.c), c=self.c)
            outputs.append(o)
            hidden_state_c.append(_c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()
        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, _c)

class HTLSTM(nn.Module):
    """
    Forecasting model using LSTM

    B*5*30*N
    """

    def __init__(
        self,
        text_embed_dim,
        intraday_hiddenDim,
        interday_hiddenDim,
        intraday_numLayers,
        interday_numLayers,
        maxlen=30,
        outdim=2,
        device=torch.device("cpu"),
    ):
        """"""
        super(HTLSTM, self).__init__()
        self.lstm1 = TimeLSTMHyp(
            input_size=text_embed_dim,
            hidden_size=intraday_hiddenDim,
            device=device,
        )
        self.intraday_hiddenDim = intraday_hiddenDim
        self.lstm1_outshape = intraday_hiddenDim
        self.maxlen = maxlen
        self.lstm2_outshape = interday_hiddenDim
        self.cell = functools.partial(hyrnn.MobiusGRU, c=1.0)
        self.cell_source = self.cell(self.lstm1_outshape, interday_hiddenDim, 1)
        self.linear3 = nn.Linear(self.lstm2_outshape, 256)
        self.linear4 = nn.Linear(256, outdim)
        self.linear5 = nn.Linear(256,1)

        self.drop = nn.Dropout(p=0.3)
        self.batchnorm = nn.BatchNorm1d(128)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device = device
        self.c = torch.tensor([1.0]).to(self.device)

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.intraday_hiddenDim)).to(
            self.device)
        c = Variable(torch.zeros(self.bs, self.intraday_hiddenDim)).to(
            self.device)

        return (h, c)

    def forward(self, sentence_feats, len_tweets, time_feats):
        """
        sentence_feat: sentence features (B*5*30*N),
        len_tweets: (B*5)
        time_feats: (B*5*30)
        """
        sentence_feats = pmath_geo.expmap0(sentence_feats, c=self.c)
        # time_feats = pmath_geo.expmap0(time_feats, c=self.c)
        sentence_feats = sentence_feats.permute(1, 0, 2, 3)
        len_days, self.bs, _, _ = sentence_feats.size()
        h_init, c_init = self.init_hidden()

        len_tweets = len_tweets.permute(1, 0)
        time_feats = time_feats.permute(1, 0, 2)

        lstm1_out = torch.zeros(
            len_days, self.bs, self.lstm1_outshape).to(self.device)

        for i in range(len_days):
            temp_lstmout, (_, _) = self.lstm1(
                sentence_feats[i], time_feats[i], (h_init, c_init)
            )
            last_idx = len_tweets[i]
            last_idx = last_idx.type(torch.int).tolist()
            temp_hn = torch.zeros(self.bs, 1, self.lstm1_outshape)
            for j in range(self.bs):
                temp_hn[j] = temp_lstmout[j, last_idx[j]-1, :]
            lstm1_out[i] = temp_hn.squeeze(1)
        lstm1_out = lstm1_out.permute(1, 0, 2)
        batch_size = lstm1_out.shape[0]
        num_of_timesteps = lstm1_out.shape[1]
        '''
        Hyberpolic exp
        '''
        all_outputs, cell_output = self.cell_source(lstm1_out.permute(1,0,2))

        cell_output = cell_output[-1]
        x = pmath_geo.logmap0(cell_output, c=self.c)
        x = self.drop(self.relu(self.linear3(x)))
        cse_output = self.linear4(x)
        margin_output = self.linear5(x)
        return cse_output, margin_output