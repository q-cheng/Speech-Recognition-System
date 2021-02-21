import torch
from utils import make_pad_mask,to_device
import torch.nn.functional as F 
import configargparse

class Attention(torch.nn.Module):
    def __init__(self, params: configargparse.Namespace):
        """
        Location-aware attention 
        Reference: Attention-Based Models for Speech Recognition
            (https://arxiv.org/pdf/1506.07503.pdf)
        :param params: Training and Model Parameters used in train.py 
        """
        super(Attention, self).__init__()
        self.keys = torch.nn.Linear(params.eprojs, params.att_dim)
        self.query = torch.nn.Linear(params.dhiddens, params.att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(params.aconv_chans, params.att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1,
            params.aconv_chans,
            (1, 2 * params.aconv_filts + 1),
            padding=(0, params.aconv_filts),
            bias=False,
        )
        self.gvec = torch.nn.Linear(params.att_dim, 1)
        self.dunits = params.dhiddens
        self.eprojs = params.eprojs
        self.att_dim = params.att_dim

    def reset(self):
        """
        Resets hs, hlens, key,mask,att_wt for a new batch of examples in the decoder
        """
        self.hlens = None
        self.enc_h = None
        self.key = None
        self.mask = None
        self.att_wt = None

    def forward(self,hs: torch.Tensor ,hlens: list,dec_state: torch.Tensor,att_prev:torch.Tensor):
        """
        Performs forward propagation.
        :param torch.Tensor hs: padded encoder hidden state (B x T_max x D_enc)
        :param list hlens: padded encoder hidden state length (B)
        :param torch.Tensor dec_state: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attention weight (B x T_max)
        :return: torch.Tensor c: attention weighted encoder state (B, D_enc)
        :return: torch.Tensor w: attention weights (B x T_max)
        """
        batch = len(hs)
        ## The key, encoder state are the same across decoder time-steps for a batch
        ## Here we pre-compute those at the first decoder time-step
        if self.key is None:
            self.hs = hs  
            self.hlens = self.hs.size(1)
            ## Project the encoder hidden state to obtain the key vector
            self.key = self.keys(self.hs)

        ## Initialized the decoder state with zeros at the first decoder time or reshape to (batch_size,dhiddens)
        if dec_state is None:
            dec_state = hs.new_zeros(batch, self.dunits)
        else:
            dec_state = dec_state.view(batch, self.dunits)
        ##  Project the decoder hidden state to obtain the query
        query = self.query(dec_state).view(batch, 1, self.att_dim)


        ## If previous attention weight does not exist, then initialize uniform attention weights 
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(hlens).to(
                device=dec_state.device, dtype=dec_state.dtype
            )
            att_prev = att_prev / att_prev.new(hlens,device=dec_state.device).unsqueeze(-1)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.hlens))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)
        # utt x frame x att_dim -> utt x frame
        ## Compute the energy using the query,key, and convolved prev attention weight
        e = self.gvec(
            torch.tanh(att_conv + self.key + query)
        ).squeeze(2)

        ## Create the padding mask on the encoder output hs 
        if self.mask is None:
            self.mask = to_device(hs, make_pad_mask(hlens))
        e.masked_fill_(self.mask, -float("inf"))
        ## Compute the attention weight
        w = F.softmax(e, dim=1)
        ## Update self.att_wt to store attention weights for all timesteps until now
        if self.att_wt is None:
            self.att_wt = w.unsqueeze(1)
        else:
            self.att_wt = torch.cat((self.att_wt,w.unsqueeze(1)),dim=1)
        ## Compute the encoder context using the attention weight and encoder hidden states 
        c = torch.sum(self.hs * w.view(batch, self.hlens, 1), dim=1)

        return c, w