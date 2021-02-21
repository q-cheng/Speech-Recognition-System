import numpy as np
import math
import torch 
import random
import torch.nn.functional as F 
from torch.nn import Embedding, LSTMCell, GRUCell, Linear
from torch.nn import Dropout,LogSoftmax
from torch.nn import ModuleList
from utils import pad_list,to_device
import configargparse

class LockedDropout(torch.nn.Module): 
    def __init__(self, p=0.5):
        self.p = p
        super().__init__() 

    def forward(self, x):
        if not self.training or not self.p: 
            return x
        x = x.clone()
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p) 
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


class Speller(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace,att: torch.nn.Module=None):
        """
        Neural Network Module for the Sequence to Sequence LAS Model
        :params configargparse.Namespace params: The training options
        :params torch.nn.Module att: The attention module
        """        
        super(Speller,self).__init__()
        ## Embedding Layer
        self.embed = Embedding(params.odim,params.demb_dim)
        ## Decoder with LSTM Cells
        self.decoder = ModuleList()
        self.dropout_dec = ModuleList()
        self.dtype = params.dtype
        self.dunits = params.dhiddens
        self.dlayers = params.dlayers
        self.decoder += [
                LSTMCell(params.eprojs + params.demb_dim, params.dhiddens)
                if self.dtype == "lstm"
                else GRUCell(params.eprojs + params.demb_dim, params.dhiddens)
            ]
        self.dropout_dec += [Dropout(p=params.ddropout)]
        self.dropout_emb = Dropout(p=params.ddropout)
        ## Other decoder layers if > 1 decoder layer
        for i in range(1,params.dlayers): 
            self.decoder += [
                LSTMCell(params.dhiddens, params.dhiddens)
                if self.dtype == "lstm"
                else GRUCell(params.dhiddens, params.dhiddens)
            ]
            self.dropout_dec += [LockedDropout(p=params.ddropout)] # Dropout
        
        ## Project to Softmax Space- Output
        self.projections = Linear(params.dhiddens, params.odim)
        ## Attention Module
        self.att = att
        ## Scheduled Sampling
        self.sampling_probability = params.ssprob
        ## Initialize EOS, SOS
        self.eos = len(params.char_list) -1
        self.sos = self.eos
        self.ignore_id = params.text_pad

    def rnn_forward(self, lstm_input: torch.Tensor, dec_hidden_states: list, dec_hidden_cells: list, dec_hidden_states_prev: list, dec_hidden_cells_prev: list):
        """
        Performs forward pass through LSTMCells in the decoder
        :param torch.Tensor lstm_input- concatenated embedding vector and attention context that is input to first LSTMCell layer
        :param list(torch.Tensor) dec_hidden_states- Hidden states of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_cells- Hidden cells of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_states_prev- Hidden states of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_states_cells- Hidden cells of the decoder for all layers in a list.
        :returns list(torch.Tensor) dec_hidden_states- Hidden states of the decoder for all layers in a list.
        :returns list(torch.Tensor) dec_hidden_cells- Hidden cells of the decoder for all layers in a list.
        """
        dec_hidden_states[0], dec_hidden_cells[0] = self.decoder[0](lstm_input, (dec_hidden_states_prev[0], dec_hidden_cells_prev[0]))
        for i in range(1, self.dlayers):
            dec_hidden_states[i], dec_hidden_cells[i] = self.decoder[i](
                self.dropout_dec[i - 1](dec_hidden_states[i - 1]), (dec_hidden_states_prev[i], dec_hidden_cells_prev[i])
            )
        return dec_hidden_states, dec_hidden_cells
    
    def forward(self,hs:torch.Tensor,hlens:list,ys_out:torch.LongTensor,ylen:list):
        """
        Performs the forward pass over the decoder 
        :param torch.Tensor hs- Encoded output sequence
        :param list hlens- Lengths of the encoded output sequence without padding
        :param torch.LongTensor ys_out- Target output sequence with padding 
        :param list ylen- Target sequence lengths without padding
        :returns torch.Tensor logits: Output projection to vocabulary space [Batch,Max_token_length,Vocabulary_size] 
        :returns torch.LongTensor- Target output sequence with <eos> added to the end for loss computation
        """
        max_length = max(ylen)
        # print(max_length, hs.shape, ctc_out.shape)
        batch_size, _, D_enc = hs.shape
        # print('Start decoding!')
        dec_hidden_states = []  # n * 23 * 512
        dec_hidden_cells = []   # n * 23 * 512
        logits_list = []

        # Init first hidden state and cells.
        for _ in range(self.dlayers):
            dec_hidden_states.append(to_device(self, torch.zeros(batch_size, D_enc*2)))
            dec_hidden_cells.append(to_device(self, torch.zeros(batch_size, D_enc*2)))
        dec_hidden_states_prev = dec_hidden_states
        dec_hidden_cells_prev = dec_hidden_cells

        ## Pre-compute embedding.
        embed_list = []
        # Get unpad sequences. 
        ys_unpadded = [(y[y!=-1]).tolist() for y in ys_out]
        
        ys_charemb = [to_device(self, torch.LongTensor([self.sos] + y)) for y in ys_unpadded]
        ys_charemb_padding = pad_list(ys_charemb, self.eos).T

        for i in range(max_length+1):
            index_long_tesnor = ys_charemb_padding[i]
            new_embed = self.embed(index_long_tesnor)
            embed_list.append(new_embed)

        self.att.reset()
        attention_w = None
        # Go through all timestamp.
        for i in range(max_length+1):
            embeding = embed_list[i]
            attention_c, attention_w = self.att(hs=hs, hlens=hlens, dec_state=dec_hidden_states[0], att_prev=attention_w)
            l_inp = torch.cat([attention_c, embeding], dim=-1)   # l_inp -> batch * (old + embed)
            new_dec_hidden_states, new_dec_hidden_cells = self.rnn_forward(l_inp, dec_hidden_states, dec_hidden_cells,
                                                                           dec_hidden_states_prev, dec_hidden_cells_prev)
            dec_hidden_states_prev, dec_hidden_cells_prev = dec_hidden_states, dec_hidden_cells
            dec_hidden_states, dec_hidden_cells = new_dec_hidden_states, new_dec_hidden_cells
            l_class = self.projections(dec_hidden_states[-1]) # batch * hidden_dim
            logits_list.append(l_class)

        ## Get logits, [Batch,Max_token_length,Vocabulary_size].
        logits = torch.stack(logits_list, dim=1)
    
        ## Get Target output sequence with <eos> added to the end. Need to change this.
        ys_ref = [torch.LongTensor(y + [self.eos]) for y in ys_unpadded]
        ys_out_with_eos = to_device(self, pad_list(ys_ref, -1))

        # print('Finish one batch decoding!')
        # print('=' * 50)

        ## Return data_length, label_length ?
        return logits, ys_out_with_eos
             

    def greedy_decode(self,hs:torch.Tensor,hlens:torch.LongTensor,ctc_out,params:configargparse.Namespace):
        """
        Performs greedy decoding  
        :param torch.Tensor hs- Encoded output sequence
        :param list hlens- Lengths of the encoded output sequence without padding
        :param Namespace params- Decoding options
        """
        print('Start greedy decode')
        # Need to rewrite this.
        maxlenratio = params.maxlenratio
        max_length = 200
        batch_size, _, D_enc = hs.shape
        dec_hidden_states = []
        dec_hidden_cells = []
        tokens_list = []
        final_tokens_list = []

        # Put into GPU.
        hs = to_device(self, hs)

        # Init first hidden state and cells.
        for _ in range(self.dlayers):
            dec_hidden_states.append(to_device(self,torch.zeros(batch_size, D_enc * 2)))
            dec_hidden_cells.append(to_device(self,torch.zeros(batch_size, D_enc * 2)))
        dec_hidden_states_prev = dec_hidden_states
        dec_hidden_cells_prev = dec_hidden_cells

        # Init embedding as <sos>.
        init_embed = self.embed(to_device(self, torch.LongTensor(torch.ones(batch_size).long()) * self.sos))

        last_tokens = None
        # How do we stop when some sentence reach eos, do not need to return eos.
        # Try to compute with whole batch.
        log_softmax_layer = LogSoftmax(1)

        print('Start greedy decoding!')
        attention_w = None
        self.att.reset()
        # print(ctc_out.shape, max_length)
        for i in range(max_length):
            if i == 0:
                # First step, use <sos> as embedding.
                embeding = init_embed
            else:
                # Use previous token as embedding.
                embeding = self.embed(last_tokens)

            attention_c, attention_w = self.att.forward(hs=hs, hlens=hlens, dec_state=dec_hidden_states[0], att_prev=attention_w)
            l_inp = torch.cat([attention_c, embeding], dim=-1)  
            new_dec_hidden_states, new_dec_hidden_cells = self.rnn_forward(l_inp, dec_hidden_states, dec_hidden_cells,
                                                                           dec_hidden_states_prev,
                                                                           dec_hidden_cells_prev)
            dec_hidden_states_prev, dec_hidden_cells_prev = dec_hidden_states, dec_hidden_cells
            dec_hidden_states, dec_hidden_cells = new_dec_hidden_states, new_dec_hidden_cells
            l_class = self.projections(dec_hidden_states[-1])

            ## Add ctc.
            if i <= ctc_out.shape[1] - 1:
                l_class = 0.9 * l_class + 0.1 * ctc_out[:, i, :]

            last_tokens = torch.argmax(log_softmax_layer(l_class), dim=1)
        
            ## Add encoder output.
            tokens_list.append(last_tokens)
            
        # Get tokens, [Batch,Max_token_length]. From [batch, 1] -> [Batch,Max_token_length].
        tokens = torch.stack(tokens_list, dim=1)
        print('Start to find EOS of each prediction!')
        # Find eos and get the valid tokens. Will the return sequence contains EOS ?
        for i in range(tokens.shape[0]):
            # Numpy find the first index. Need to change this later.
            for j in range(tokens.shape[1]):
                if tokens[i][j] == self.eos:
                    final_tokens_list.append(tokens[i, :j])
                    break
                if j == tokens.shape[1] - 1:
                    final_tokens_list.append(tokens[i, :])

        print('Finish greedy decoding of one batch!')
        print('=' * 50)
        return tokens




    def beam_decode(self,hs:torch.Tensor,hlens:torch.LongTensor,params:configargparse.Namespace):
        return None