import numpy as np
import sys
import torch
from torch.nn import LSTM,RNN,GRU,Linear,Dropout
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pack_sequence
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

class RNNLayer(torch.nn.Module):
    
    def __init__(self, idim: int ,hdim: int ,nlayers: int=1, enc_type: str ="blstm"):
        """
        This represents the computation that happens for 1 RNN Layer
        Uses packing,padding utils from Pytorch
        :param int input_dim- The input size of the RNN
        :param int hidden_dim- The hidden size of the RNN
        :param int nlayers- Number of RNN Layers
        :param str enc_type : Type of encoder- RNN/GRU/LSTM
        """
        super(RNNLayer,self).__init__()
        bidir = True if enc_type[0] == 'b' else False
        enc_type = enc_type[1:] if enc_type[0] == 'b' else enc_type
        if enc_type == "rnn":
            self.elayer = RNN(idim, hdim, nlayers, batch_first=True, bidirectional=bidir)
        elif enc_type == "lstm":
            self.elayer = LSTM(idim, hdim, nlayers, batch_first=True, bidirectional=bidir)
        else:
            self.elayer = GRU(idim, hdim, nlayers, batch_first=True, bidirectional=bidir)

    def forward(self, x: torch.Tensor, inp_lens: torch.LongTensor):
        """
        Foward propogation for the RNNLayer
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        :returns Encoder hidden state
        """
        total_length = x.size(1)
        packed_x = pack_padded_sequence(x, inp_lens,batch_first=True)
        self.elayer.flatten_parameters()
        output, (hidden,_) = self.elayer(packed_x)
        unpacked_out,inp_lens = pad_packed_sequence(output,batch_first=True,total_length=total_length)
        return unpacked_out,inp_lens,hidden
        

class pBLSTM(torch.nn.Module):
    
    def __init__(self, input_dim: int , hidden_dim: int,subsample_factor:int = 2,enc_type: str ="blstm"):
        """
        Pyramidal BLSTM Layer: 
        :param int input_dim- The input size of the RNN
        :param int hidden_dim- The hidden size of the RNN
        :param int subsample_factor- Determines the factor by which the time dimension is downsampled and 
                                the hidden state is upsampled. Value 2 was used in the LAS paper
        :param str enc_type : Type of encoder- RNN/GRU/LSTM
        It takes in a sequence of shape [bs,enc_length,hiddens], 
        and converts it to a sequence of shape [bs,enc_length//subsample_factor, hidden*subsample_factor]
        """
        super(pBLSTM, self).__init__()
        self.factor = subsample_factor
        self.pblstm = RNNLayer(self.factor*input_dim,hidden_dim,1,enc_type)
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor , inp_lens: torch.LongTensor): 
        """
        Foward propogation for the pBLSTM
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        :returns Encoder hidden state
        """
        batch_size,seq_len,_ = x.size()
        factor = 2
        # print(x.shape, inp_lens)
        ## Handle seq_len % factor !=0 by dropping the last few frames
        if seq_len % factor != 0:
            new_length_drop = seq_len % factor
            x = x[:, :(-new_length_drop), :]  # Drop last several elements to make sure seq_len % factor == 0.
            # mask = new_length * torch.ones(inp_lens.size()).long()
            # inp_lens = torch.where(inp_lens==seq_len, mask, inp_lens)
        # Perform the Pyramidal Subsampling and pass through the LSTM
        inp_lens = inp_lens // 2
        new_x = x.reshape(batch_size, seq_len // factor, -1)
        unpacked_out, inp_lens, hidden = self.pblstm(new_x, inp_lens)
        return unpacked_out, inp_lens, hidden
        # raise NotImplementedError
    


class Listener(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace):
        """
        Neural Network Module for the Listener
        :params configargparse.Namespace params: The training options
        """      
        super(Listener, self).__init__()
        self.elayers = params.elayers
        self.etype = params.etype
        self.edropout = params.edropout
        rnn0 = RNNLayer(params.idim,params.ehiddens,1,params.etype)
        setattr(self, "%s0" % (params.etype),rnn0)
        dropout0 = Dropout(self.edropout)
        setattr(self, "dropout0", dropout0)
        for i in range(params.elayers):
            rnn = pBLSTM(2*params.ehiddens,params.ehiddens,params.pfactor,params.etype)
            setattr(self, "%s%d" % (params.etype, i+1), rnn)
            if i == params.elayers-1:
                projection = Linear(2*params.ehiddens,params.eprojs)
                setattr(self,"proj%d" % (i+1),projection)
                ctc_proj = Linear(params.eprojs,len(params.char_list))
                setattr(self,"ctc_proj%d" % (i+1),ctc_proj)
            else:
                # Add dropout layer.
                dropout = LockedDropout(self.edropout)
                setattr(self, "dropout%d" % (i+1), dropout)
    

    def forward(self,x: torch.Tensor,inp_lens: torch.LongTensor):
        """
        Foward propogation for the encoder
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        """
        rnn0 = getattr(self, '{}0'.format(self.etype))
        x, inp_lens, hidden = rnn0.forward(x, inp_lens)
        dropout0 = getattr(self, 'dropout0')
        x = dropout0(x)
        for i in range(self.elayers):
            cur_layer = getattr(self, '{}{}'.format(self.etype, i+1))
            # Add drop out.
            x, inp_lens, _ = cur_layer(x, inp_lens)
            if i == self.elayers - 1:
                project = getattr(self, 'proj{}'.format(i+1))
                ctc_proj = getattr(self, 'ctc_proj{}'.format(i+1))
                # Get ctc output.
                
                # Go through the last projection layer.
                x = project(x)
                ctc_out = ctc_proj(x)
                
            else:
                dropout_layer = getattr(self, 'dropout{}'.format(i+1))
                x = dropout_layer(x)

        return x, inp_lens.squeeze().tolist(), ctc_out
        # raise NotImplementedError
        

