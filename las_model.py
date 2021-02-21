import torch 
import numpy as np
import math
from models.attention import Attention
from models.encoder import Listener
from models.decoder import Speller
from utils import StatsCalculator
import configargparse


class SpeechLAS(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace):
        """
        Neural Network Module for the Sequence to Sequence LAS Model
        :params configargparse.Namespace params: The training options
        """
        super(SpeechLAS,self).__init__()

        self.encoder = Listener(params)
        self.att = Attention(params)
        self.decoder = Speller(params,self.att)
        self.stat_calculator = StatsCalculator(params) 
        ## Initialize Weights and Biases
        self.initialize_weights()
    

    def initialize_weights(self):
        """
        Weight Initialization for SpeechLAS Model
        W ~ Normal(0, fan_in ** -0.5), b = 0
        Pytorch uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5); Kyoto University (Hirofumi Inaguma)
        """
        for p in self.parameters():
            data = p.data
            if data.dim() == 1:
                # bias
                data.zero_()
            elif data.dim() == 2:
                # linear weight
                n = data.size(1)
                stdv = 1.0 / math.sqrt(n)
                data.normal_(0, stdv)
            elif data.dim() in (3, 4):
                # conv weight
                n = data.size(1)
                for k in data.size()[2:]:
                    n *= k
                stdv = 1.0 / math.sqrt(n)
                data.normal_(0, stdv)
            else:
                raise NotImplementedError        
        self.decoder.embed.weight.data.normal_(0, 1)
        for i in range(len(self.decoder.decoder)):
            bias = self.decoder.decoder[i].bias_ih
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.0)


        
    def forward(self,xs: torch.Tensor,xlens: list,ys_ref: torch.LongTensor,ylen: list):
        """
        Forward propogation for SpeechLAS Model
        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences 
        :params torch.LongTensor ys_ref- Padded Text Tokens 
        :params list ylen- Lengths of unpadded text sequences 

        """
        ## 1. Encoder
        encoder_output,hlens,ctc_proj = self.encoder(xs,xlens)
        ## 2. Decode Outputs
        ys_pred,ys_ref = self.decoder(encoder_output,hlens,ys_ref,ylen)        
        
        ## 3. Compute CE Loss, Accuracy/CER, Perplexity and WER
        ## Pass ylen, hlens into stat_calculator
        self.loss,acc,wer = self.stat_calculator(ys_pred,ys_ref,ylen,ctc_proj,hlens) # ,ylen,encoder_output,hlens
        
        return self.loss,acc,wer
    
    
    def decode_greedy(self,xs: torch.Tensor,xlens: torch.LongTensor,decode_params: configargparse.Namespace):
        """
        Performs Greedy Decoding using trained SpeechLAS Model
        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences 
        :param argparse.Namespace params: The decoding options
        """
        encoder_output,hlens,ctc_out = self.encoder(xs,xlens)
        ys_pred = self.decoder.greedy_decode(encoder_output,hlens,ctc_out,decode_params)
        return ys_pred                

    def decode_beam(self,xs: torch.Tensor,xlens: torch.LongTensor,decode_params:configargparse.Namespace):
        """
        Performs Beam Decoding using trained SpeechLAS Model
        :params torch.Tensor xs- Speech feature input
        :params list xlens- Lengths of unpadded feature sequences 
        :param argparse.Namespace params: The decoding options
        """
        raise NotImplementedError
    
    