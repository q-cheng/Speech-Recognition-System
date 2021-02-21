import os
import numpy as np
import torch 
import editdistance
from torch.nn import CrossEntropyLoss, CTCLoss
import torch.nn.functional as F
import math 
import configargparse


class StatsCalculator(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace):
        """
        Calculates Loss, Accuracy, Perplexity Statistics
        : param argparse.Namespace params: The training options
        """
        super(StatsCalculator,self).__init__()
        self.ignore_label = params.text_pad
        self.char_list = params.char_list 
        self.criterion = CrossEntropyLoss(ignore_index=self.ignore_label,reduction="mean")
        self.ctc = CTCLoss(zero_infinity=True)
        
    def compute_accuracy(self,pad_pred:torch.Tensor, pad_targets:torch.Tensor):
        """
        Computes the Token Accuracy using the Predictions and Targets with SOS
        :param torch.Tensor pad_pred: The logits from the decoder 
        :param torch.Tensor pad_targets: The targets with SOS padding 
        :returns float accuracy: Accuracy between Predicted tokens and targets
        """
        pad_pred = pad_pred.argmax(2)
        mask = pad_targets != self.ignore_label
        numerator = torch.sum(
            pad_pred.masked_select(mask) == pad_targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
        return float(numerator) / float(denominator)

    def compute_perplexity(self,loss: float):
        """
        Computes the perplexity as the exponential of the loss
        :param float loss: The CE Loss
        :returns float perplexity
        """
        return math.exp(loss)
        
    def compute_wer(self,pad_pred:torch.Tensor, pad_targets:torch.Tensor):
        """
        Computes the Word Error Rate using the Predictions and Targets with SOS
        :param torch.Tensor pad_pred: The logits from the decoder 
        :param torch.Tensor pad_targets: The targets with SOS padding 
        :returns float wer_score: WER between Predicted tokens and targets
        """
        ## Get HYP and REF Sentences using pad_pred and pad_targets
        target_tokens = [y[y!= self.ignore_label] for y in pad_targets]
        pred_lens = [len(y[y!= self.ignore_label]) for y in pad_targets]
        pred_pad = np.argmax(F.log_softmax(pad_pred,dim=-1).detach().cpu().numpy(),axis=-1) 
        pred_tokens = [y[:pred_lens[i]] for i,y in enumerate(pred_pad)]
        self.pred_tokens = pred_tokens
        ref_lens = []
        word_eds = []
        for tgt,pred in zip(target_tokens,pred_tokens):
            self.ref = "".join([self.char_list[x] for x in tgt if x != -1]).replace("<space>"," ").replace("<eos>","")
            self.hyp = "".join([self.char_list[x] for x in pred]).replace("<space>"," ").replace("<eos>","")
            word_eds.append(editdistance.eval(self.ref.split(' '),self.hyp.split(' ')))
            ref_lens.append(len(self.ref.split(' ')))
        return float(sum(word_eds))/sum(ref_lens)

    def forward(self,pad_pred:torch.Tensor, pad_targets:torch.Tensor, ylen, ctc_out, hlens): # , ylen, encoder_output, hlens
        """
        Computes the loss, accuracy, wer
        :param torch.Tensor pad_pred: The logits from the decoder 
        :param torch.Tensor pad_targets: The targets with SOS padding
        :returns float wer_score: WER between Predicted tokens and targets
        :returns float accuracy: Accuracy between Predicted tokens and targets
        :returns torch.Tensor loss: CELoss between Predicted tokens and targets
        """
        batch_size,maxlen,_ = pad_pred.size()
        self.loss = self.criterion(pad_pred.view(batch_size*maxlen,-1),pad_targets.view(-1))
        ppl = self.compute_perplexity(self.loss.item())
        self.loss *= (np.mean([ len(x[x!=self.ignore_label]) for x in pad_targets]) - 1)

        ## Try to add CTC loss.
        ylen_tensor = torch.LongTensor([length for length in ylen])
        hlens_tensor = torch.LongTensor([length for length in hlens])
        ctc_pad_targets = torch.clone(pad_targets)
        ctc_pad_targets[ctc_pad_targets==-1] = 0
        self.loss = 0.9 * self.loss + 0.1 * \
                    self.ctc(ctc_out.permute(1, 0, 2).log_softmax(-1), 
                    ctc_pad_targets, hlens_tensor, ylen_tensor)

        if self.training:
            return self.loss,self.compute_accuracy(pad_pred,pad_targets),None
        else:
            return self.loss,self.compute_accuracy(pad_pred,pad_targets),self.compute_wer(pad_pred,pad_targets)

            

def pad_list(xs: torch.Tensor, pad_value: int):
    """
    Performs padding for the list of tensors.
    :param xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
    :param pad_value (float): Value for padding.
    : returns Tensor: Padded tensor (B, Tmax, `*`).
    Example:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

def to_device(m: torch.nn.Module, x:torch.Tensor):
    """
    Sends tensor into the device of the module.
    :params torch.nn.Module m : Torch module.
    :params torch.tensor x: Torch tensor.
    :returns torch.Tensor: Torch tensor located in the same device as torch module.
    """
    if isinstance(m, torch.nn.Module):
        device = next(m.parameters()).device
    elif isinstance(m, torch.Tensor):
        device = m.device
    else:
        raise TypeError(
            "Expected torch.nn.Module or torch.tensor, " f"bot got: {type(m)}"
        )
    return x.to(device)

def make_pad_mask(lengths:list, xs:torch.Tensor=None, length_dim:int=-1):
    """
    Creates a mask tensor containing indices of padded part.
    :param torch.LongTensor or list lengths: Batch of lengths (B,).
    :param torch.Tensor xs (optional): The reference tensor. If set, masks will be the same shape as this tensor.
    :param int length_dim (optional): Dimension indicator of the above tensor.
    :returns torch.boolTensor mask: Mask tensor containing indices of padded part.
    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


