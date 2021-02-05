# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
import torch 
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import rouge_score
from rouge_score import rouge_scorer

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)



@register_task("simple_vae_translation")
class SimpleVaeTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.period = int(args.period)
        self.store_gradient = False


        self.mse_loss = nn.MSELoss()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
    @staticmethod
    def add_args(parser):    
        TranslationTask.add_args(parser)
        parser.add_argument('--period', 
                            help='use pretrained model when training [True, ...]')
        parser.add_argument("--grad")

    def calculate_rouge(self, tensor1, tensor2):
        rouges = []
               

        for i in range(tensor1.size(0)):
            s1, s2 = tensor1[i,:], tensor2[i,:]
            s1, s2 = s1.tolist(), s2.tolist()
            s1, s2 = " ".join(list(map(str, s1))), " ".join(list(map(str, s2)))
            rouge = self.rouge_scorer.score(s1,s2)['rouge1'][2]
            rouges.append(rouge)
        return torch.tensor(rouges).cuda().unsqueeze(dim=1)

    def optimize_rouge(self, sample, model, num_updates ,ignore_grad=False):
        optimizer = torch.optim.Adam(model.rouge_predict_net.parameters(), lr=1e-4)
        net_input = sample['net_input']['src_tokens']      # [B, T1]
        src_lengths = sample['net_input']['src_lengths']
        target = sample['target']                          # [B, T2]

        # --- predict ROUGE score
        encoder_out = model.encoder.forward(net_input, src_lengths)
        alpha = model.rouge_predict_net(encoder_out['encoder_out'][0])
        
        # --- calculate ROUGE score
        rouges = self.calculate_rouge(net_input, target)
        loss = self.mse_loss(alpha, rouges)


        # --- optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)

        rouge_loss = self.optimize_rouge(sample, model, update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        
        logging_output['rouge_loss'] = rouge_loss.item()

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
            if self.store_gradient and update_num%50==0:
                target_names = [ "XZtoV.weight", "XtoV.weight", "ZtoV.weight"]
                dic = {}
                for name, param in model.named_parameters():
                    if name in target_names:
                        #dic[name]=param.grad.data.norm(2).item()
                        dic[name]=param.grad.data.abs().mean().item()
                        #torch.save(param.grad.data.norm(2), f"{self.gradient_dir}/{name}_{update_num}.pt")
                # self.writer.add_scalars("Grad/Norm", dic, update_num)
        
        # if update_num==200:
        #     for param in model.encoder.parameters():
        #         param.requires_grad = False
        
        return loss, sample_size, logging_output
