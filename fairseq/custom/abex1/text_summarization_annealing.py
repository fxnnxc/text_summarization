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

import numpy as np
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
        self.store_gradient = True
        
        
        self.gradient_dir = None
        i = 1
        while True:
            self.gradient_dir = "gradients/grad"+str(i)
            if not os.path.isdir(self.gradient_dir):
                break
            i +=1
        os.mkdir(self.gradient_dir)
        
        

    @staticmethod
    def add_args(parser):    
        TranslationTask.add_args(parser)
        parser.add_argument('--period', 
                            help='use pretrained model when training [True, ...]')
        parser.add_argument("--grad")
        
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
        with torch.autograd.profiler.record_function("forward"):
            (loss,kld_loss), sample_size, logging_output = criterion(model, sample)
            FREQ = self.period
            R = 0.99
            tau = (update_num%FREQ)/FREQ
            beta = (tau/R)**5 if tau<=R else 1
            beta /=24660
            elbo = loss + beta*kld_loss
        if ignore_grad:
            elbo *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(elbo)
            for name, param in model.named_parameters():
                print(name)
            assert False

            if self.store_gradient and update_num%10==0:
                target_names = ["XZtoV.weight", "XtoV.weight", "ZtoV.weight"]
                for name, param in model.named_parameters():
                    if name in target_names:
                        torch.save(param.grad.data.norm(2), f"{self.gradient_dir}/{name}_{update_num}.pt")
        return elbo, sample_size, logging_output
