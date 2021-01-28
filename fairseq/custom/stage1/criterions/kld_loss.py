# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("kld_loss")
class KLDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.count = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        with torch.autograd.set_detect_anomaly(True):
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=True)
            sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            #--------------------------------------------------------------------------------------------
            x, extra = net_output
            x_lv, y_lv = extra['x_lv'],  extra['y_lv']
            x_mu, y_mu = extra['x_mu'],  extra['y_mu']
            x_z, y_z   = extra['x_z'],   extra['y_z']
            #kld_loss = torch.nn.functional.mse_loss(x_mu,y_mu) + torch.nn.functional.mse_loss(x_lv, y_lv)
            #l_x =  torch.logsumexp(torch.cat([l_x1.unsqueeze(0), l_x2.unsqueeze(0)]), dim=0)
            #kld_loss1 = torch.mean(-0.5 * torch.sum(1+ l_x1 - m_x1**2- l_x1.exp(), dim=1), dim=-1)
            kld_loss = - 0.5 * (1 + y_lv - x_lv 
                                   - torch.div(torch.pow(y_mu-x_mu, 2), torch.exp(x_lv))
                                   - torch.div(torch.exp(y_lv), torch.exp(x_lv))).sum(dim=1).mean()
            #kld_loss = kld_loss1 #+ kld_loss2
            elbo = loss + kld_loss*sample_size
            #--------------------------------------------------------------------------------------------
            mu_dictionary = {f"x_mu{i}":x_mu[0,0,i].item() for i in range(0, x_mu.size(2), x_mu.size(2)//10 ) }
            lv_dictionary = {f"x_lv{i}":x_lv[0,0,i].item() for i in range(0, x_lv.size(2), x_mu.size(2)//10 ) }

        logging_output = {
            "elbo" :elbo.data,
            "kld"  :kld_loss.data,
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        logging_output.update(mu_dictionary)
        logging_output.update(lv_dictionary)


        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return (loss, kld_loss), sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        elbo_sum = sum(log.get("elbo", 0) for log in logging_outputs)
        kld_sum = sum(log.get("kld", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        for key in logging_outputs[0].keys():
            if "mu" in key or "lv" in key:
                metrics.log_scalar(
                    key, logging_outputs[0][key], 1 , round=3
                )

        metrics.log_scalar(
            "elbo", elbo_sum / sample_size / math.log(2), sample_size, round=4
        )
        
        metrics.log_scalar(
            "kld", kld_sum/sample_size , sample_size, round=4
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=4
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=4
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
