from fairseq.tasks import register_task 
from fairseq.tasks.translation import TranslationTask
import torch


@register_task('text_summarization_annealing')
class TextSummarization(TranslationTask):
    """Translation Task"""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        # parser.add_argument("--beta", default=1.0)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

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
            (loss, kld_loss),sample_size, logging_output = criterion(model, sample, update_num)
        
        FREQ = 30000
        R = 0.99
        tau = (update_num%FREQ)/FREQ
        beta = (tau/R)**5/10 if tau<=R else 1/10    
        elbo = loss + beta*kld_loss
        if ignore_grad:
            elbo *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(elbo)
        return elbo, sample_size, logging_output

    
