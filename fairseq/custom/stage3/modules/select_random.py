
import torch 
import torch.nn as nn
import random 
import numpy

class SelectRandomLayer(nn.Module):
    def __init__(self):
        super(SelectRandomLayer, self).__init__()
        self.ratio = 0.7
        self.linear = nn.Linear(10,10) # Gradient Plot에서 필요함

    def forward(self, x):
        """
        X: Token, Batch, X_dim
        """
        pop_index = [i for i in range(x.size(0))]
        index = []
        for i in range(int(x.size(0)*self.ratio)):
            a = pop_index.pop(numpy.random.randint(len(pop_index)))
            index.append(a)
        index.sort()
        
        x1 = x.index_select(0, torch.tensor(index).cuda())
        x2 = x.index_select(0, torch.tensor(pop_index).cuda())
        return x1, x2
