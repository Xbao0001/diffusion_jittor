import jittor as jt
import jittor.nn as nn


class Embedding(nn.Module):
    def __init__(self, num, dim):
        self.num = num
        self.dim = dim
        self.weight = jt.init.gauss([num, dim], 'float32')

    def execute(self, x):
        res = self.weight[x.flatten()].reshape(x.shape + [self.dim])
        return res


class ModuleList(nn.ModuleList):
    def insert(self, index: int, module: nn.Module):
        for i in range(len(self.layers), index, -1):
            self.layers[str(i)] = self.layers[str(i - 1)]
        self.layers[str(index)] = module
