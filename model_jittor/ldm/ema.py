"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/07/06 10:43:01
@Description: 

"""
import jittor as jt
import jittor.nn as nn
import copy


class EMAModel():
    def __init__(self, model: nn.Module, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def step(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.detach().clone() + decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        raise NotImplementedError('still has bug')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.detach().clone()
                self.model[name] = self.shadow[name]

    def restore(self):
        raise NotImplementedError('still has bug')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                self.model[name] = self.backup[name]
        self.backup = {}


class EMAModel_V2:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
    ):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()

        self.decay = 0.0
        self.num_updates = 0

    @jt.no_grad()
    def step(self, new_model):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        for key, param in new_model.named_parameters():
            if not param.requires_grad:
                raise ValueError()
            else:
                ema_param = self.decay * ema_params[key] + (1 - self.decay) * param

            ema_state_dict[key] = ema_param

        self.averaged_model.load_state_dict(ema_state_dict)
        self.num_updates += 1
