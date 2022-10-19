import torch.nn as nn
from typing import Union

def init(
    module: Union[nn.Linear, nn.Conv2d], 
    weight_init: nn.init, 
    bias_init: nn.init, 
    gain: float = 1.0,
):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
