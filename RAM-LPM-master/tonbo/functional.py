import numpy as np
import torch


def periodicpad(inputs, pad):
    """
            
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    'periodic' pad, similar to torch.nn.functional.pad 
    # https://github.com/ZichaoLong/aTEAM/blob/master/nn/functional/utils.py
    """
    n = inputs.dim()
    inputs = inputs.permute(*list(range(n - 1, -1, -1)))
    pad = iter(pad)
    i = 0
    indx = []
    for a in pad:
        b = next(pad)
        assert a < inputs.size()[i] and b < inputs.size()[i]
        permute = list(range(n))
        permute[i] = 0
        permute[0] = i
        inputs = inputs.permute(*permute)
        inputlist = [
            inputs,
        ]
        if a > 0:
            inputlist = [inputs[slice(-a, None)], inputs]
        if b > 0:
            inputlist = inputlist + [
                inputs[slice(0, b)],
            ]
        if a + b > 0:
            inputs = torch.cat(inputlist, dim=0)
        inputs = inputs.permute(*permute)
        i += 1
    inputs = inputs.permute(*list(range(n - 1, -1, -1)))
    return inputs
