"""
idea from: https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
see: https://stackoverflow.com/questions/46204503/why-is-my-custom-jsonencoder-default-ignoring-booleans
"""

from functools import singledispatch

import numpy as np
from torch.utils.data import DataLoader


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    return np.float64(val)


@to_serializable.register(DataLoader)
def ts_dataloader(val):
    return "dataloader!"
