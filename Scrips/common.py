"""Shared neural-network utilities.

This module keeps small, stable helpers that are imported by multiple model
implementations.  It intentionally avoids depending on project configuration so
model modules can be imported independently.
"""

import torch.nn as nn


def init_network_weights(net: nn.Module, std: float = 0.01) -> None:
    """Initialize Linear and LayerNorm layers in a model.

    Parameters
    ----------
    net:
        Module tree whose child layers should be initialized.
    std:
        Standard deviation used for Linear layer weights.

    Notes
    -----
    This mirrors the initializer already used by the ODE-RNN implementation and
    restores the public helper imported by ``models/SNN.py``.
    """
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
