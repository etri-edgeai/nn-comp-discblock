""" Model Loader

"""

from torch import nn

from .lm import RNNModel, TransformerModel

def load_model(
    model_type,
    ntokens,
    ninp,
    nhid,
    nlayers,
    dropout=0.5,
    nhead=None,
    device="cuda:0"):
    """ Model loader """

    model = RNNModel(
            model_type, ntokens, ninp, nhid, nlayers, dropout, False)

    model.init_weights()

    return model
