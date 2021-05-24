
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

    embedding = nn.Embedding(ntokens, ninp)
    classifier = nn.Linear(nhid, ntokens)

    if "LSTM" in model_type or "GRU" in model_type:
        model = RNNModel(model_type, ntokens, ninp, nhid, nlayers, dropout, False, encoder=embedding, decoder=classifier)
    elif model_type == "Transformer":
        model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout, encoder=embedding, decoder=classifier)

    model.init_weights()

    return model
