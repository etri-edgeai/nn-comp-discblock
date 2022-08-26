import torch
import torchtext
import torch.nn as nn
import time
import math

def get_data(data_name, batch_size, bptt, device):
    """ Data retrieval Function """

    if data_name == "wikitext2":
        train_iters, val_iters, test_iters = torchtext.datasets.WikiText2.iters(batch_size=batch_size, bptt_len=bptt, device=device)
    elif data_name == "wikitext103":
        train_iters, val_iters, test_iters = torchtext.datasets.WikiText103.iters(batch_size=batch_size, bptt_len=bptt, device=device)
    elif data_name == "ptb":
        train_iters, val_iters, test_iters = torchtext.datasets.PennTreebank.iters(batch_size=batch_size, bptt_len=bptt, device=device)
    else:
        raise NotImplementedError()
    return train_iters, val_iters, test_iters

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model_type, model, batch_size, train_iters, val_iters, epoch, lr, clip=0.25, log_interval=200, gate_clamping=None, gate_lr=None):
    """ Training function """

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(train_iters.dataset.fields["text"].vocab)
    criterion = nn.NLLLoss()

    if model_type != 'Transformer':
        hidden = model.init_hidden(batch_size)
    for batch, item in enumerate(train_iters):
        data = item.text
        targets = item.target.view(-1)
        model.zero_grad()
        if model_type == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        sparsity_loss = 0.0
        for c in model.children():
            if type(c).__name__ == "DifferentiableEmbedding" or type(c).__name__ == "DifferentiableEmbeddingClassifier":
                sparsity_loss += c.get_sparsity_loss()
        (loss + sparsity_loss).backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if gate_lr is not None and "gates" in name:
                p.data.add_(-gate_lr, p.grad.data)
            else:
                p.data.add_(-lr, p.grad.data)

        if gate_clamping is not None:
            for c in model.children():
                if type(c).__name__ == "DifferentiableEmbedding":
                    #torch.clamp_(c.gates.weight.data, min=1.0, max=c.embedding.weight.size()[1])
                    if type(c.gates).__name__ == "Embedding":
                        torch.clamp_(c.gates.weight.data, min=gate_clamping[0], max=gate_clamping[1])
                    else:
                        torch.clamp_(c.gates.data, min=gate_clamping[0], max=gate_clamping[1])
                elif type(c).__name__ == "DifferentiableEmbeddingClassifier":
                    #torch.clamp_(c.gates.weight.data, min=1.0, max=c.weight.size()[0])
                    if type(c.gates).__name__ == "Embedding":
                        torch.clamp_(c.gates.weight.data, min=gate_clamping[0], max=gate_clamping[1])
                    else:
                        torch.clamp_(c.gates.data, min=gate_clamping[0], max=gate_clamping[1])
        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_iters), lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(model_type, model, data_source, ntokens, eval_batch_size):
    """ Evaluation Function """ 
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    criterion = nn.NLLLoss()
    if model_type != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    cnt = 0
    with torch.no_grad():
        for item in data_source:
            data = item.text
            targets = item.target.view(-1)
            if model_type == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
            cnt += len(data)
    return total_loss / cnt
