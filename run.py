import torch
import tqdm
import torch.nn as nn
import random
from datetime import datetime
from data import read_data
from utils import Vocab, BatchIterator
from model import BiLSTM

torch.backends.cudnn.benchmark = True

train_path = 'corpus.data'
valid_path = 'valid.data'

epochs = 20
batch_size = 128
hidden_size = 300
log_interval = 10
num_labels = 3
input_dropout = 0.0
output_dropout = 0.0
bidirectional = True
num_layers = 2
min_count = 2
pooling = 'max'
lr = 0.0001

gradient_clipping = 0.25
embedding_size = 300
cuda = True


train_names = read_data(train_path)
valid_names = read_data(valid_path)
# add valid data to training data to bias our colors a bit

train_names += (valid_names * 5)

random.shuffle(train_names)
random.shuffle(valid_names)


vocab = Vocab(train_names, min_count=min_count, add_padding=True)

embeddings = nn.Embedding(len(vocab.index2token),
                           embedding_size,
                           padding_idx=vocab.PAD.hash)

model = BiLSTM(embeddings=embeddings,
               hidden_size=hidden_size,
               num_labels=num_labels,
               input_dropout=input_dropout,
               output_dropout=output_dropout,
               bidirectional=bidirectional,
               num_layers=num_layers,
               pooling=pooling)

if cuda:
    model.cuda()

print(model)

train_batches = BatchIterator(train_names, vocab, batch_size, cuda=cuda)
valid_batches = BatchIterator(valid_names, vocab, batch_size, cuda=cuda)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
pbar = tqdm.trange(epochs, desc='Training...')

for epoch in pbar:
    epoch_total = 0
    epoch_loss = 0
    training_loss = 0
    for i, batch in enumerate(train_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        loss, logits = model.forward(padded_x_slice,
                                     x_slice_lengths,
                                     y_slice)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clipping)

        optimizer.step()
        total = y_slice.size(0)
        epoch_total += total
        epoch_loss += loss.data[0]
        training_loss = epoch_loss
        if i % log_interval == 0 and i > 0:
            pbar.write('Training Loss: {}'.format(epoch_loss / log_interval))
            epoch_total = 0
            epoch_loss = 0

    test_epoch_total = 0
    test_epoch_loss = 0

    for i, batch in enumerate(valid_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        loss, logits = model.forward(padded_x_slice,
                                     x_slice_lengths,
                                     y_slice)

        total = y_slice.size(0)
        test_epoch_total += total
        test_epoch_loss += loss.data[0]

    pbar.write('\n---------------------')
    pbar.write('Validation Loss: {}'.format(test_epoch_loss / len(valid_batches)))
    pbar.write('---------------------\n')
