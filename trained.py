import torch
import torch.nn as nn
from collections import namedtuple
from data import read_data
from utils import Vocab, BatchIterator, lab2hex
from model import BiLSTM
import fileinput

cuda = False

if cuda:
    torch.backends.cudnn.benchmark = True

train_path = 'corpus.data'

epochs = 20
batch_size = 1
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

print("Reading the data")
Name = namedtuple('Name', ['index', 'name', 'color'])
train_names = read_data(train_path)
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
    weights = 'model_fitted.pt'
    model.cuda()
else:
    weights = 'model_fitted_cpu.pt'

model.load_state_dict(torch.load(weights))

print(model)


def format_data(line):
    return [Name(index=0, name=line, color=(0., 0., 0.))]


print('give me your color:')
for line in fileinput.input():
    train_batches = BatchIterator(
        format_data(line),
        vocab, batch_size, cuda=cuda)

    for i, batch in enumerate(train_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        _, logits = model.forward(padded_x_slice, x_slice_lengths, y_slice)
        break

    onecolor = (logits.cpu() if cuda else logits).squeeze().detach().numpy()
    print("getting logits", onecolor)
    l, a, b = onecolor
    print(lab2hex(l, a, b))
    print('give me your color:')
