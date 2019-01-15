import os
import time
import pickle
import torch
import torch.nn as nn
from collections import namedtuple

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from color import settings
from color.download_data import read_data
from color.utils import Vocab, BatchIterator, lab2hex, lab2rgb
from color.model import BiLSTM


cuda = False

if cuda:
    torch.backends.cudnn.benchmark = True

train_path = os.path.join(settings.DATA_DIR, 'corpus.data.gzip')

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

Name = namedtuple('Name', ['index', 'name', 'color'])
pickled_vocab_path = os.path.join(settings.DATA_DIR, 'vocab.pickle')


def get_color(line):
    if not os.path.exists(pickled_vocab_path):
        train_names = read_data(train_path)
        vocab = Vocab(train_names, min_count=min_count, add_padding=True)
        with open(pickled_vocab_path, "wb") as f:
            pickle.dump(vocab, f)
    else:
        with open(pickled_vocab_path, "rb") as f:
            vocab = pickle.load(f)

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
        weights = os.path.join(settings.MODEL_OUTPUT_DIR, 'model_fitted.pt')
        model.cuda()
    else:
        weights = os.path.join(settings.MODEL_OUTPUT_DIR, 'model_fitted_cpu.pt')

    model.load_state_dict(torch.load(weights))

    def format_data(line):
        return [Name(index=0, name=line, color=(0., 0., 0.))]

    train_batches = BatchIterator(
        format_data(line),
        vocab, batch_size, cuda=cuda)

    for i, batch in enumerate(train_batches):
        (id_sice, padded_x_slice, x_slice_lengths, y_slice) = batch
        _, logits = model.forward(padded_x_slice, x_slice_lengths, y_slice)
        break

    onecolor = (logits.cpu() if cuda else logits).squeeze().detach().numpy()
    l, a, b = onecolor
    color_obj = {
        "hex": lab2hex(l, a, b),
        "rgb": lab2rgb(l, a, b),
        "lab": [str(l), str(a), str(b)]
    }
    return color_obj


def get_color_in_shell():
    result = input("Give me your color name: ")
    color_obj = get_color(result)
    print("s:", color_obj["hex"])
    print("rgb:", color_obj["rgb"])
    print("lab:", color_obj["lab"])

    time.sleep(1)
