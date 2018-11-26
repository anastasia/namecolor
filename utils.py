import torch
import random
from torch.autograd import Variable

random.seed(42)

UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'


class VocabItem:

    def __init__(self, string, hash=None):
        self.string = string
        self.count = 0
        self.hash = hash

    def __str__(self):
        return 'VocabItem({})'.format(self.string)

    def __repr__(self):
        return self.__str__()


def tokenizer(x):
    return list(x)


def token_function(x):
    return x.lower()


class Vocab:
    def __init__(self, sequences, tokenizer=tokenizer,
                 token_function=token_function,
                 min_count=0, add_padding=False, add_bos=False,
                 add_eos=False, unk=None):

        vocab_items = []
        vocab_hash = {}
        item_count = 0

        self.token_function = token_function
        self.tokenizer = tokenizer
        self.special_tokens = []

        self.UNK = None
        self.PAD = None
        self.BOS = None
        self.EOS = None

        index2token = []
        token2index = {}

        for sequence in sequences:
            for item in tokenizer(sequence.name):
                real_item = token_function(item)
                if real_item not in vocab_hash:
                    vocab_hash[real_item] = len(vocab_items)
                    vocab_items.append(VocabItem(real_item))

                vocab_items[vocab_hash[real_item]].count += 1
                item_count += 1

                # if item_count % 100 == 0:
                #    print("Reading item {}".format(item_count))

        tmp = []
        if unk:
            self.UNK = VocabItem(unk, hash=0)
            self.UNK.count = vocab_items[vocab_hash[unk]].count
            index2token.append(self.UNK)
            self.special_tokens.append(self.UNK)

            for token in vocab_items:
                if token.string != unk:
                    tmp.append(token)
        else:
            self.UNK = VocabItem(UNK, hash=0)
            index2token.append(self.UNK)
            self.special_tokens.append(self.UNK)

            for token in vocab_items:
                if token.count <= min_count:
                    self.UNK.count += token.count
                else:
                    tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        if add_bos:
            self.BOS = VocabItem(BOS)
            tmp.append(self.BOS)
            self.special_tokens.append(self.BOS)

        if add_eos:
            self.EOS = VocabItem(EOS)
            tmp.append(self.EOS)
            self.special_tokens.append(self.EOS)

        if add_padding:
            self.PAD = VocabItem(PAD)
            tmp.append(self.PAD)
            self.special_tokens.append(self.PAD)

        index2token += tmp
        for i, token in enumerate(index2token):
            token2index[token.string] = i
            token.hash = i

        self.index2token = index2token
        self.token2index = token2index

        print('Unknown vocab size:', self.UNK.count)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.index2token[i]

    def __len__(self):
        return len(self.index2token)

    def __iter__(self):
        return iter(self.index2token)

    def __contains__(self, key):
        return key in self.token2index

    def string2indices(self, string, add_bos=False, add_eos=False):

        string_seq = []
        if add_bos:
            string_seq.append(self.BOS.hash)
        for item in self.tokenizer(string):
            processed_token = self.token_function(item)
            string_seq.append(self.token2index.get(processed_token, self.UNK.hash))
        if add_eos:
            string_seq.append(self.EOS.hash)
        return string_seq

    def indices2tokens(self, indices, ignore_ids=()):

        tokens = []
        for idx in indices:
            if idx in ignore_ids:
                continue
            tokens.append(self.index2token[idx])

        return tokens


class BatchIterator(object):

    def __init__(self,
                 sentences, vocab, batch_size,
                 shuffle=False, cuda=False, ids=None):

        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_id = self.vocab.PAD.hash

        self.id_examples = []
        self.examples = []
        self.y_examples = []

        for i, sentence in enumerate(sentences):
            example = vocab.string2indices(sentence.name)
            self.examples.append(torch.LongTensor(example))

            if sentence.index is not None:
                self.id_examples.append(int(sentence.index))
            else:
                self.id_examples.append(i)

            y_example = [float(i) for i in sentence.color]
            self.y_examples.append(torch.FloatTensor([y_example]))
        assert len(self.examples) == len(self.y_examples)
        self.cuda = self.is_cuda = cuda
        self.num_batches = (len(self.examples) + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):

        if index >= self.num_batches:
            if self.shuffle:
                c = list(zip(self.id_examples,
                             self.examples,
                             self.y_examples))
                random.shuffle(c)

                (self.id_examples, self.examples, self.y_examples) = zip(*c)

            raise IndexError("index is greater "
                             "than the number of batches")

        id_slice = self.id_examples[index * self.batch_size:(index + 1) * self.batch_size]
        x_slice = self.examples[index * self.batch_size:(index + 1) * self.batch_size]
        y_slice = self.y_examples[index * self.batch_size:(index + 1) * self.batch_size]

        padded_x_slice, x_slice_lengths = pad_list(x_slice, pad_value=self.pad_id)

        y_slice = torch.stack(y_slice).transpose(0, 1)

        padded_x_slice = Variable(padded_x_slice)
        y_slice = Variable(y_slice)

        if self.cuda:
            padded_x_slice = padded_x_slice.cuda()
            y_slice = y_slice.cuda()

        return id_slice, padded_x_slice, x_slice_lengths, y_slice


def pad_list(raw_input_list, dim0_pad=None, dim1_pad=None, align_right=False, pad_value=0):
    input_list = [torch.LongTensor(sublist) for sublist in raw_input_list]

    if not dim0_pad:
        dim0_pad = len(input_list)

    if not dim1_pad:
        dim1_pad = max(x.size(0) for x in input_list)

    out = input_list[0].new(dim0_pad, dim1_pad).fill_(pad_value)

    lengths = []

    for i in range(len(input_list)):
        data_length = input_list[i].size(0)
        data_length = data_length if data_length < dim1_pad else dim1_pad
        lengths.append(data_length)
        offset = dim1_pad - data_length if align_right else 0
        out[i].narrow(0, offset, data_length).copy_(input_list[i])

    out = out.t()
    return out, lengths
