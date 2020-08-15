# -*- coding: utf-8 -*-

import torch
from torchtext import data
from torchtext import datasets
from torchtext import vocab


def load_data(path, id_field, word_field, label_field, train_batch_size, dev_batch_size, test_batch_size, device, word_embed_file, cache_dir):

    fields = {
        'article': ('article', word_field),
        'question': ('question', word_field),
        'option_0': ('option_0', word_field),
        'option_1': ('option_1', word_field),
        'option_2': ('option_2', word_field),
        'option_3': ('option_3', word_field),
        'option_4': ('option_4', word_field),
        'label': ('label', label_field)
    }

    word_vectors = vocab.Vectors(word_embed_file, cache_dir)

    train, dev, test = data.TabularDataset.splits(
        path=path, train='train.jsonl', validation='dev.jsonl',
        test='test.jsonl', format='json', fields=fields)
    
    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(dev.examples), len(test.examples)))
    
    word_field.build_vocab(train, dev, test, max_size=50000,
                           vectors=word_vectors, unk_init=torch.Tensor.normal_)
    
    label_field.build_vocab(train, dev, test)
    
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(train_batch_size, dev_batch_size, test_batch_size), sort_key=lambda x: len(x.article), device=device, shuffle=True)

    return train_iter, dev_iter, test_iter

