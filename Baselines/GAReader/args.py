# -*- coding: utf-8 -*-

import argparse


def get_args(data_dir, cache_dir, embedding_folder, model_dir, log_dir):

    parser = argparse.ArgumentParser(description='ReCAM')

    parser.add_argument("--model_name", default="GAReader",
                        type=str, help="name of the model")

    parser.add_argument("--seed", default=1234, type=int, help="random seed")

    # data_util
    parser.add_argument(
        "--data_path", default=data_dir, type=str, help="path of the data")

    parser.add_argument(
        "--cache_dir", default=cache_dir, type=str, help="path of the cache"
    )

    parser.add_argument(
        "--sequence_length", default=800, type=int, help="length of sentence"
    )

    # 输出文件名
    parser.add_argument(
        "--output_dir", default=model_dir + "GAReader/", type=str, help="output path of the model"
    )
    parser.add_argument(
        "--log_dir", default=log_dir + "GAReader/", type=str, help="path of logging file"
    )

    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--print_step", default=200,
                        type=int, help="save the model after how many steps")
    
    # 模型参数
    parser.add_argument("--output_dim", default=5, type=int)
                        
    # 优化参数
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)

    parser.add_argument("--epoch_num", default=30, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")

    parser.add_argument("--clip", default=10, type=int, help="")

    # LSTM 参数
    parser.add_argument("--hidden_size", default=128, type=int, help="the number of dimension of hidden size")
    parser.add_argument('--rnn_num_layers', default=1, type=int, help='number of RNN layer')
    parser.add_argument("--bidirectional", default=True, type=bool)

    # GAReader
    parser.add_argument('--ga_layers', default=1, type=int, help='the layer of GAReader')

    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu id")
    
    # word Embedding
    parser.add_argument(
        '--glove_word_file',
        default=embedding_folder + 'glove.840B.300d.txt',
        type=str, help='path of word embedding file')
    parser.add_argument(
        '--glove_word_size',
        default=int(2.2e6), type=int,
        help='Corpus size for Glove')
    parser.add_argument(
        '--glove_word_dim',
        default=300, type=int,
        help='word embedding size (default: 300)')

    config = parser.parse_args()

    return config
