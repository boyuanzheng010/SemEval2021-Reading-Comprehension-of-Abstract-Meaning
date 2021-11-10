# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import argparse
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets
from torchtext import vocab

from Baselines.Utils.utils import word_tokenize, get_device, epoch_time, classifiction_metric
from Baselines.Utils.arc_embedding_utils import load_data


def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir,
          print_step, clip):
    model.train()
    logging_dir = log_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    writer = SummaryWriter(
        log_dir=logging_dir)

    global_step = 0
    best_dev_loss = float('inf')
    best_acc = 0.0

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch + 1:02} ----------')

        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            optimizer.zero_grad()

            logits = model(batch)

            loss = criterion(logits.view(-1, len(label_list)), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:

                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list)
                c = global_step // print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/dev", dev_loss, c)

                writer.add_scalar("acc/train", train_acc, c)
                writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                print_list = ['macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                # if dev_loss < best_dev_loss:
                #     best_dev_loss = dev_loss

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(), out_model_file)

                model.train()

    writer.close()


def evaluate(model, iterator, criterion, label_list):
    model.eval()

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    with torch.no_grad():
        for batch in iterator:
            with torch.no_grad():
                logits = model(batch)

            loss = criterion(logits.view(-1, len(label_list)), batch.label)

            labels = batch.label.detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    acc, report = classifiction_metric(
        all_preds, all_labels, label_list)

    return epoch_loss / len(iterator), acc, report


def main(config, model_filename):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    model_file = os.path.join(
        config.output_dir, model_filename)

    # Prepare the device
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    # Set Random Seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    # Prepare the data
    id_field = data.RawField()
    id_field.is_target = False
    text_field = data.Field(tokenize='spacy', lower=True,
                            include_lengths=True)
    label_field = data.LabelField(dtype=torch.long)

    train_iterator, dev_iterator, test_iterator = load_data(
        config.data_path, id_field, text_field, label_field, config.train_batch_size, config.dev_batch_size,
        config.test_batch_size, device, config.glove_word_file, config.cache_dir)

    # Word Vector
    word_emb = text_field.vocab.vectors

    if config.model_name == "GAReader":
        from Baselines.GAReader.GAReader import GAReader
        model = GAReader(
            config.glove_word_dim, config.output_dim, config.hidden_size,
            config.rnn_num_layers, config.ga_layers, config.bidirectional,
            config.dropout, word_emb)
        print(model)

    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if config.do_train:
        train(config.epoch_num, model, train_iterator, dev_iterator, optimizer, criterion, ['0', '1', '2', '3', '4'],
              model_file, config.log_dir, config.print_step, config.clip)

    model.load_state_dict(torch.load(model_file))

    test_loss, test_acc, test_report = evaluate(
        model, test_iterator, criterion, ['0', '1', '2', '3', '4'])
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))


if __name__ == "__main__":

    model_name = "GAReader"
    data_dir = "C:/Users/Dell/Desktop/sample/multi_sample/"
    embedding_folder = "C:/Users/Dell/Desktop/mrc_data/embedding/glove/"
    output_dir = "./ga/output"
    cache_dir = "./ga/cache"
    log_dir = "./ga/log/"

    model_filename = "model_adam1.pt"

    if model_name == "GAReader":
        from Baselines.GAReader import args, GAReader

        main(args.get_args(data_dir, cache_dir, embedding_folder, output_dir, log_dir), model_filename)
