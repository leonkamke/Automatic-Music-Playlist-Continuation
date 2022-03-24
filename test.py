import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def create_train_valid_test_data(num_rows_train, num_rows_valid, num_rows_test):
    # read training data from "track_sequences"
    # rows(track_sequences) = 1.000.000 (0 to 999.999)
    train_data = []
    valid_data = []
    test_data = []
    with open('data/spotify_million_playlist_dataset_csv/data/id_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row_str in enumerate(csv_reader):
            row = [int(id) for id in row_str]
            if index >= num_rows_train + num_rows_valid + num_rows_test:
                break
            elif index < num_rows_train:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.Tensor(row[0:i])
                trg = torch.Tensor(row[i: len(row)])
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.Tensor(row[0:i])
                trg = torch.Tensor(row[i: len(row)])
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = torch.Tensor(row[0:i])
                trg = torch.Tensor(row[i: len(row)])
                test_data.append([src, trg])
    # shape of train data: (num_rows_train, 2)
    # shape of validation data: (num_rows_valid, 2)
    # shape of test data: (num_rows_test, 2)
    return np.array(train_data), np.array(valid_data), np.array(test_data)

def create_dict():
    with open('data/spotify_million_playlist_dataset_csv/data/vocabulary.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        track_to_index = {}
        index_to_track = {}
        for index, row in enumerate(csv_reader):
            track_to_index[row[1]] = int(row[0])
            index_to_track[row[0]] = row[1]
        return track_to_index, index_to_track


# skript for creating the vocabulary (list of all tracks)
if __name__ == '__main__':
    BATCH_SIZE = 11
    train_data, valid_data, test_data = create_train_valid_test_data(1000, 10, 10)
    embedding = nn.Embedding(10000, 100)
    x_seq = [torch.tensor([5, 18, 29], dtype=torch.int32), torch.tensor([32, 100], dtype=torch.int32), torch.tensor([699, 6, 9, 17], dtype=torch.int32)]

    # pad the input sequence with zeros
    x_padded = pad_sequence(x_seq, batch_first=True, padding_value=0)
    print(x_padded)
    # sort the list of sequences by sequence length
    lengths = torch.IntTensor([torch.max(x_padded[i, :].data.nonzero()) + 1 for i in range(x_padded.size()[0])])
    lengths, perm_idx = lengths.sort(0, descending=True)
    x_padded = x_padded[perm_idx][:, :lengths.max()]
    print(x_padded)
    # save the length in a list

    packed_sequences = pack_padded_sequence(x_padded, batch_first=True, lengths=lengths, enforce_sorted=True)
    print(packed_sequences)
    embedding(packed_sequences.data)
    # x_padded = [[5, 18, 29, 0], [32, 100, 0, 0], [699, 6, 9, 17]]

