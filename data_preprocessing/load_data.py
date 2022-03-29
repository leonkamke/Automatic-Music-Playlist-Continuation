import numpy as np
import torch
import csv
from torch.nn.utils.rnn import pad_sequence


# Return sequences of the track indices with padding at the and
def create_train_data(num_rows_train, num_rows_valid, num_rows_test, word2vec):
    # read training data from "track_sequences"
    src_uri = []
    trg_uri = []
    with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row in enumerate(csv_reader):
            if index >= num_rows_train:
                break
            elif index < num_rows_train and len(row) > 3:
                    is_odd = (len(row)-2) % 2 == 1
                    i = int(len(row) / 2 + 1)
                    src_i = row[2:i]
                    trg_i = row[i:len(row)]
                    if is_odd:
                        trg_i = row[i:len(row)-1]

                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
        src_idx = []
        trg_idx = []
        for i in range(len(src_uri)):
            indices = []
            for uri in src_uri[i]:
                indices.append(word2vec.wv.get_index(uri))
            src_idx.append(torch.LongTensor(indices))
            indices = []
            for uri in trg_uri[i]:
                indices.append(word2vec.wv.get_index(uri))
            trg_idx.append(torch.LongTensor(indices))
        src = pad_sequence(src_idx, batch_first=True)
        trg = pad_sequence(trg_idx, batch_first=True, padding_value=-1)
    # src is padded with zeros; trg is padded with -1
    return src, trg


def get_max_len(list):
    max_len = 0
    for tensor in list:
        if len(tensor) > max_len:
            max_len = len(tensor)
    return max_len

def get_min_len(list):
    min_len = np.inf
    for tensor in list:
        if len(tensor) < min_len:
            min_len = len(tensor)
    return min_len


if __name__ == '__main__':
    print("Hello world")