import torch
import torch.nn as nn
import gensim
import csv
import seq2seq_no_batch_pretrained_emb as seq2seq

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

def create_train_valid_test_data(num_rows_train, num_rows_valid, num_rows_test):
    # read training data from "track_sequences"
    train_data = []
    valid_data = []
    test_data = []
    with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row in enumerate(csv_reader):
            if index >= num_rows_train + num_rows_valid + num_rows_test:
                break
            elif index < num_rows_train:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = row[2:i]
                trg = row[i: len(row)]
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = row[2:i]
                trg = row[i: len(row)]
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = row[2:i]
                trg = row[i: len(row)]
                test_data.append([src, trg])
    # shape of train data: (num_rows_train, 2)
    # shape of validation data: (num_rows_valid, 2)
    # shape of test data: (num_rows_test, 2)
    return train_data, valid_data, test_data


if __name__ == '__main__':
    # predictions.shape = (ln(trg), vocab_size)
    # trg.shape = (ln(trg))
    trg0 = torch.LongTensor([5, 6, 7, 8, 9])
    predictions0 = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    predictions0 = torch.FloatTensor(predictions0)
    trg1 = torch.LongTensor([3])
    predictions1 = [[0, 0, 0, 0.01, 0]]
    predictions1 = torch.FloatTensor(predictions1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions1, trg1)

    print(loss.item())
