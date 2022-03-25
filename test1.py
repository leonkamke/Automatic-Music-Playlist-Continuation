import torch
import torch.nn as nn
import gensim
import csv

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
    model = gensim.models.Word2Vec.load("./models/gensim_word2vec/word2vec-song-vectors.model")

    # 'spotify:track:74tqql9zP6JjF5hjkHHUXp', 'spotify:track:4erhEGuOGQgjv3p1bccnpn',
    # 'spotify:track:4hRA2rCPaCOpoEIq5qXaBz', 'spotify:track:1enx9LPZrXxaVVBxas5rRm'

    weights = torch.FloatTensor(model.wv.vectors)
    # weights.shape == (169657, 100)
    #print(weights.shape)
    x = model.wv.

    # train emb_dir and give the seq2seq model the gensim model as a parameter
