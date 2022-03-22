import numpy as np
import pandas as pd
import csv

def create_train_valid_test_data(num_rows_train, num_rows_valid, num_rows_test):
    # read training data from "track_sequences"
    # rows(track_sequences) = 1.000.000 (0 to 999.999)
    train_data = []
    valid_data = []
    test_data = []
    with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file
        for index, row in enumerate(csv_reader):
            if index == num_rows_train + num_rows_valid + num_rows_test:
                break
            elif index < num_rows_train:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = np.array(row[2:i])
                trg = np.array(row[i: len(row)])
                train_data.append([src, trg])
            elif index < num_rows_train + num_rows_valid:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = np.array(row[2:i])
                trg = np.array(row[i: len(row)])
                valid_data.append([src, trg])
            else:
                if len(row) <= 3:
                    continue
                i = int(len(row) / 2 + 1)
                src = np.array(row[2:i])
                trg = np.array(row[i: len(row)])
                test_data.append([src, trg])
    train_data = np.array(train_data, dtype=object)     # shape of train data: (num_rows_train, 2)
    valid_data = np.array(valid_data, dtype=object)     # shape of validation data: (num_rows_valid, 2)
    test_data = np.array(test_data, dtype=object)       # shape of test data: (num_rows_test, 2)
    return train_data, valid_data, test_data

if __name__ == '__main__':
    train_data, valid_data, test_data = create_train_valid_test_data(100000, 1000, 1000)
    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)