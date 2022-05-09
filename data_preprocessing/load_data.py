import torch
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import gensim


class PlaylistDataset(Dataset):
    def __init__(self, word2vec, num_rows_train):
        # data loading
        self.word2vec = word2vec
        self.num_rows_train = num_rows_train
        self.src, self.trg, self.trg_len = self.read_train_data()
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.trg_len[index]

    def __len__(self):
        return self.n_samples

    def read_train_data(self):
        # read training data from "track_sequences"
        src_uri = []
        trg_uri = []
        with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if index >= self.num_rows_train:
                    break
                elif len(row) > 3:
                    is_odd = (len(row) - 2) % 2 == 1
                    i = int(len(row) / 2 + 1)
                    src_i = row[2:i]
                    trg_i = row[i:len(row)]
                    if is_odd:
                        trg_i = row[i:len(row) - 1]
                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
            # create lists of track indices according to the indices of the word2vec model
            src_idx = []
            trg_idx = []
            trg_len = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                indices = []
                for uri in trg_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                trg_idx.append(torch.LongTensor(indices))
                trg_len.append(len(indices))
        return src_idx, trg_idx, trg_len


class NextTrackDataset(Dataset):
    def __init__(self, word2vec, num_rows_train):
        # data loading
        self.word2vec = word2vec
        self.num_rows_train = num_rows_train
        self.src, self.trg = self.read_train_data()
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return self.n_samples

    def read_train_data(self):
        # read training data from "track_sequences"
        src_uri = []
        trg_uri = []
        with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if index >= self.num_rows_train:
                    break
                elif len(row) > 3:
                    src_i = row[2:-2]
                    trg_i = row[-1]
                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
            # create lists of track indices according to the indices of the word2vec model
            src_idx = []
            trg_idx = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                trg_index = self.word2vec.wv.get_index(trg_uri[i])
                trg_idx.append(trg_index)
        return src_idx, trg_idx


def collate_fn(data):
    """
    args:       data: list of tuple (src, trg)
    return:     padded_src - Padded src, tensor of shape (batch_size, padded_length)
                length - Original length of each sequence(without padding) tensor of shape (batch_size)
                padded_trg - Padded trg, tensor of shape (batch_size, padded_length)
    """
    src, trg, trg_len = zip(*data)
    src = pad_sequence(src, batch_first=True)
    trg = pad_sequence(trg, batch_first=True, padding_value=-1)
    return src, torch.LongTensor(trg), trg_len

def collate_fn_next_track(data):
    """
    args:       data: list of tuple (src, trg)
    return:     padded_src - Padded src, tensor of shape (batch_size, padded_length)
                length - Original length of each sequence(without padding) tensor of shape (batch_size)
                padded_trg - Padded trg, tensor of shape (batch_size, padded_length)
    """
    src, trg = zip(*data)
    src = pad_sequence(src, batch_first=True)
    trg = pad_sequence(trg, batch_first=True, padding_value=-1)
    return src, torch.LongTensor(trg), trg_len


def get_word2vec_model(word2vec_model):
    print("load word2vec from file")
    model = gensim.models.Word2Vec.load("./models/gensim_word2vec/" + word2vec_model + "/word2vec-song-vectors.model")
    print("word2vec loaded from file")
    return model
