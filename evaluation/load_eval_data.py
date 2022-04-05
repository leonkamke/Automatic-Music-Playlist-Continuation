import torch
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import data_preprocessing.load_data as ld


# Dataset which creates src tracks with size size_seed and the corresponding targets
# uses the last 50.000 playlists
class EvaluationDataset(Dataset):
    def __init__(self, word2vec, last_n_playlists, src_size):
        # data loading
        self.word2vec = word2vec
        self.src_size = src_size
        self.last_n_playlists = last_n_playlists
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
        trg_len = []
        with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if index >= self.last_n_playlists and len(row) > 4+self.src_size:
                    src_i = row[2:2+self.src_size]
                    trg_i = row[2+self.src_size:len(row)]
                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
                    trg_len.append(len(trg_i))
            # create lists of track indices according to the indices of the word2vec model
            src_idx = []
            trg_idx = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                indices = []
                for uri in trg_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                trg_idx.append(torch.LongTensor(indices))
        return src_idx, trg_idx, trg_len
