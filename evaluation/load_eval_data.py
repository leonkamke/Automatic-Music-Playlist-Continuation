import torch
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import data_preprocessing.load_data as ld


# Dataset which creates src tracks with size size_seed and the corresponding targets
# uses the last 50.000 playlists
# cuts each playlist in the middle
class EvaluationDataset(Dataset):
    def __init__(self, word2vec_tracks, word2vec_artists, end_idx):
        # data loading
        self.word2vec_tracks = word2vec_tracks
        self.word2vec_artists = word2vec_artists
        self.end_idx = end_idx
        self.src, self.trg = self.read_train_data()
        # artist_dict: track_id -> artist_id
        self.artist_dict = self.init_artist_dict()
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
                if index < self.end_idx:
                    is_odd = len(row) % 2 == 1
                    i = int(len(row) / 2 + 1)
                    src_i = row[2:i]
                    trg_i = row[i:len(row)]
                    if is_odd:
                        trg_i = row[i:len(row) - 1]
                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
                if index > self.end_idx:
                    break
            # create lists of track indices according to the indices of the word2vec model
            src_idx = []
            trg_idx = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec_tracks.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                indices = []
                for uri in trg_uri[i]:
                    indices.append(self.word2vec_tracks.wv.get_index(uri))
                trg_idx.append(torch.LongTensor(indices))
        return src_idx, trg_idx

    def init_artist_dict(self):
        with open('data/spotify_million_playlist_dataset_csv/data/track_artist_dict_unique.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
            track_artist_dict = {}
            for index, row in enumerate(csv_reader):
                if row[0] not in track_artist_dict:
                    track_id = self.word2vec_tracks.wv.get_index(row[0])
                    artist_id = self.word2vec_artists.wv.get_index(row[1])
                    track_artist_dict[track_id] = artist_id
                print("line " + str(index) + " in track_artist_dict_unique.csv")
        return track_artist_dict


