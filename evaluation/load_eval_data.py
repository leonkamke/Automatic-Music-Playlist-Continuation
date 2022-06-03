import torch
import csv
from torch.utils.data import Dataset
import load_attributes as la


# Dataset which creates src tracks with size size_seed and the corresponding targets
# uses the last 50.000 playlists
# cuts each playlist in the middle
class EvaluationDataset(Dataset):
    def __init__(self, word2vec_tracks, word2vec_artists, start_idx, end_idx):
        # data loading
        self.word2vec_tracks = word2vec_tracks
        self.word2vec_artists = word2vec_artists
        self.start_idx = start_idx
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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if self.start_idx <= index < self.end_idx and len(row) >= 4:
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
        with open(la.path_track_artist_dict_unique(),
                  encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
            track_artist_dict = {}
            for index, row in enumerate(csv_reader):
                if row[0] not in track_artist_dict:
                    track_id = self.word2vec_tracks.wv.get_index(row[0])
                    artist_id = self.word2vec_artists.wv.get_index(row[1])
                    track_artist_dict[track_id] = artist_id
        return track_artist_dict


"""
Spotify Evaluation Dataset contains:
- first track (1000x)
- first 5 tracks (2000x)
- first 10 tracks (2000x)
- first 25 tracks (2000x)
- first 100 tracks (2000x)
"""


class SpotifyEvaluationDataset(Dataset):
    def __init__(self, word2vec_tracks, word2vec_artists, start_idx, end_idx):
        # data loading
        self.word2vec_tracks = word2vec_tracks
        self.word2vec_artists = word2vec_artists
        self.start_idx = start_idx
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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if self.start_idx <= index < self.end_idx and len(row) >= 10:
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
        with open(la.path_track_artist_dict_unique(),
                  encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
            track_artist_dict = {}
            for index, row in enumerate(csv_reader):
                if row[0] not in track_artist_dict:
                    track_id = self.word2vec_tracks.wv.get_index(row[0])
                    artist_id = self.word2vec_artists.wv.get_index(row[1])
                    track_artist_dict[track_id] = artist_id
        return track_artist_dict


class FirstFiveEvaluationDataset(Dataset):
    def __init__(self, word2vec_tracks, word2vec_artists, start_idx, end_idx):
        # data loading
        self.word2vec_tracks = word2vec_tracks
        self.word2vec_artists = word2vec_artists
        self.start_idx = start_idx
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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                if self.start_idx <= index < self.end_idx and len(row) > 10:
                    src_i = row[2:7]
                    trg_i = row[7:-1]
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
        with open(la.path_track_artist_dict_unique(),
                  encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
            track_artist_dict = {}
            for index, row in enumerate(csv_reader):
                if row[0] not in track_artist_dict:
                    track_id = self.word2vec_tracks.wv.get_index(row[0])
                    artist_id = self.word2vec_artists.wv.get_index(row[1])
                    track_artist_dict[track_id] = artist_id
        return track_artist_dict
