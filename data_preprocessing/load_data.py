import torch
import csv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import load_attributes as la


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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
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


class NextTrackDatasetOnlyOneTarget(Dataset):
    def __init__(self, word2vec, num_rows_train):
        # data loading
        self.word2vec = word2vec
        self.num_rows_train = num_rows_train
        self.src, self.trg, self.src_len = self.read_train_data()
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.src_len[index]

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
            src_len = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                trg_index = self.word2vec.wv.get_index(trg_uri[i])
                trg_idx.append(trg_index)
                src_len.append(len(indices))
        return src_idx, trg_idx, src_len


class NextTrackDatasetOnlyOneTargetReduced(Dataset):
    def __init__(self, word2vec, word2vec_reduced, num_rows_train):
        # data loading
        self.word2vec = word2vec
        self.word2vec_reduced = word2vec_reduced
        self.num_rows_train = num_rows_train
        self.src, self.trg, self.src_len = self.read_train_data()
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.src_len[index]

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
            src_len = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                if trg_uri[i] in self.word2vec_reduced.wv:
                    print("key exists")
                    trg_index = self.word2vec_reduced.wv.get_index(trg_uri[i])
                else:
                    print("key don't exists")
                    trg_index = -1
                trg_idx.append(trg_index)
                src_len.append(len(indices))
        return src_idx, trg_idx, src_len


class NextTrackDatasetOnlyOneTargetReducedFixedSteps(Dataset):
    def __init__(self, word2vec, word2vec_reduced, num_rows_train, num_steps):
        # data loading
        self.word2vec = word2vec
        self.num_steps = num_steps
        self.word2vec_reduced = word2vec_reduced
        self.num_rows_train = num_rows_train
        self.src, self.trg, self.src_len = self.read_train_data()
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.src_len[index]

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
                if len(src_uri) >= self.num_rows_train:
                    break
                elif len(row) > 2 + self.num_steps:
                    src_i = row[2:2+self.num_steps]
                    trg_i = row[2+self.num_steps]
                    src_uri.append(src_i)
                    trg_uri.append(trg_i)
            # create lists of track indices according to the indices of the word2vec model
            src_idx = []
            trg_idx = []
            src_len = []
            for i in range(len(src_uri)):
                indices = []
                for uri in src_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                src_idx.append(torch.LongTensor(indices))
                if trg_uri[i] in self.word2vec_reduced.wv:
                    trg_index = self.word2vec_reduced.wv.get_index(trg_uri[i])
                else:
                    trg_index = -1
                trg_idx.append(trg_index)
                src_len.append(len(indices))
        return src_idx, trg_idx, src_len


class NextTrackDatasetShiftedTarget(Dataset):
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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                playlist_len = len(row) - 2
                if index >= self.num_rows_train:
                    break
                elif len(row) > 3:
                    src_i = row[2:playlist_len - 1]
                    trg_i = row[3:playlist_len]
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
                indices = []
                for uri in trg_uri[i]:
                    indices.append(self.word2vec.wv.get_index(uri))
                trg_idx.append(torch.LongTensor(indices))
            return src_idx, trg_idx


class NextTrackDatasetShiftedTargetReducedFixedStep(Dataset):
    def __init__(self, word2vec, word2vec_reduced, num_rows_train, num_steps):
        # data loading
        self.num_steps = num_steps
        self.word2vec = word2vec
        self.word2vec_reduced = word2vec_reduced
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
        with open(la.path_track_sequences_path(), encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file and create lists of track uri's
            for index, row in enumerate(csv_reader):
                playlist_len = len(row) - 2
                if len(src_uri) >= self.num_rows_train:
                    break
                elif len(row) > 2 + self.num_steps:
                    src_i = row[2:self.num_steps]
                    trg_i = row[3:self.num_steps+1]
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
                indices = []
                for uri in trg_uri[i]:
                    if uri in self.word2vec_reduced.wv.vocab:
                        indices.append(self.word2vec_reduced.wv.key_to_index[uri])
                    else:
                        indices.append(-1)
                trg_idx.append(torch.LongTensor(indices))
            return src_idx, trg_idx


def collate_fn_shifted_target(data):
    """
    args:       data: list of tuple (src, trg)
    return:     padded_src - Padded src, tensor of shape (batch_size, padded_length)
                length - Original length of each sequence(without padding) tensor of shape (batch_size)
                padded_trg - Padded trg, tensor of shape (batch_size, padded_length)
    """
    src, trg = zip(*data)
    src = pad_sequence(src, batch_first=True)
    trg = pad_sequence(trg, batch_first=True, padding_value=-1)
    return torch.LongTensor(src), torch.LongTensor(trg)


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


def collate_fn_next_track_one_target(data):
    """
    args:       data: list of tuple (src, trg)
    return:     padded_src - Padded src, tensor of shape (batch_size, padded_length)
                length - Original length of each sequence(without padding) tensor of shape (batch_size)
                padded_trg - Padded trg, tensor of shape (batch_size, padded_length)
    """
    src, trg, src_len = zip(*data)
    src = pad_sequence(src, batch_first=True)
    return src, torch.LongTensor(trg), src_len


# safes a tensor of shape (2262292, 200) in a file. concatenation of tracks and artists
def get_track_artist_vectors(word2vec_tracks, word2vec_artists):
    weights_tracks = torch.FloatTensor(word2vec_tracks.wv.get_normed_vectors())
    weights_artists = torch.FloatTensor(word2vec_artists.wv.get_normed_vectors())
    track_artist_dict = get_artist_dict(word2vec_tracks, word2vec_artists)
    # create tensor for returning the output
    output = torch.zeros((len(word2vec_tracks), 200), dtype=torch.float)

    for i in range(len(word2vec_tracks.wv)):
        track_vec = weights_tracks[i]
        artist_vec_id = track_artist_dict[i]
        artist_vec = weights_artists[artist_vec_id]
        track_artist_cat = torch.cat((track_vec, artist_vec), dim=0)
        output[i] = track_artist_cat
    # safe output in a file
    torch.save(output, 'track_artist_embed.pt')


# Returns the following dictionary: artist_dict: track_id -> artist_id
def get_artist_dict(word2vec_tracks, word2vec_artists):
    with open(la.path_track_artist_dict_unique(), encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
        track_artist_dict = {}
        for index, row in enumerate(csv_reader):
            if row[0] not in track_artist_dict:
                track_id = word2vec_tracks.wv.get_index(row[0])
                artist_id = word2vec_artists.wv.get_index(row[1])
                track_artist_dict[track_id] = artist_id
            print("line " + str(index) + " in track_artist_dict_unique.csv")
    return track_artist_dict


def get_reduced_to_normal_dict(word2vec_tracks_reduced, word2vec_tracks):
    output_dict = {}
    for i in range(len(word2vec_tracks_reduced.wv)):
        uri = word2vec_tracks_reduced.wv.index_to_key[i]
        index = word2vec_tracks.wv.key_to_index[uri]
        output_dict[i] = index
    return output_dict
