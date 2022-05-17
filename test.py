import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import gensim
import sys
import load_attributes as la


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pre_trained_embedding, hid_dim, n_layers, dropout=0):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(100, hid_dim, n_layers, batch_first=True, dropout=dropout)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100), when batch_first=True
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True
        # x = self.fc_out(x)
        # returned_matrix.shape == (batch_size, seq_len, vocab_size)
        return self.fc_out(x)


# safes a tensor of shape (2262292, 200) in a file. concatenation of tracks and artists
def get_track_album_artist_vectors(word2vec_tracks, word2vec_albums, word2vec_artists):
    weights_tracks = torch.FloatTensor(word2vec_tracks.wv.get_normed_vectors())
    weights_artists = torch.FloatTensor(word2vec_artists.wv.get_normed_vectors())
    weights_albums = torch.FloatTensor(word2vec_albums.wv.get_normed_vectors())

    track_artist_dict = get_artist_dict(word2vec_tracks, word2vec_artists)
    track_album_dict = get_album_dict(word2vec_tracks, word2vec_albums)
    # create tensor for returning the output
    output = torch.zeros((len(word2vec_tracks.wv), 300), dtype=torch.float)

    for i in range(len(word2vec_tracks.wv)):
        track_vec = weights_tracks[i]
        # get artist_id and album_id
        artist_id = track_artist_dict[i]
        album_id = track_album_dict[i]
        # get artist_vec and album_vec
        artist_vec = weights_artists[artist_id]
        album_vec = weights_albums[album_id]
        track_artist_cat = torch.cat((track_vec, album_vec, artist_vec), dim=0)
        output[i] = track_artist_cat
        print("track_id: ", i)
        print("artist_id: ", artist_id)
        print("album_id: ", album_id)
        print("track_artist_cat.shape: ", track_artist_cat.shape)
        print("track_artist_cat: ", track_artist_cat)
    # safe output in a file
    torch.save(output, la.output_path_model() + "/track_album_artist_embed.pt")
    print("output.shape = ", output.shape)
    print(output)


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


# Returns the following dictionary: artist_dict: track_id -> album_id
def get_album_dict(word2vec_tracks, word2vec_albums):
    with open(la.path_track_album_dict_unique(), encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
        track_album_dict = {}
        for index, row in enumerate(csv_reader):
            if row[0] not in track_album_dict:
                track_id = word2vec_tracks.wv.get_index(row[0])
                album_id = word2vec_albums.wv.get_index(row[1])
                track_album_dict[track_id] = album_id
            print("line " + str(index) + " in track_album_dict_unique.csv")
    return track_album_dict


if __name__ == "__main__":
    """# test_batch.shape == (2, 3, 3)
    src = torch.FloatTensor([[[0.5, 0.5, 0.5],
                                     [0.3, 0.3, 0.3],
                                     [0.4, 0.4, 0.4]],
                                    [[0.1, 0.1, 0.1],
                                    [0.2, 0.2, 0.2],
                                   [0.9, 0.9, 0.9]]])

    rnn = nn.LSTM(3, 2, 1, batch_first=True)
    output, (hn, cn) = rnn(src)
    # output.shape == (2, 3, 2)
    fc_out = nn.Linear(2, 10)
    output = fc_out(output)
    # output.shape == (2, 3, 10)
    print(output.shape)"""

    """# IMPORTANT!!! This is how Cross entropy works!!!
    # batch_size = 2; trg_len = 3, output_dim = 6
    trg = torch.LongTensor([4, 2, 2, 0, -1, 1])
    output = torch.FloatTensor([[0, 0, 0, 0, 1, 0],
                                [1, 1, 10, 1, 1, 1],
                                [10, 10, 20, 10, 10, 10],
                                [1, -1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 10],
                                [0, 1, 0, 0, 0, 0]])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(output, trg)
    print(loss.item())"""

    """from torch.nn.utils.rnn import pad_sequence
    a = torch.LongTensor([1, 2])
    b = torch.LongTensor([1, 10, 4, 0])
    c = torch.LongTensor([1, 2, 3, 4, 3, 4, -1, -1])
    out = pad_sequence([a, b, c], batch_first=True)
    print(out.size())
    print(out)"""

    """model = gensim.models.Word2Vec.load("./models/gensim_word2vec/100_thousand_playlists/word2vec-song-vectors.model")
    model.wv.get_index('spotify:track:2xvWdq0XdHYATHsD3i5QMq')"""

    """x = [[[1, 2, 3],
          [2, 5, 2],
          [3, 4, 400],
          [0, -1, 1]]]
    x = torch.Tensor(x)
    y = torch.rand((5, 4, 3))
    v = torch.zeros((5, 3))
    print(v)
    y[:, 0, :] = v
    print(y)"""

    """x = torch.LongTensor([10, 2, 3, 8, 5])
    _, top_k = torch.topk(x, dim=0, k=3)
    print(top_k)"""

    """x = torch.Tensor([[1, 2, 3, 4],
                      [3, 6, 2, 7],
                      [1, 2, 2, 2]])
    x = torch.mean(x, dim=0)
    print(x)
    #output has to be of size one"""

    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print(len(word2vec_tracks.wv))
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    print(len(word2vec_artists.wv))
    word2vec_albums = gensim.models.Word2Vec.load(la.path_album_to_vec_model())
    print(len(word2vec_albums.wv))


    get_track_album_artist_vectors(word2vec_tracks, word2vec_albums, word2vec_artists)

    """embeddings = torch.load("models/pretrained_embedded_matrix/track_artist_embed.pt", map_location=torch.device("cuda"))
    index = word2vec_tracks.wv.get_index("spotify:track:1zCoCopxgQmozHBuuyfW2K")
    track_vec2 = word2vec_tracks.wv.get_vector("spotify:track:1zCoCopxgQmozHBuuyfW2K", norm=True)
    #track_vec2 = word2vec_tracks.wv.word_vec("")
    track_vec = embeddings[index][0:100]

    artist_vec = embeddings[index][100:]
    artist_vec2 = word2vec_artists.wv.get_vector("spotify:artist:6p5JxpTc7USNnBnLzctyd4", norm=True)
    print(artist_vec2)
    print(artist_vec)
    print(track_vec2)
    print(track_vec)"""

    """word2vec_tracks = gensim.models.Word2Vec()
    print(word2vec_tracks.epochs)
    print(word2vec_tracks.window)
    print(word2vec_tracks.min_alpha)
    print(word2vec_tracks.alpha)"""
    # standart configuration: lr (alpha) = 0.025, epochs = 5, window_size = 5, min_alpha = 0.0001

