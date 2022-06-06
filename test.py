import torch
import torch.nn as nn
import csv
import gensim
from data_preprocessing import load_data as ld
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

    """word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print(word2vec_tracks.corpus_count)
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    print(len(word2vec_artists.wv))
    word2vec_albums = gensim.models.Word2Vec.load(la.path_album_to_vec_model())
    print(len(word2vec_albums.wv))

    get_track_album_artist_vectors(word2vec_tracks, word2vec_albums, word2vec_artists)"""

    """word2vec_tracks = gensim.models.Word2Vec.load("models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model")
    print(len(word2vec_tracks.wv))
    word2vec_artists = gensim.models.Word2Vec.load("models/gensim_word2vec/1_mil_playlists_artists/word2vec-song-vectors.model")
    print(len(word2vec_artists.wv))
    word2vec_albums = gensim.models.Word2Vec.load("models/gensim_word2vec/1_mil_playlists_albums/word2vec-song-vectors.model")
    print(len(word2vec_albums.wv))"""

    """word2vec_tracks = gensim.models.Word2Vec.load("models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model")
    word2vec_tracks_reduced = gensim.models.Word2Vec.load("models/gensim_word2vec/1_mil_playlists_reduced/word2vec-song-vectors.model")
    print(len(word2vec_tracks.wv))
    print(len(word2vec_tracks_reduced.wv))

    id_dict = ld.get_reduced_to_normal_dict(word2vec_tracks_reduced, word2vec_tracks)

    id = 50000
    print(word2vec_tracks_reduced.wv.index_to_key[id])
    print(word2vec_tracks.wv.index_to_key[id_dict[id]])"""

    print("load dictionaries from file")
    reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()
    reducedArtistUri2reducedId = ld.get_reducedArtistUri2reducedArtistID()
    reducedAlbumUri2reducedId = ld.get_reducedAlbumUri2reducedAlbumID()

    reduced_trackId2trackId = ld.get_reduced_trackid2trackid()
    trackId2reducedTrackId = ld.get_trackid2reduced_trackid()
    trackId2reducedArtistId = ld.get_trackid2reduced_artistid()
    trackId2reducedAlbumId = ld.get_trackid2reduced_albumid()

    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    print("loaded dictionaries from file")

    # test the lengths of the dictionaries -------------------------------------------------------------------------
    print("reducedTrackUri2reducedId.len ", len(set(reducedTrackUri2reducedId.values())))
    print("reducedArtistUri2reducedId.len ", len(set(reducedArtistUri2reducedId.values())))
    print("reducedAlbumUri2reducedId.len ", len(set(reducedAlbumUri2reducedId.values())))

    print("reduced_trackId2trackId.len ", len(set(reduced_trackId2trackId.values())))
    print("trackId2reducedTrackId.len ", len(set(trackId2reducedTrackId.values())))
    print("trackId2reducedArtistId.len ", len(set(trackId2reducedArtistId.values())))
    print("trackId2reducedAlbumId.len ", len(set(trackId2reducedAlbumId.values())))

    print("trackId2artistId.len ", len(set(trackId2artistId.values())))
    print("trackUri2trackId.len ", len(set(trackUri2trackId.values())))
    print("artistUri2artistId.len ", len(set(artistUri2artistId.values())))

    # test the mapping of the dictionaries -------------------------------------------------------------------------
    """
    artist_name": "Ty Dolla $ign",
    "track_uri": "spotify:track:7t2bFihaDvhIrd2gn2CWJO",
    "artist_uri": "spotify:artist:7c0XG5cIJTrrAgEC3ULPiq",
    "album_uri": "spotify:album:3SHx7bBQFI4J8QRr6D5cOK"
    """
    track_uri = "spotify:track:7t2bFihaDvhIrd2gn2CWJO"
    artist_uri = "spotify:artist:7c0XG5cIJTrrAgEC3ULPiq"
    album_uri = "spotify:album:3SHx7bBQFI4J8QRr6D5cOK"

    # reducedTrackuri_2_id, reducedArtisturi_2_id
    # ==
    # tracks_reduced, artists_reduced
    print("load word2vec models")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    word2vec_tracks_reduced = gensim.models.Word2Vec.load(la.path_track_to_vec_reduced_model())
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    word2vec_artists_reduced = gensim.models.Word2Vec.load(la.path_artist_to_vec_reduced_model())
    print("finished")

    uri_id = word2vec_tracks_reduced.wv.get_index(track_uri)
    uri_id2 = reducedTrackUri2reducedId[track_uri]
    print(uri_id, uri_id2)

    artist_id = word2vec_artists_reduced.wv.get_index(artist_uri)
    artist_id2 = reducedArtistUri2reducedId[artist_uri]
    print(artist_id, artist_id2)

    track_id = 100
    track_uri = word2vec_tracks.wv.index_to_key[track_id]
    if track_uri in word2vec_tracks_reduced.wv.key_to_index:
        new_track_id1 = word2vec_tracks_reduced.wv.key_to_index[track_uri]
    new_track_id2 = trackId2reducedTrackId[track_id]
    print(new_track_id1, new_track_id2)

    # test map_sequence2vector ----------------------------------------------------------------------------------
    sequence = [41000, 50000, 4, 80, 234, 543, 8345, 77777, 7, 93]

    def map_sequence2vector_old(sequence):
        track2artist_dict = ld.get_artist_dict(word2vec_tracks, word2vec_artists)
        # input.shape == (seq_len)
        track_vector = torch.zeros(600000)
        artist_vector = torch.zeros(600000)
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        for track_id in sequence:
            track_uri = word2vec_tracks.wv.index_to_key[track_id]
            if track_uri in word2vec_tracks_reduced.wv.key_to_index:
                new_track_id = word2vec_tracks_reduced.wv.key_to_index[track_uri]
                track_vector[new_track_id] = 1

                artist_id = track2artist_dict[int(track_id)]
                artist_uri = word2vec_artists.wv.index_to_key[artist_id]
                if artist_uri in word2vec_artists_reduced.wv.key_to_index:
                    artist_id_reduced = word2vec_artists_reduced.wv.key_to_index[artist_uri]
                    artist_vector[artist_id_reduced] = 1
        return torch.cat((track_vector, artist_vector))

    def map_sequence2vector(sequence):
        # input.shape == (seq_len)
        track_vector = torch.zeros(600000)
        artist_vector = torch.zeros(600000)
        #album_vector = torch.zeros(self.num_albums)
        # map sequence to vector of 1s and 0s (vector.shape == (input_size))
        for track_id in sequence:
            if track_id in trackId2reducedTrackId:
                new_track_id = trackId2reducedTrackId[track_id]
                track_vector[int(new_track_id)] = 1
            if track_id in trackId2reducedArtistId:
                new_artist_id = trackId2reducedArtistId[track_id]
                artist_vector[int(new_artist_id)] = 1

        # return torch.cat((track_vector, artist_vector, album_vector))
        return torch.cat((track_vector, artist_vector))
    a = map_sequence2vector(sequence)
    print(torch.equal(map_sequence2vector_old(sequence), a))

    _, top_k = torch.topk(a, k=100)

    output = []
    for track_id in top_k:
        track_uri = word2vec_tracks_reduced.wv.index_to_key[track_id]
        new_track_id = word2vec_tracks.wv.key_to_index[track_uri]
        output.append(new_track_id)

    output1 = []
    for reduced_track_id in top_k:
        track_id = reduced_trackId2trackId[int(reduced_track_id)]
        output1.append(track_id)

    print(output == output1)

