import csv
import pickle

import gensim

OUTPUT_PATH = "/netscratch/kamke/dictionaries/"


def build_track_uri2id():
    print("build track_uri2id dict")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(track2vec.wv)):
        id = i
        uri = track2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'track_uri2id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_reducedTrackUri2reducedTrackID():
    print("build track_uri2id dict")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_reduced/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(track2vec.wv)):
        id = i
        uri = track2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'reduced_track_uri2reduced_id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_reducedArtistUri2reducedArtistID():
    print("build artist_uri2id dict")
    artist2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_artists_reduced/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(artist2vec.wv)):
        id = i
        uri = artist2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'reduced_artist_uri2reduced_id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_reducedAlbumUri2reducedAlbumID():
    print("build album_uri2id dict")
    album2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_albums_reduced/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(album2vec.wv)):
        id = i
        uri = album2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'reduced_album_uri2reduced_id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_id2track_uri():
    print("build track_uri2id dict")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(track2vec.wv)):
        id = i
        uri = track2vec.wv.index_to_key[id]
        dict[id] = uri
    print(len(dict))
    with open(OUTPUT_PATH + 'id2track_uri.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_track_id2reduced_track_id():
    print("build track_id 2 reduced_track_id")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    track2vec_reduced = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_reduced/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(track2vec.wv)):
        id = i
        uri = track2vec.wv.index_to_key[id]
        if uri in track2vec_reduced.wv.key_to_index:
            reduced_id = track2vec_reduced.wv.key_to_index[uri]
            dict[id] = reduced_id
    print(len(dict))
    with open(OUTPUT_PATH + 'trackid2reduced_trackid.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


DICT_PATH = '/netscratch/kamke/dictionaries/'


def build_track_id2reduced_artist_id():
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    artist2vec_reduced = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_artists_reduced/word2vec-song-vectors.model')
    artist2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_artists/word2vec-song-vectors.model')
    print("len(track2vec) = ", len(track2vec.wv))
    print("len(artist2vec_reduced) = ", len(artist2vec_reduced.wv))
    print("len(artist2vec) = ", len(artist2vec.wv))

    with open(DICT_PATH + 'trackid2artistid.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        my_dict = {}
        c = 0
        for i in range(len(track2vec.wv)):
            c += 1
            track_id = i
            artist_id = loaded_dict[track_id]
            artist_uri = artist2vec.wv.index_to_key[artist_id]
            if artist_uri in artist2vec_reduced.wv.key_to_index:
                reduced_artist_id = artist2vec_reduced.wv.key_to_index[artist_uri]
                my_dict[track_id] = reduced_artist_id
        print(len(my_dict))
        print("c = ", c)
        print("-----------")
        with open(OUTPUT_PATH + 'trackid2reduced_artistid.pkl', 'wb') as f1:
            pickle.dump(my_dict, f1)


def build_track_id2reduced_album_id():
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    album2vec_reduced = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_albums_reduced/word2vec-song-vectors.model')
    album2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_albums/word2vec-song-vectors.model')
    with open(DICT_PATH + 'trackid2albumid.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        dict = {}
        for i in range(len(track2vec.wv)):
            track_id = i
            album_id = loaded_dict[track_id]
            album_uri = album2vec.wv.index_to_key[album_id]
            if album_uri in album2vec_reduced.wv.key_to_index:
                reduced_album_id = album2vec_reduced.wv.key_to_index[album_uri]
                if track_id not in dict:
                    dict[track_id] = reduced_album_id
        print(len(dict))
        with open(OUTPUT_PATH + 'trackid2reduced_albumid.pkl', 'wb') as f1:
            pickle.dump(dict, f1)


def build_artist_uri2id():
    print("build artist_uri2id dict")
    word2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_artists/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(word2vec.wv)):
        id = i
        uri = word2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'artist_uri2id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_album_uri2id():
    print("build artist_uri2id dict")
    word2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_albums/word2vec-song-vectors.model')
    dict = {}
    for i in range(len(word2vec.wv)):
        id = i
        uri = word2vec.wv.index_to_key[id]
        dict[uri] = id
    print(len(dict))
    with open(OUTPUT_PATH + 'albums_uri2id.pkl', 'wb') as f:
        pickle.dump(dict, f)
    print("finished")


def build_trackid2artistid():
    print("build trackid2artistid")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    artist2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_artists/word2vec-song-vectors.model')
    with open('/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/track_artist_dict_unique.csv',
              encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
        track_artist_dict = {}
        for index, row in enumerate(csv_reader):
            if row[0] not in track_artist_dict:
                track_id = track2vec.wv.get_index(row[0])
                artist_id = artist2vec.wv.get_index(row[1])
                track_artist_dict[track_id] = artist_id
        print(len(track_artist_dict))
    with open(OUTPUT_PATH + 'trackid2artistid.pkl', 'wb') as f:
        pickle.dump(track_artist_dict, f)
    print("finished")


def build_trackid2albumid():
    print("build trackid2albumid")
    track2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists/word2vec-song-vectors.model')
    album2vec = gensim.models.Word2Vec.load(
        '/netscratch/kamke/models/word2vec/1_mil_playlists_albums/word2vec-song-vectors.model')
    with open('/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/track_album_dict_unique.csv',
              encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create dictionary of track_uri -> artist_uri
        track_album_dict = {}
        for index, row in enumerate(csv_reader):
            if row[0] not in track_album_dict:
                track_id = track2vec.wv.get_index(row[0])
                artist_id = album2vec.wv.get_index(row[1])
                track_album_dict[track_id] = artist_id
        print(len(track_album_dict))
    with open(OUTPUT_PATH + 'trackid2albumid.pkl', 'wb') as f:
        pickle.dump(track_album_dict, f)
    print("finished")


# track_all_id2artist_id
# track_all_id2album_id
if __name__ == "__main__":
    """build_trackid2artistid()
    build_trackid2albumid()"""

    build_track_id2reduced_artist_id()
    build_track_id2reduced_album_id()


