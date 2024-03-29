"""
Create and train Track2Vec model with Gensim
"""

import gensim
import os.path
import numpy as np
from os import path
import argparse
import json
import csv
from playlist_continuation.config import load_attributes as la


# --------------- some helper functions --------------------------------------------------------------------------------
# returns a numpy array with the track_uris from each track in playlist pid; for slice 198000-198999
def get_track_uris_from_playlist(playlist_id):
    data_path = '../../data/spotify_million_playlist_dataset/data'
    parser = argparse.ArgumentParser(description="get playlist")
    parser.add_argument(data_path, default=None)
    # print(listdir(data_path))
    file = path.join(data_path, 'mpd.slice.8000-8999.json')
    playlist_uris = []
    with open(file) as json_file:
        json_slice = json.load(json_file)  # dict
        playlist_obj = json_slice['playlists'][playlist_id - 8000]  # dict
        for track in playlist_obj['tracks']:
            playlist_uris.append(track['track_uri'])
    return np.array(playlist_uris)


# returns the mean vector for a given list of track_uris and a given Word2Vec model
def calc_mean_vector(model, track_uris):
    vec = []
    for track in track_uris:
        vec.append(model.wv.get_vector(track))
    return np.mean(vec, axis=0)


# ----------------------------------------------------------------------------------------------------------------------
class Word2VecModel:
    def __init__(self, word2vec_tracks):
        self.word2vec_tracks = word2vec_tracks

    def predict(self, input, num_predictions):
        # calculate mean-vector of given tracks
        vec = []
        for track_id in input:
            track_id = int(track_id)
            track_vector = self.word2vec_tracks.wv[track_id]
            vec.append(track_vector)
        mean_vector = np.mean(vec, axis=0)
        # get similar tracks
        output_keys = self.word2vec_tracks.wv.similar_by_vector(mean_vector, topn=num_predictions)
        # convert the list of keys (output) to a list of indices
        output_indices = []
        for key in output_keys:
            # type(key) = (track_uri, percent)
            output_indices.append(self.word2vec_tracks.wv.get_index(key[0]))
        return output_indices


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # if trained model exits then load model else train and safe model
    word2vec_tracks = None
    if os.path.isfile("../../models/gensim_word2vec/1_mil_playlists/word2vec-song-vectors.model"):
        print("load model from file")
        word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
        print("model loaded from file")
    else:
        num_playlists_to_read = 1000000
        print("read data from database")
        # make matrix with each row is a playlist(list of track_uri)
        playlists = []
        with open('../../data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file
            for idx, row in enumerate(csv_reader):
                if idx >= num_playlists_to_read:
                    break
                # playlists.append(list(np.unique(row[2:])))
                playlists.append(row[2:])
                print(idx)
        print("build vocabulary...")
        # standart configuration: lr (alpha) = 0.025, epochs = 5, window_size = 5, min_alpha = 0.0001
        # mincount = all words/tracks which appear fewer than this number are not handled
        word2vec_tracks = gensim.models.Word2Vec(min_count=6)  # AMD Ryzen 5 2600x with 6 cores
        word2vec_tracks.build_vocab(playlists, progress_per=1000)
        print("builded vocabulary")
        print("Train model (this can take a lot of time)...")
        word2vec_tracks.train(playlists, total_examples=word2vec_tracks.corpus_count, epochs=word2vec_tracks.epochs)
        # save model
        word2vec_tracks.save("./models/gensim_word2vec/1_mil_playlists_reduced/word2vec-song-vectors.model")
        # model.wv.save_word2vec_format("./models/word2vec-song-vectors.model")
        print("trained and saved the model")
