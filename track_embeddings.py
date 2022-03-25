import gensim
import pandas as pd
import os.path
import numpy as np
from os import listdir
from os import path
import argparse
import json
import csv


# --------------- some helper functions --------------------------------------------------------------------------------
# returns a numpy array with the track_uris from each track in playlist pid; for slice 198000-198999
def get_track_uris_from_playlist(playlist_id):
    data_path = './data/spotify_million_playlist_dataset/data'
    parser = argparse.ArgumentParser(description="get playlist")
    parser.add_argument(data_path, default=None)
    # print(listdir(data_path))
    file = path.join(data_path, 'mpd.slice.198000-198999.json')
    playlist_uris = []
    with open(file) as json_file:
        json_slice = json.load(json_file)   # dict
        playlist_obj = json_slice['playlists'][playlist_id-198000]  # dict
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

if __name__ == '__main__':
    # if trained model exits then load model else train and safe model
    model = None
    if os.path.isfile("models/gensim_word2vec/word2vec-song-vectors.model"):
        print("load model from file")
        model = gensim.models.Word2Vec.load("./models/gensim_word2vec/word2vec-song-vectors.model")
        print("model loaded from file")
    else:
        num_playlists_to_read = 10000
        print("read data from database")
        # make matrix with each row is a playlist(list of track_uri)
        playlists = []
        with open('data/spotify_million_playlist_dataset_csv/data/track_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file
            for idx, row in enumerate(csv_reader):
                if idx >= num_playlists_to_read:
                    break
                playlists.append(row[2:])
        print("build vocabulary and train model...")
        model = gensim.models.Word2Vec(window=30, min_count=1, workers=4)  # AMD Ryzen 5 2600x with 6 cores
        model.build_vocab(playlists, progress_per=1000)
        model.train(playlists, total_examples=model.corpus_count, epochs=model.epochs)
        # save model
        model.save("./models/gensim_word2vec/word2vec-song-vectors.model")
        # model.wv.save_word2vec_format("./models/word2vec-song-vectors.model")
        print("trained and saved model")


    '''some playlists:
    old bangers 198000
    brunch 198154'''
    print(model.wv.get_index("spotify:track:1KHdq8NK9QxnGjdXb55NiG"))

    """track_uris = get_track_uris_from_playlist(198087)
    x = calc_mean_vector(model, track_uris)
    print(model.wv.similar_by_vector(x, topn=500))
    # print(model.wv.similar_by_key('spotify:track:0muI8DpTEpLqqibPm3sKYf'))"""
