import gensim
import pandas as pd
import os.path
import numpy as np
from os import listdir
from os import path
import argparse
import json
import csv


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # if trained model exits then load model else train and safe model
    model = None
    if os.path.isfile("models/gensim_word2vec/10_thousand_playlists_artists/word2vec-song-vectors.model"):
        print("load model from file")
        model = gensim.models.Word2Vec.load("./models/gensim_word2vec/10_thousand_playlists_artists/word2vec-song-vectors.model")
        print("model loaded from file")
    else:
        num_playlists_to_read = 10000
        print("read data from database")
        # make matrix with each row is a playlist(list of track_uri)
        playlists = []
        with open('data/spotify_million_playlist_dataset_csv/data/artist_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file
            for idx, row in enumerate(csv_reader):
                if idx >= num_playlists_to_read:
                    break
                playlists.append(row[2:])

        print("build vocabulary...")
        model = gensim.models.Word2Vec(window=30, min_count=1, workers=4)  # AMD Ryzen 5 2600x with 6 cores
        model.build_vocab(playlists, progress_per=1000)
        print("builded vocabulary")
        print("Train model (this can take a lot of time)...")
        model.train(playlists, total_examples=model.corpus_count, epochs=model.epochs)
        # save model
        model.save("./models/gensim_word2vec/10_thousand_playlists_artists/word2vec-song-vectors.model")
        # model.wv.save_word2vec_format("./models/word2vec-song-vectors.model")
        print("trained and saved the model")
