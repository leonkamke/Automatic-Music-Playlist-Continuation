import gensim
import os.path
import csv


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # if trained model exits then load model else train and safe model
    model = None
    if os.path.isfile("models/gensim_word2vec/1_mil_playlists_albums/word2vec-song-vectors.model"):
        print("load model from file")
        model = gensim.models.Word2Vec.load("./models/gensim_word2vec/1_mil_playlists_albums/word2vec-song-vectors.model")
        print("model loaded from file")
    else:
        num_playlists_to_read = 1000000
        print("read data from database")
        # make matrix with each row is a playlist(list of track_uri)
        playlists = []
        with open('data/spotify_million_playlist_dataset_csv/data/album_sequences.csv', encoding='utf8') as read_obj:
            csv_reader = csv.reader(read_obj)
            # Iterate over each row in the csv file
            for idx, row in enumerate(csv_reader):
                print(idx)
                if idx >= num_playlists_to_read:
                    break
                playlists.append(row[2:])

        print("build vocabulary...")
        # standart configuration: lr (alpha) = 0.025, epochs = 5, window_size = 5, min_alpha = 0.0001
        model = gensim.models.Word2Vec()  # AMD Ryzen 5 2600x with 6 cores
        model.build_vocab(playlists, progress_per=1000)
        print("builded vocabulary")
        print("Train model (this can take a lot of time)...")
        model.train(playlists, total_examples=model.corpus_count, epochs=model.epochs)
        # save model
        model.save("./models/gensim_word2vec/1_mil_playlists_albums/word2vec-song-vectors.model")
        # model.wv.save_word2vec_format("./models/word2vec-song-vectors.model")
        print("trained and saved the model")
