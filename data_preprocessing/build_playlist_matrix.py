import csv
import gensim
import load_attributes as la
import torch

if __name__ == '__main__':
    print("load word2vec from file")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("word2vec loaded from file")
    print(len(word2vec_tracks.wv))
    playlist_matrix = torch.zeros(1000000, len(word2vec_tracks))

    with open("/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/track_sequences.csv", encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv-track_sequences file
        for i, row in enumerate(csv_reader):
            if len(row) >= 3:
                print("playlist: " + str(i))
                for uri in range(2, len(row)):
                    # get index of spotify_uri
                    track_id = word2vec_tracks.wv.get_index(uri)
                    playlist_matrix[i, track_id] = 1

    # safe torch tensor in file
    torch.save(playlist_matrix, "/ds/audio/MPD/mpd_matrices/playlist_matrix.pt")
