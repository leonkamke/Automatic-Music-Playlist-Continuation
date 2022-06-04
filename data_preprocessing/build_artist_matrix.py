import csv
import gensim
import torch

if __name__ == '__main__':
    print("load word2vec from file")
    word2vec_artists = gensim.models.Word2Vec.load("/netscratch/kamke/models/word2vec/1_mil_playlists_artists/word2vec-song-vectors.model")
    print("word2vec loaded from file")
    print(len(word2vec_artists.wv))
    artists_matrix = torch.zeros(1000000, len(word2vec_artists.wv))

    with open("/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/artist_sequences.csv", encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv-track_sequences file
        for i, row in enumerate(csv_reader):
            if len(row) >= 3:
                print("playlist: " + str(i))
                for artist_uri in range(2, len(row)):
                    # get index of spotify_artist_uri
                    artist_id = word2vec_artists.wv.get_index(artist_uri)
                    artists_matrix[i, artist_id] = 1

    # safe torch tensor in file
    torch.save(artists_matrix, "/ds/audio/MPD/mpd_matrices/artist_matrix.pt")


