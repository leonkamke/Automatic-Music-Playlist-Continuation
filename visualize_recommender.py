import gensim
import torch
import load_attributes as la
from data_preprocessing import load_data as ld
from evaluation import eval
from autoencoder import Autoencoder
from evaluation import load_eval_data as eval_data

if __name__ == "__main__":
    device = torch.device("cpu")

    print("load word2vec models")
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("finished")

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

    NUM_TRACKS = len(reducedTrackUri2reducedId)
    NUM_ARTISTS = len(reducedArtistUri2reducedId)
    NUM_ALBUMS = len(reducedAlbumUri2reducedId)
    HID_DIM = 256

    print("create Autoencoder model...")
    # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
    #                  reducedTrackId2trackId)
    model = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId, trackId2reducedArtistId,
                        reduced_trackId2trackId)
    print("finished")

    print("create dataset")
    evaluation_dataset = eval_data.SpotifyEvaluationDataset(trackUri2trackId, artistUri2artistId, 0, 100000)
    print("finished")

    src, trg, pid = evaluation_dataset[4877]
    print(pid)
    print(len(evaluation_dataset))
