import gensim
import torch
import load_attributes as la
from data_preprocessing import load_data as ld
from evaluation import eval
from autoencoder import Autoencoder
from evaluation import load_eval_data as eval_data
from seq2seq_v4_nll_reduced import Seq2Seq
import torch.nn as nn
from ensemble import Ensemble


def trackIds2trackUris(track_ids, track2vec):
    track_uris = []
    for track_id in track_ids:
        track_id = int(track_id)
        track_uris.append(track2vec.wv.index_to_key[track_id])
    return track_uris


if __name__ == "__main__":
    device = torch.device("cpu")

    print("load word2vec models")
    # word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
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

    """print("create Autoencoder model...")
    save_file_name = "/autoencoder.pth"
    foldername = "/autoencoder_1"
    # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
    #                  reducedTrackId2trackId)
    model = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId, trackId2reducedArtistId,
                        trackId2reducedAlbumId, reduced_trackId2trackId)
    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    model.eval()
    print("created model")"""

    """print("create seq2seq model for ensemble")
    weights_path = la.path_embedded_weights()
    seq2seq_path = la.output_path_model() + "/tracks2rec/seq2seq_v4_reduced_nll.pth"
    weights = torch.load(weights_path, map_location=device)
    # weights.shape == (2262292, 300)
    # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    model = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1)
    model.load_state_dict(torch.load(seq2seq_path))
    model.eval()
    print("finished")

    print("create dataset")
    evaluation_dataset = eval_data.VisualizeDataset(trackUri2trackId, artistUri2artistId, 0, 100000)
    print("finished")"""

    """
    "name": "Bachata Playlist", 
            "collaborative": "false", 
            "pid": 720, 
            "modified_at": 1340150400, 
            "num_tracks": 51, 
            "num_albums": 46, 
            "num_followers": 2, 
            
    "name": "rap", 
            "collaborative": "false", 
            "pid": 840, 
            "modified_at": 1505692800, 
            "num_tracks": 117, 
            "num_albums": 52, 
            "num_followers": 3,  
            
    "name": "christmas", 
            "collaborative": "false", 
            "pid": 842, 
            "modified_at": 1479340800, 
            "num_tracks": 136, 
            "num_albums": 19, 
            
    "name": "Hits", 
            "collaborative": "false", 
            "pid": 37, 
            "modified_at": 1456790400, 
            "num_tracks": 72, 
            "num_albums": 69,
            
    "name": "SpRiNg BrEaK", 
            "collaborative": "false", 
            "pid": 541, 
            "modified_at": 1505088000, 
            "num_tracks": 33, 
            "num_albums": 28, 
            "num_followers": 1, 
            
    "name": "vibin'", 
            "collaborative": "false", 
            "pid": 542, 
            "modified_at": 1505520000, 
            "num_tracks": 68, 
            "num_albums": 63, 
            "num_followers": 1, 
    """
    model = Ensemble(word2vec_tracks)
    print("create dataset")
    evaluation_dataset = eval_data.VisualizeDataset(trackUri2trackId, artistUri2artistId, 0, 100000)
    print("finished")
    playlist_id = 542
    playlist_uris, playlist_ids, pid, playlist_name = evaluation_dataset[playlist_id]
    print("Playlist ID: ", pid)
    print("Playlist name: ", playlist_name)
    print("Length of the playlist: ", len(playlist_ids))
    num_predictions = 50
    recommendation_ids = model.predict(playlist_name, playlist_ids[0:-1], num_predictions)

    # print recommendations
    print(trackIds2trackUris(recommendation_ids, word2vec_tracks))

