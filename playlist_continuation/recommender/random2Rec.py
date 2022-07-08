import gensim
import numpy as np
import torch
import playlist_continuation.data_preprocessing.load_data as ld
from playlist_continuation.config import load_attributes as la
import playlist_continuation.evaluation.eval as eval
import random


# ----------------------------------------------------------------------------------------------------------------------
class Random2Rec:
    def __init__(self, word2vec_tracks):
        self.word2vec_tracks = word2vec_tracks

    # def predict(self, title, src, num_predictions, only_title=False):
    def predict(self, title, src, num_predictions, only_title=False):
        output_indices = []
        for _ in range(num_predictions):
            track_id = random.randint(0, len(self.word2vec_tracks.wv) - 1)
            output_indices.append(track_id)
        return output_indices


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # load dictionaries
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

    # create model
    print("create playlistVec2Rec model")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    model = Random2Rec(word2vec_tracks)
    print("finished")

    # evaluate word2vec model
    results_str = eval.spotify_evaluation(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                          la.get_start_idx(), la.get_end_idx(), torch.device("cuda"))
