import gensim
import numpy as np
import torch
import playlist_continuation.data_preprocessing.load_data as ld
from playlist_continuation.config import load_attributes as la
import playlist_continuation.evaluation.eval as eval


# --------------- some helper functions --------------------------------------------------------------------------------

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
    model = Word2VecModel(word2vec_tracks)
    print("finished")

    # evaluate word2vec model
    results_str = eval.evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), torch.device("cpu"))
