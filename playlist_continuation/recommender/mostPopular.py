import gensim
import numpy as np
import torch
import playlist_continuation.data_preprocessing.load_data as ld
from playlist_continuation.config import load_attributes as la
import playlist_continuation.evaluation.eval as eval


class MostPopularModel:
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
        output_keys = self.word2vec_tracks.wv.similar_by_vector(mean_vector, topn=2000)
        # compute the num_prediction popular tracks of output_keys
        popularity_vec = torch.zeros((len(self.word2vec_tracks.wv)))
        for uri in output_keys:
            uri_id = self.word2vec_tracks.wv.get_index(uri[0])
            popularity = word2vec_tracks.wv.get_vecattr(uri[0], "count")
            popularity_vec[uri_id] = popularity

        _, top_k = torch.topk(popularity_vec, dim=0, k=num_predictions)
        return top_k


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
    print("create mostPopular model")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    model = MostPopularModel(word2vec_tracks)
    print("finished")

    # evaluate word2vec model
    results_str = eval.evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), torch.device("cpu"))
