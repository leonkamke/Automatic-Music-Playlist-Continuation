import gensim
import torch
import load_attributes as la
from autoencoder import Autoencoder
import track_embeddings
from evaluation import eval
from data_preprocessing import load_data as ld

import seq2seq_v3_track_album_artist


class Ensemble:
    def __init__(self, model_list, autoencoder, word2vec):
        self.model_list = model_list
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292
        self.track2vec = word2vec

    def predict(self, input, num_predictions):
        """# x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)"""

        rankings = torch.zeros(self.vocab_size, dtype=torch.float)
        # for each model make border-count

        for model in self.model_list:
            prediction = model.predict(input, num_predictions)
            for i, track_id in enumerate(prediction):
                track_id = int(track_id)
                rankings[track_id] += (num_predictions - i)

        _, top_k = torch.topk(rankings, dim=0, k=num_predictions)
        rankings = torch.zeros(self.vocab_size, dtype=torch.float)

        for track_id in top_k:
            track_id = int(track_id)
            track_uri = self.track2vec.wv.index_to_key[track_id]
            popularity = self.track2vec.wv.get_vecattr(track_uri, "count")
            rankings[track_id] = popularity

        _, top_k = torch.topk(rankings, dim=0, k=num_predictions)
        return top_k


if __name__ == "__main__":
    model_list = []

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
    print("loaded dictionaries from file")

    print("create word2vec model for ensemble")
    model_word2vec = track_embeddings.Word2VecModel(word2vec_tracks)
    model_list.append(model_word2vec)
    print("finished")
    # 15, 12
    print("create autoencoder for ensemble")
    NUM_TRACKS = len(reducedTrackUri2reducedId)
    NUM_ARTISTS = len(reducedArtistUri2reducedId)
    NUM_ALBUMS = len(reducedAlbumUri2reducedId)
    HID_DIM = 256

    save_file_name = "/autoencoder.pth"
    # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
    #                  reducedTrackId2trackId)
    autoencoder = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId,
                              trackId2reducedArtistId, trackId2reducedAlbumId, reduced_trackId2trackId)
    autoencoder.load_state_dict(torch.load(la.output_path_model() + la.get_folder_name() + save_file_name))
    model_list.append(autoencoder)
    print("created model")
    print("finished")

    print("model_list.len = ", len(model_list))

    # create ensemble model
    ensemble_model = Ensemble(model_list, autoencoder, word2vec_tracks)

    # evaluate ensemble model:
    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.evaluate_model(ensemble_model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), device)
