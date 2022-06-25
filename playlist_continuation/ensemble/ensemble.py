import gensim
import torch
from playlist_continuation.config import load_attributes as la
from playlist_continuation.recommender.autoencoder import Autoencoder
from playlist_continuation.evaluation import eval
from playlist_continuation.data_preprocessing import load_data as ld
import torch.nn as nn
from playlist_continuation.recommender.seq2seq_v4_nll_reduced import Seq2Seq
from playlist_continuation.recommender.title2rec import Title2Rec


class Ensemble:
    def __init__(self, track2vec, device):
        self.track2vec = track2vec
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292

        # -----------------------------------------------------------------------------------------------
        model_list = []

        self.device = device

        print("load dictionaries from file")
        reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()
        reducedArtistUri2reducedId = ld.get_reducedArtistUri2reducedArtistID()
        reducedAlbumUri2reducedId = ld.get_reducedAlbumUri2reducedAlbumID()
        reduced_trackId2trackId = ld.get_reduced_trackid2trackid()
        trackId2reducedTrackId = ld.get_trackid2reduced_trackid()
        trackId2reducedArtistId = ld.get_trackid2reduced_artistid()
        trackId2reducedAlbumId = ld.get_trackid2reduced_albumid()
        print("loaded dictionaries from file")
        NUM_TRACKS = len(reducedTrackUri2reducedId)
        NUM_ARTISTS = len(reducedArtistUri2reducedId)
        NUM_ALBUMS = len(reducedAlbumUri2reducedId)

        print("create Title2Rec model for ensemble")
        NUM_CHARS = 43
        N_LAYERS = 1
        HID_DIM = 256
        model = Title2Rec(NUM_CHARS, NUM_TRACKS, HID_DIM, N_LAYERS, reduced_trackId2trackId)
        foldername = "/title2rec"
        save_file_name = "/seq2seq_v4_track_album_artist.pth"
        model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
        model.to(device)
        model.eval()
        self.title2rec = model
        print("finished")

        """
        print("create autoencoder_1 for ensemble")
        HID_DIM = 256
        save_file_name = "/autoencoder.pth"
        # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
        #                  reducedTrackId2trackId)
        autoencoder = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId,
                                  trackId2reducedArtistId, trackId2reducedAlbumId, reduced_trackId2trackId)
        autoencoder.load_state_dict(torch.load(la.output_path_model() + "/autoencoder_1" + save_file_name))
        autoencoder.to(device)
        autoencoder.eval()
        model_list.append(autoencoder)
        print("finished")

        print("create autoencoder_2 for ensemble")
        HID_DIM = 256
        save_file_name = "/autoencoder.pth"
        # (self, hid_dim, num_tracks, num_artists, num_albums, trackId2reducedTrackId, trackId2reducedArtistId,
        #                  reducedTrackId2trackId)
        autoencoder = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, trackId2reducedTrackId,
                                  trackId2reducedArtistId, trackId2reducedAlbumId, reduced_trackId2trackId)
        autoencoder.load_state_dict(torch.load(la.output_path_model() + "/autoencoder_2" + save_file_name))
        autoencoder.to(device)
        autoencoder.eval()
        model_list.append(autoencoder)
        print("finished")"""

        print("create seq2seq model for ensemble")
        weights_path = la.path_embedded_weights()
        seq2seq_path = la.output_path_model() + "/tracks2rec/seq2seq_v4_reduced_nll.pth"
        weights = torch.load(weights_path, map_location=device)
        # weights.shape == (2262292, 300)
        # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
        embedding_pre_trained = nn.Embedding.from_pretrained(weights)
        seq2seq = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1, 300)
        seq2seq.load_state_dict(torch.load(seq2seq_path))
        seq2seq.to(device)
        seq2seq.eval()
        model_list.append(seq2seq)
        print("finished")

        print("create seq2seq_5 model for ensemble")
        weights_path = la.path_embedded_weights()
        seq2seq_path = la.output_path_model() + "/tracks2rec_5/seq2seq_v4_reduced_nll.pth"
        weights = torch.load(weights_path, map_location=device)
        # weights.shape == (2262292, 300)
        # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
        embedding_pre_trained = nn.Embedding.from_pretrained(weights)
        seq2seq_5 = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1, 300)
        seq2seq_5.load_state_dict(torch.load(seq2seq_path))
        seq2seq_5.to(device)
        seq2seq_5.eval()
        model_list.append(seq2seq_5)
        print("finished")

        """print("create seq2seq_2 model for ensemble")
        weights_path = la.path_embedded_weights_tracks()
        seq2seq_path = la.output_path_model() + "/tracks2rec_2/seq2seq_v4_reduced_nll.pth"
        weights = torch.load(weights_path, map_location=device)
        # weights.shape == (2262292, 100)
        # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
        embedding_pre_trained = nn.Embedding.from_pretrained(weights)
        seq2seq_2 = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1, 100)
        seq2seq_2.load_state_dict(torch.load(seq2seq_path))
        seq2seq_2.to(device)
        seq2seq_2.eval()
        model_list.append(seq2seq_2)
        print("finished")"""

        """print("create seq2seq_3 model for ensemble")
        weights_path = la.path_embedded_weights_tracks()
        seq2seq_path = la.output_path_model() + "/tracks2rec_3/seq2seq_v4_reduced_nll.pth"
        weights = torch.load(weights_path, map_location=device)
        # weights.shape == (2262292, 100)
        # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
        embedding_pre_trained = nn.Embedding.from_pretrained(weights)
        seq2seq_3 = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1, 100)
        seq2seq_3.load_state_dict(torch.load(seq2seq_path))
        seq2seq_3.to(device)
        seq2seq_3.eval()
        model_list.append(seq2seq_3)
        print("finished")"""
        print("len(model_list) = ", len(model_list))
        self.model_list = model_list

    def predict(self, title, src, num_predictions, only_title=False):
        """# x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)"""
        if only_title:
            return self.title2rec.predict(title, num_predictions)
        else:
            rankings = torch.zeros(self.vocab_size, dtype=torch.float)
            # for each model make border-count
            n = num_predictions

            for model in self.model_list:
                prediction = model.predict(src, n)
                for i, track_id in enumerate(prediction):
                    track_id = int(track_id)
                    rankings[track_id] += (n - i)

            _, top_k = torch.topk(rankings, dim=0, k=num_predictions)
            return top_k


class EnsembleRecall:
    def __init__(self, track2vec, device):
        self.track2vec = track2vec
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292

        # -----------------------------------------------------------------------------------------------
        model_list = []

        print("load dictionaries from file")
        reducedTrackUri2reducedId = ld.get_reducedTrackUri2reducedTrackID()
        reducedArtistUri2reducedId = ld.get_reducedArtistUri2reducedArtistID()
        reducedAlbumUri2reducedId = ld.get_reducedAlbumUri2reducedAlbumID()
        reduced_trackId2trackId = ld.get_reduced_trackid2trackid()
        self.trackId2reducedTrackId = ld.get_trackid2reduced_trackid()
        trackId2reducedArtistId = ld.get_trackid2reduced_artistid()
        trackId2reducedAlbumId = ld.get_trackid2reduced_albumid()
        NUM_TRACKS = len(reducedTrackUri2reducedId)
        NUM_ARTISTS = len(reducedArtistUri2reducedId)
        NUM_ALBUMS = len(reducedAlbumUri2reducedId)
        print("loaded dictionaries from file")

        print("create Title2Rec model for ensemble")
        NUM_CHARS = 43
        N_LAYERS = 1
        HID_DIM = 256
        model = Title2Rec(NUM_CHARS, NUM_TRACKS, HID_DIM, N_LAYERS, reduced_trackId2trackId)
        foldername = "/title2rec"
        save_file_name = "/seq2seq_v4_track_album_artist.pth"
        model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
        model.to(device)
        model.eval()
        self.title2rec = model
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
        autoencoder = Autoencoder(HID_DIM, NUM_TRACKS, NUM_ARTISTS, NUM_ALBUMS, self.trackId2reducedTrackId,
                                  trackId2reducedArtistId, trackId2reducedAlbumId, reduced_trackId2trackId)
        autoencoder.load_state_dict(torch.load(la.output_path_model() + "/autoencoder_1" + save_file_name))
        autoencoder.to(device)
        autoencoder.eval()
        self.autoencoder = autoencoder
        print("finished")

        print("create seq2seq model for ensemble")
        weights_path = la.path_embedded_weights()
        seq2seq_path = la.output_path_model() + "/tracks2rec_5/seq2seq_v4_reduced_nll.pth"
        weights = torch.load(weights_path, map_location=device)
        # weights.shape == (2262292, 300)
        # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
        embedding_pre_trained = nn.Embedding.from_pretrained(weights)
        seq2seq = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, 256, 1, 300)
        seq2seq.load_state_dict(torch.load(seq2seq_path))
        seq2seq.to(device)
        seq2seq.eval()
        self.seq2seq = seq2seq
        print("finished")

    def predict(self, title, input, num_predictions, only_title=False):
        if only_title:
            return self.title2rec.predict(title, num_predictions)
        else:
            pred_autoencoder = self.autoencoder.predict(input, num_predictions)
            # pred_autoencoder = sequence of track id's

            pred_seq2seq, _ = self.seq2seq.forward(input)
            # pred_seq2seq.shape = (seq_len, num_tracks)
            pred_seq2seq = torch.mean(pred_seq2seq, dim=0)
            # pred_seq2seq.shape = (num_tracks)

            mydict = {}
            values = []
            for trackId in pred_autoencoder:
                trackId = int(trackId)
                reducedTrackId = self.trackId2reducedTrackId[trackId]
                value = -1.0 * float(pred_seq2seq[reducedTrackId])
                mydict[value] = trackId
                values.append(value)
            values = sorted(values)

            output = []
            for value in values:
                output.append(mydict[value])
            # output has to be a list of track_id's
            # outputs.shape == (num_predictions)"""
            return output


if __name__ == "__main__":
    device = torch.device("cuda")
    print("load word2vec models")
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("finished")
    # create ensemble model
    # ensemble_model = EnsembleRecall(word2vec_tracks)
    ensemble_model = Ensemble(word2vec_tracks, device)

    # evaluate ensemble model:
    trackId2artistId = ld.get_trackid2artistid()
    trackUri2trackId = ld.get_trackuri2id()
    artistUri2artistId = ld.get_artist_uri2id()
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.spotify_evaluation(ensemble_model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                          981000, 981500, device)
