import gensim
import torch
import load_attributes as la
import seq2seq_v3
import torch.nn as nn

import track_embeddings
from evaluation import eval

import seq2seq_v3_track_album_artist


class Ensemble:
    def __init__(self, model_list):
        self.model_list = model_list
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292

    def predict(self, input, num_predictions):
        """# x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)"""
        rankings = torch.zeros(self.vocab_size, dtype=torch.float)
        # for each model make border-count
        for model in self.model_list:
            prediction = model.predict(input, num_predictions)
            for i, track_id in enumerate(prediction):
                rankings[track_id] += (num_predictions - i)
        _, top_k = torch.topk(rankings, dim=0, k=num_predictions)
        return top_k


if __name__ == "__main__":
    model_list = []

    device = torch.device("cpu")

    print("load pretrained embedding layer")
    weights = torch.load(la.path_embedded_weights(), map_location=device)
    # weights.shape == (2262292, 200)
    # pre_trained embedding reduces the number of trainable parameters from  454 million to 228,572,292
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    print("finished")

    # model parameters
    # VOCAB_SIZE == 2262292
    VOCAB_SIZE = 2262292
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()

    filename = "/seq2seq_v3_track_album_artist.pth"

    print("load word2vec models")
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    print("finished")

    print("create first Seq2Seq model...")
    model_1 = seq2seq_v3_track_album_artist.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_1.load_state_dict(torch.load(la.output_path_model() + "/seq2seq_v3_track_album_artist_2" + filename))
    model_1.eval()
    model_list.append(model_1)
    print("finished")

    print("create word2vec model")
    model_2 = track_embeddings.Word2VecModel(word2vec_tracks)
    model_list.append(model_2)

    """print("create second Seq2Seq model...")
    model_2 = seq2seq_v3_track_album_artist.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_2.load_state_dict(torch.load(la.output_path_model() + "/seq2seq_v3_track_album_artist_2" + filename))
    model_2.eval()
    model_list.append(model_2)
    print("finished")

    print("create third Seq2Seq model...")
    model_3 = seq2seq_v3_track_album_artist.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_3.load_state_dict(torch.load(la.output_path_model() + "/seq2seq_v3_track_album_artist_3" + filename))
    model_3.eval()
    model_list.append(model_3)
    print("finished")

    print("create fourth Seq2Seq model...")
    model_4 = seq2seq_v3_track_album_artist.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_4.load_state_dict(torch.load(la.output_path_model() + "/seq2seq_v3_track_album_artist_4" + filename))
    model_4.eval()
    model_list.append(model_4)
    print("finished")"""

    print("create fifth Seq2Seq model...")
    model_5 = seq2seq_v3_track_album_artist.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_5.load_state_dict(torch.load(la.output_path_model() + "/seq2seq_v3_track_album_artist_5" + filename))
    model_5.eval()
    model_list.append(model_5)
    print("finished")

    # create ensemble model
    ensemble_model = Ensemble(model_list)

    # evaluate ensemble model:
    eval.evaluate_model(ensemble_model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(), device)
