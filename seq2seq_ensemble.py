import gensim
import torch
import load_attributes as la
import seq2seq_v3
import torch.nn as nn


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

    device = torch.device(la.get_device())

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

    print("create first Seq2Seq model...")
    model_1 = seq2seq_v3.Seq2Seq(VOCAB_SIZE, embedding_pre_trained, HID_DIM, N_LAYERS).to(device)
    model_1.load_state_dict(torch.load(la.output_path_model() + '/seq2seq_v3.pth'))
    model_1.eval()
    model_list.append(model_1)
    print("finished")

    # create ensemble model
    ensemble_model = Ensemble(model_list)

    # evaluate ensemble model:
    word2vec_artists = gensim.models.Word2Vec.load(la.path_artist_to_vec_model())
    word2vec_tracks = gensim.models.Word2Vec.load(la.path_track_to_vec_model())
    eval.evaluate_model(ensemble_model, word2vec_tracks, word2vec_artists, la.get_start_idx(), la.get_end_idx(), device)
