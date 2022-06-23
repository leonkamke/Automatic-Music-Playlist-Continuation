"""
training: train by computing the Cross Entropy Loss based on the shifted target (the output
            is shifted in one timestamp in comparison to the input)
prediction: do_rank (take k largest values (indices) for the prediction)
            do_mean_rank (take mean over all vector's correlating to each timestamp and then take the k
                            largest values (indices) for the prediction)
"""
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import os
import playlist_continuation.data_preprocessing.load_data as ld
from torch.utils.data import DataLoader
import playlist_continuation.evaluation.eval as eval
from playlist_continuation.config import load_attributes as la


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2Seq(nn.Module):
    def __init__(self, reducedTrackId2trackId, vocab_size, pre_trained_embedding, hid_dim, n_layers, embedded_dim,
                 dropout=0):
        super(Seq2Seq, self).__init__()
        self.reducedTrackId2trackId = reducedTrackId2trackId
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # input shape of embedding: (*) containing the indices
        # output shape of embedding: (*, embed_dim == 100)
        self.embedding = pre_trained_embedding
        # input shape of LSTM has to be (batch_size, seq_len, embed_dim == 100) when batch_first=True
        # output shape of LSTM: output.shape == (batch_size, seq_len, hid_dim)  when batch_first=True
        #                       h_n.shape == (n_layers, batch_size, hid_dim)
        #                       c_n.shape == (n_layers, batch_size, hid_dim)
        self.rnn = nn.LSTM(embedded_dim, hid_dim, n_layers, batch_first=True, dropout=0.2)
        # input shape of Linear: (*, hid_dim)
        # output shape of Linear: (*, vocab_size)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        # input.shape == (batch_size, seq_len)
        x = self.embedding(input)
        # x.shape == (batch_size, seq_len, embed_dim == 100)
        x, (h_n, c_n) = self.rnn(x)
        # x.shape == (batch_size, seq_len, hid_dim), when batch_first=True

        x = self.fc_out(x)
        # x.shape == (batch_size, seq_len, vocab_size)

        x = self.log_softmax(x)

        return x, (h_n, c_n)

    def predict(self, input, num_predictions):
        # input = torch.LongTensor(input)
        # input.shape == seq_len
        x, _ = self.forward(input)
        # x.shape == (seq_len, num_tracks)
        x = x[-1]
        # x.shape == (num_tracks)
        _, top_k = torch.topk(x, dim=0, k=num_predictions, largest=True)
        # top_k.shape == (num_predictions)
        output = []
        for reduced_track_id in top_k:
            track_id = self.reducedTrackId2trackId[int(reduced_track_id)]
            output.append(track_id)
        # output has to be a list of track_id's
        # outputs.shape == (num_predictions)
        return output

    def predict_do_summed_rank(self, input, num_predictions):
        # input.shape == seq_len
        x, _ = self.forward(input)
        # x.shape == (seq_len, vocab_size)
        x = torch.mean(x, dim=0)
        # x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)
        return top_k

    def predict_(self, input, num_predictions):
        # input.shape == seq_len
        outputs = torch.zeros(num_predictions)
        outputs_vectors = torch.zeros(num_predictions, self.vocab_size)
        # outputs.shape = (num_predictions)
        x, (h_n, c_n) = self.forward(input)
        # x.shape == (seq_len, vocab_size)
        idx = torch.argmax(x[-1])
        outputs[0] = idx
        outputs_vectors[0] = x[-1]

        # idx.shape == (1)
        for i in range(1, num_predictions):
            x = self.embedding(idx)
            x = torch.unsqueeze(x, dim=0)
            # x.shape == (1, embed_dim == 300)
            x, (h_n, c_n) = self.rnn(x, (h_n, c_n))
            # x.shape == (1, hid_dim)
            x = self.fc_out(x)
            # x.shape == (1, vocab_size)
            outputs_vectors[i] = x[0]
            idx = torch.argmax(x[0])
            outputs[i] = idx
        # outputs.shape == (num_predictions)
        return outputs


def train_shifted_target(model, dataloader, optimizer, criterion, device, num_epochs, max_norm):
    model.train()
    num_iterations = 1
    for epoch in range(num_epochs):
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            # trg.shape = src.shape = (batch_size, seq_len)
            optimizer.zero_grad()
            output, _ = model(src)
            output = output.permute(0, 2, 1)
            # output.shape = (batch_size, num_steps, num_tracks)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            print("epoch ", epoch+1, " iteration ", num_iterations, " loss = ", loss.item())
            num_iterations += 1
        num_iterations = 1


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device(la.get_device())

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

    print("load pretrained embedding layer...")
    weights = torch.load(la.path_embedded_weights(), map_location=device)
    # weights.shape == (2262292, 300)
    # pre_trained embedding reduces the number of trainable parameters from 34 mill to 17 mill
    embedding_pre_trained = nn.Embedding.from_pretrained(weights)
    print("finished")

    # Training and model parameters
    learning_rate = la.get_learning_rate()
    num_epochs = la.get_num_epochs()
    batch_size = la.get_batch_size()
    num_playlists_for_training = la.get_num_playlists_training()
    # VOCAB_SIZE == 2262292
    NUM_TRACKS = len(reducedTrackUri2reducedId)
    HID_DIM = la.get_recurrent_dimension()
    N_LAYERS = la.get_num_recurrent_layers()
    num_steps = 20
    max_norm = 5

    print("create Seq2Seq model...")
    model = Seq2Seq(reduced_trackId2trackId, NUM_TRACKS, embedding_pre_trained, HID_DIM, N_LAYERS)
    model.to(device)
    print("finished")

    print("init weights...")
    model.apply(init_weights)
    print("finished")

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("The size of the vocabulary is: ", NUM_TRACKS)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    # optimizer = optim.SGD(model.parameters(), learning_rate)
    criterion = nn.NLLLoss(ignore_index=-1)

    print("Create train data...")
    # dataset = (self, trackUri2trackId, reducedTrackUri2reducedTrackId, num_rows_train, num_steps):
    #dataset = ld.NextTrackDatasetShiftedTargetReducedFixedStep(trackUri2trackId, reducedTrackUri2reducedId,
    #                                                           num_playlists_for_training, num_steps)
    # = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Created train data")

    foldername = la.get_folder_name()
    save_file_name = "/seq2seq_v4_reduced_nll.pth"

    #os.mkdir(la.output_path_model() + foldername)
    #shutil.copyfile("../config/attributes", la.output_path_model() + foldername + "/attributes.txt")
    # def train(model, src, trg, optimizer, criterion, device, batch_size=10, clip=1, epochs=2)
    #train_shifted_target(model, dataloader, optimizer, criterion, device, num_epochs, max_norm)
    #torch.save(model.state_dict(), la.output_path_model() + foldername + save_file_name)

    model.load_state_dict(torch.load(la.output_path_model() + foldername + save_file_name))
    device = torch.device("cpu")
    model.to(device)
    # evaluate model:
    model.eval()
    # def evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId, start_idx, end_idx, device)
    results_str = eval.evaluate_model(model, trackId2artistId, trackUri2trackId, artistUri2artistId,
                                      la.get_start_idx(), la.get_end_idx(), device)

    # write results in a file with setted attributes
    f = open(la.output_path_model() + foldername + "/results.txt", "w")
    f.write(results_str)
    f.write("\nseq2seq_v4_nlll_reduced: ")
    f.close()
